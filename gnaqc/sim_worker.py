"""Persistent subprocess worker for Aer tensor-network GPU simulation.

Why: cuTensorNet's contractor autotuner occasionally picks a contraction path
that runs for many minutes while holding the Python GIL. This makes every
in-process timeout mechanism (threading, SIGALRM) unable to interrupt it —
signal handlers only run at GIL release points.

Isolating the simulator call in a subprocess lets us kill -9 it on timeout,
bypassing the GIL entirely. A persistent worker amortizes the Aer/cuQuantum/
CUDA init cost across episodes (~5-10s one-time startup).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
from typing import Any

from qiskit import QuantumCircuit

logger = logging.getLogger(__name__)


def _worker_target(req_q, resp_q, sim_config: dict):
    """Subprocess entry: loop serving (circuit, noise_model, shots) requests."""
    # Import inside the child so its CUDA context is independent.
    from qiskit_aer import AerSimulator

    while True:
        msg = req_q.get()
        if msg is None:
            return
        req_id, circuit, noise_model, shots = msg
        try:
            sim = AerSimulator(noise_model=noise_model, **sim_config)
            counts = sim.run(circuit, shots=shots).result().get_counts()
            resp_q.put((req_id, "ok", counts))
        except Exception as e:
            resp_q.put((req_id, "err", f"{type(e).__name__}: {str(e)[:300]}"))


class SimWorkerTimeout(TimeoutError):
    """Raised when the worker subprocess did not respond within the budget."""


class SimWorker:
    """Persistent subprocess that runs AerSimulator calls with OS-level timeout.

    Usage:
        w = SimWorker()
        counts = w.run(compiled_circuit, noise_model, shots=1000, timeout_s=120)
        w.shutdown()

    On timeout the worker process is terminate/kill'd and a fresh one is spawned.
    The next `run()` call will incur one-time CUDA re-initialization (~5-10s).
    """

    def __init__(self, sim_config: dict | None = None):
        self.sim_config = sim_config or {"method": "tensor_network", "device": "GPU"}
        self._ctx = mp.get_context("spawn")
        self._req_id = 0
        self._spawn()

    def _spawn(self) -> None:
        self._req_q = self._ctx.Queue()
        self._resp_q = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_worker_target,
            args=(self._req_q, self._resp_q, self.sim_config),
            daemon=True,
        )
        self._proc.start()

    def _respawn(self) -> None:
        logger.warning("SimWorker: respawning after timeout/kill")
        try:
            self._proc.terminate()
            self._proc.join(5)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join()
        except Exception:
            pass
        self._spawn()

    def run(
        self,
        circuit: QuantumCircuit,
        noise_model: Any,
        shots: int,
        timeout_s: float,
    ) -> dict[str, int]:
        """Submit a simulation and wait up to `timeout_s` for the result.

        Raises SimWorkerTimeout if the budget is exceeded; the worker is killed
        and respawned so the next call gets a clean slate.
        Raises RuntimeError if the worker reported a simulation error.
        """
        if not self._proc.is_alive():
            self._spawn()

        self._req_id += 1
        rid = self._req_id
        self._req_q.put((rid, circuit, noise_model, shots))

        try:
            got_rid, status, payload = self._resp_q.get(timeout=timeout_s)
        except queue.Empty:
            self._respawn()
            raise SimWorkerTimeout(f"worker exceeded {timeout_s}s")

        if status == "ok":
            return payload
        raise RuntimeError(payload)

    def shutdown(self) -> None:
        try:
            self._req_q.put(None)
            self._proc.join(5)
        except Exception:
            pass
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join()
