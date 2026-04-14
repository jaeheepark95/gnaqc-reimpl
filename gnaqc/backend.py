"""Backend utilities for GNAQC.

Provides backend loading and native 2Q gate detection for Qiskit FakeBackendV2.
Self-contained — no external project dependencies.
"""

from __future__ import annotations

from typing import Any

# Registry: short_name -> FakeBackendV2 class name in qiskit_ibm_runtime.fake_provider
BACKEND_REGISTRY: dict[str, str] = {
    # --- 5Q ---
    "athens": "FakeAthensV2",
    "belem": "FakeBelemV2",
    "bogota": "FakeBogotaV2",
    "burlington": "FakeBurlingtonV2",
    "essex": "FakeEssexV2",
    "lima": "FakeLimaV2",
    "london": "FakeLondonV2",
    "manila": "FakeManilaV2",
    "ourense": "FakeOurenseV2",
    "quito": "FakeQuitoV2",
    "rome": "FakeRomeV2",
    "santiago": "FakeSantiagoV2",
    "valencia": "FakeValenciaV2",
    "vigo": "FakeVigoV2",
    "yorktown": "FakeYorktownV2",
    # --- 7Q ---
    "casablanca": "FakeCasablancaV2",
    "jakarta": "FakeJakartaV2",
    "lagos": "FakeLagosV2",
    "nairobi": "FakeNairobiV2",
    "oslo": "FakeOslo",
    "perth": "FakePerth",
    # --- 15-16Q ---
    "melbourne": "FakeMelbourneV2",
    "guadalupe": "FakeGuadalupeV2",
    # --- 20Q ---
    "almaden": "FakeAlmadenV2",
    "boeblingen": "FakeBoeblingenV2",
    "johannesburg": "FakeJohannesburgV2",
    "poughkeepsie": "FakePoughkeepsieV2",
    "singapore": "FakeSingaporeV2",
    # --- 27-28Q ---
    "algiers": "FakeAlgiers",
    "auckland": "FakeAuckland",
    "cairo": "FakeCairoV2",
    "cambridge": "FakeCambridgeV2",
    "geneva": "FakeGeneva",
    "hanoi": "FakeHanoiV2",
    "kolkata": "FakeKolkataV2",
    "montreal": "FakeMontrealV2",
    "mumbai": "FakeMumbaiV2",
    "paris": "FakeParisV2",
    "peekskill": "FakePeekskill",
    "sydney": "FakeSydneyV2",
    "toronto": "FakeTorontoV2",
    # --- 33Q ---
    "prague": "FakePrague",
    # --- 53Q ---
    "rochester": "FakeRochesterV2",
    # --- 65Q ---
    "brooklyn": "FakeBrooklynV2",
    "manhattan": "FakeManhattanV2",
    # --- 127Q ---
    "brisbane": "FakeBrisbane",
    "cusco": "FakeCusco",
    "kawasaki": "FakeKawasaki",
    "kyiv": "FakeKyiv",
    "kyoto": "FakeKyoto",
    "osaka": "FakeOsaka",
    "quebec": "FakeQuebec",
    "sherbrooke": "FakeSherbrooke",
    "washington": "FakeWashingtonV2",
    # --- 133Q ---
    "torino": "FakeTorino",
}


def get_backend(name: str) -> Any:
    """Instantiate a FakeBackendV2 by short name.

    Args:
        name: Backend short name (e.g. 'nairobi', 'toronto').

    Returns:
        Instantiated FakeBackendV2 object.
    """
    import qiskit_ibm_runtime.fake_provider as fp

    name_lower = name.lower().replace("fake_", "")
    if name_lower not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {sorted(BACKEND_REGISTRY.keys())}"
        )
    cls_name = BACKEND_REGISTRY[name_lower]
    cls = getattr(fp, cls_name)
    return cls()


def get_two_qubit_gate_name(backend: Any) -> str:
    """Detect the native 2-qubit gate name for a backend.

    Checks for 'cx', 'ecr', and 'cz' in order of preference.
    """
    target = backend.target
    for gate_name in ("cx", "ecr", "cz"):
        if gate_name in target.operation_names:
            return gate_name
    raise ValueError(
        f"No supported 2-qubit gate (cx, ecr, cz) found. "
        f"Available: {target.operation_names}"
    )
