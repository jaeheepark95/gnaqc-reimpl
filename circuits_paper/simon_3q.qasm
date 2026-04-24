OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[1];
h q[0];
cx q[0],q[1];
cx q[0],q[1];
h q[0];
measure q[0] -> c[0];