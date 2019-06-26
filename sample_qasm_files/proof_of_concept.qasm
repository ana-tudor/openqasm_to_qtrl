OPENQASM 2.0;
include "qelib1.inc";

qreg q[5];
creg c[5];

h q[0];
ch q[1],q[2];
u3(pi/2,pi/2,pi/2) q[0];
cy q[1],q[2];
u2(pi/2,pi/2) q[0];
cz q[1],q[2];
u1(pi/2) q[0];
crz(pi/2) q[1],q[2];
rx(pi/2) q[0];
cu1(pi/2) q[1],q[2];
ry(pi/2) q[0];
cu3(pi/2,pi/2,pi/2) q[1],q[2];
rz(pi/2) q[0];
cx q[1],q[2];
swap q[1],q[2];
x q[0];
y q[0];
measure q[1] -> c[1];
z q[0];
s q[0];
sdg q[0];
t q[0];
tdg q[0];
