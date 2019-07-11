OPENQASM 2.0;
include "qelib1.inc";

qreg q[5];
creg c[5];

id q[0];
id q[0];
id q[0];
id q[0];

rx(pi) q[0];
rx(pi) q[0];
rx(pi) q[0];
rx(pi) q[0];

rz(pi) q[0];
rz(pi) q[0];
rz(pi) q[0];
rz(pi) q[0];

rx(pi) q[0];
rz(pi) q[0];
rx(pi) q[0];
rz(pi) q[0];

rz(pi) q[0];
rx(pi) q[0];
rz(pi) q[0];
rx(pi) q[0];

rx(pi/2) q[0];
id q[0];
rx(pi/2) q[0];
id q[0];

rz(pi/2) q[0];
id q[0];
rz(pi/2) q[0];
id q[0];

rx(pi/2) q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];

rz(pi/2) q[0];
rx(pi/2) q[0];
rz(pi/2) q[0];
rx(pi/2) q[0];

rx(pi/2) q[0];
rz(pi) q[0];
rx(pi/2) q[0];
rz(pi) q[0];

rz(pi/2) q[0];
rx(pi) q[0];
rz(pi/2) q[0];
rx(pi) q[0];

rx(pi) q[0];
rz(pi/2) q[0];
rx(pi) q[0];
rz(pi/2) q[0];

rz(pi) q[0];
rx(pi/2) q[0];
rz(pi) q[0];
rx(pi/2) q[0];

rx(pi/2) q[0];
rx(pi) q[0];
rx(pi/2) q[0];
rx(pi) q[0];

rx(pi) q[0];
rx(pi/2) q[0];
rx(pi) q[0];
rx(pi/2) q[0];

rz(pi/2) q[0];
rz(pi) q[0];
rz(pi/2) q[0];
rz(pi) q[0];

rz(pi) q[0];
rz(pi/2) q[0];
rz(pi) q[0];
rz(pi/2) q[0];

rx(pi) q[0];
id q[0];
rx(pi) q[0];
id q[0];

rz(pi) q[0];
id q[0];
rz(pi) q[0];
id q[0];

rx(pi/2) q[0];
rx(pi/2) q[0];
rx(pi/2) q[0];
rx(pi/2) q[0];

rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
