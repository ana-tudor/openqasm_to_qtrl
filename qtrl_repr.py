import numpy as np
from math import cos, sin, pi
import traceback

sign = lambda x: (1, -1)[x < 0]
j = lambda a, b: np.complex(a, b)
e = lambda t: j(cos(t), sin(t))
const = 1/np.sqrt(2)
def p(seg):
    num = 1
    denom = 1
    if isinstance(seg, float) or isinstance(seg, int):
        return seg
    if len(seg.split('*')) == 2:
        print()
        num = float(seg.split('*')[0])
    if len(seg.split('/')) ==2:
        alt_num = seg.split('/')[0]
        denom = float (seg.split('/')[1])
        if alt_num == 'pi':
            return num*pi/denom*360/(2*pi)
        else:
            alt_num = float(alt_num)
            return num*alt_num/denom*360/(2*pi)
    return num*pi*360/(2*pi)


param_gates = ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'crz', 'cu1', 'cu3']

def u3(params):
    #t[0], -90, 90]
    t1, t2, l = p(params[0]), p(params[1]), p(params[2])
    if (t1==90) and (t2==-90) and (l==90):
        return ['X90']
    mat_rep = np.array([[cos(t2/2), -e(l)*sin(t2/2)],
                        [e(t1)*sin(t2/2), e(t1+l)*cos(t2/2)]])
    z1 = 'Z'+str(round(t1 - 180))
    z2 = 'Z'+str(round(t2 + 180))
    z3 = 'Z'+str(round(l))
    gp = str(round((l + t1)/2))
    print("U3 global phase:", gp)
    # print("stack: ", traceback.print_exc(file=sys.stdout))
    return [z1, 'X90', z2, 'X90', z3]

def u2(params):
    t1, l = p(params[0]), p(params[1])
    z1 = 'Z'+str(round(t1 + 180))
    z2 = 'Z90'
    z3 = 'Z'+str(round(l))
    gp = str(round((t1 + l)/2))
    print("U2 global phase:",gp)
    return [z1, 'X90', z2, 'X90', z3]

def u1(params):
    l = p(params[0])
    z1 = 'Z'+str(round(l))
    z2 = 'Z180'
    z3 = 'Z-180'
    gp = str(round(l/2))
    print("U1 global phase:",gp)
    return [z1, 'X90', z2, 'X90', z3]

def check_u3_mat(t1, t2, l):
    mat_rep = np.array([[cos(t2/2), -e(l)*sin(t2/2)],
                        [e(t1)*sin(t2/2), e(t1+l)*cos(t2/2)]])

    def repr(t1, t2, l):
        a = t1 - pi
        b = t2 + pi
        c = l
        d = (l + t1)/2
        return calc_repr(a, b, c, d)

    return np.isclose(mat_rep, repr(t1, t2, l))

'''
The following are in terms of the u3 representation
'''

def rx(t):
    print(p(t[0]))
    if (round(p(t[0])) == 90) or (round(p(t[0])) == 180):
        return ['X'+str(round(p(t[0])))]
    else:
        return u3([t[0], -90, 90])
ry = lambda t: u3([t[0], 0, 0])
rz = lambda t: ['Z'+str(round(p(t[0])))]


'''
Controlled gates:
'''

def cu3(params):
    t1, t2, l = p(params[0]), p(params[1]), p(params[2])
    a = -t1/2 + 90
    b = t2/2
    c = -l/2 - 90
    d = 0
    output = ['Z'+str(round((c-a)/2)), \
              'CNOT', \
              'Z'+str(round(-b/2-180)), 'X90', 'Z'+str(round(180)), 'X90', 'Z'+str(round(-(c+a)/2)),\
              'CNOT',\
              'Z'+str(round(a+b/2-180)), 'X90', 'Z'+str(round(180)), 'X90']
    return output

cu1 = lambda l: cu3([0, 0, l[0]])
crz = lambda t: cu1(t)

'''
Parameter-less gates:
'''
paramless_reprs = {
'x':['X90', 'X90'], #gp 90
'y':['Z180', 'X90', 'X90'], #gp 90
'z':['X90', 'Z180', 'X90'], #gp 90
'h':['X90', 'Z90', 'X90'], #gp 90
's':['Z-90', 'X90', 'Z180', 'X90'], #gp 45
't':['Z-135', 'X90', 'Z180', 'X90'], #gp 22.5
'sdg':['Z90', 'X90', 'Z180', 'X90'], #gp 135
'tdg':['Z-225', 'X90', 'Z180', 'X90'], #gp -22.5
'id':['I'],
'ch':['X90', 'X90'],
'cx':cu3([180, 0, 180]),
'cy':cu3([180, 90, 90]),
'cz':cu1([180])
}


h = const*np.array([[1, 1], \
                    [1, -1]])
x = np.array([[0, 1,], \
              [1, 0]])
y = np.array([[0, j(0, -1),], \
              [j(0, 1), 0]])
z = np.array([[1, 0,], \
              [0, -1]])
s = np.array([[1, 0,], \
              [0, j(0, 1)]])
sdg = np.array([[1, 0,], \
                [0, j(0, -1)]])
t = np.array([[1, 0,], \
            [0, e(pi/4)]])
tdg = np.array([[1, 0,], \
              [0, e(-pi/4)]])
id = np.array([[1, 0,], \
              [0, 1]])

def Rx(t):
    return np.array([[cos(t/2), -j(0,sin(t/2))], \
                     [-j(0,sin(t/2)), cos(t/2)]])

def Rz(t):
    return np.array([[e(-t/2), 0], \
                     [0, e(t/2)]])

def calc_repr(t1, t2, t3, t4):
    res = e(t4)*Rz(t1)@Rx(pi/2)@Rz(t2)@Rx(pi/2)@Rz(t3)
    print("Representation:\n", res)
    return res

def calc_alt_repr(t1, t2, t3, t4):
    res= e(t4)*np.array([[-j(0,e(-(t1+t3)/2)*sin(t2/2)), -j(0,e(-(t1-t3)/2)*cos(t2/2))], \
                   [-j(0,e((t1-t3)/2)*cos(t2/2)), j(0,e((t1+t3)/2)*sin(t2/2))]])
    return res

def check_gate(mat, t1, t2, t3, gphase):
    print("Original matrix:\n", mat)
    return np.isclose(mat, calc_repr(t1, t2, t3, gphase))#, \
            # np.isclose(mat, calc_alt_repr(t1, t2, t3, gphase)), \
            # np.isclose(calc_repr(t1, t2, t3, gphase), calc_alt_repr(t1, t2, t3, gphase))

def check_had():
    return check_gate(tdg, -5*pi/4, pi, 0, -pi/8)
