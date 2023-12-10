import numpy as np
from sympy import pi, cos, sin, solve, solve_linear, nsolve
from sympy import Symbol, Matrix

# x = Symbol('x')
# y = Symbol('y')

# eq = y*cos(x)**2 + y*sin(x)**2 - y

# v = [x,y]

# T0_1 = np.array()

if 0:
    q1 = Symbol('q1')
    q2 = Symbol('q2')
    q3 = Symbol('q3')
    q4 = Symbol('q4')
    q5 = Symbol('q5')
    q6 = Symbol('q6')

    L1 = 2
    L2 = 4
    L3 = 2
    L4 = 3
    L5 = 1.5
    L5p = 1
    L6 = 5


    R11 = 0
    R21 = 0
    R31 = 1

    R12 = 0
    R22 = -1
    R32 = 0

    R13 = 1
    R23 = 0
    R33 = 0

    T1 = 0
    T2 = -5
    T3 = 6


    # eq = sin(q4)*sin(q6)*(cos(q2 + pi/2)*sin(q3 - pi/2) + cos(q3 - pi/2)*sin(q2 + pi/2)) - cos(q6)*(sin(q5 - pi/2)*(cos(q2 + pi/2)*cos(q3 - pi/2) - sin(q2 + pi/2)*sin(q3 - pi/2)) + cos(q4)*cos(q5 - pi/2)*(cos(q2 + pi/2)*sin(q3 - pi/2) + cos(q3 - pi/2)*sin(q2 + pi/2))) - R31

    t1 = q1
    t2 = (pi/2)+q2
    t3 = (-pi/2)+q3
    t4 = q4
    t5 = (-pi/2)+q5
    t6 = pi+q6

    T0_1 = Matrix([[cos(t1),    0,          sin(t1),    0], 
                [sin(t1),    0,          -cos(t1),   0],
                [0,          1,          0,          L1],
                [0,          0,          0,          1]])

    T1_2 = Matrix([[cos(t2),    -sin(t2),   0,          L2*cos(t2)],
                [sin(t2),    cos(t2),    0,          L2*sin(t2)],
                [0,          0,          1,          0        ],
                [0,          0,          0,          1        ]])

    T2_3 = Matrix([[cos(t3),    0,          -sin(t3),   0],
                [sin(t3),    0,          cos(t3),    0],
                [0,          -1,         0,          0],
                [0,          0,          0,          1]])

    T3_4 = Matrix([[cos(t4),    0,          sin(t4),    0],
                [sin(t4),    0,          -cos(t4),   0],
                [0,          1,          0,          L3+L4],
                [0,          0,          0,          1]])

    T4_5 = Matrix([[cos(t5),    0,          -sin(t5),   L5*cos(t5)],
                [sin(t5),    0,          cos(t5),    L5*sin(t5)],
                [0,          -1,         0,          0],
                [0,          0,          0,          1]])

    T5_6 = Matrix([[cos(t6),    -sin(t6),   0,          L6*cos(t6)],
                [sin(t6),    cos(t6),    0,          L6*sin(t6)],
                [0,          0,          1,          L5p],
                [0,          0,          0,          1]])


    T0_X = T0_1 * T1_2 * T2_3 * T3_4

    T11 = T0_X[0,0] - R11
    T21 = T0_X[1,0] - R21
    T31 = T0_X[2,0] - R31

    T12 = T0_X[0,1] - R12
    T22 = T0_X[1,1] - R22
    T32 = T0_X[2,1] - R32

    T13 = T0_X[0,2] - R13
    T23 = T0_X[1,2] - R23
    T33 = T0_X[2,2] - R33

    T14 = T0_X[0,3] - T1
    T24 = T0_X[1,3] - T2
    T34 = T0_X[2,3] - T3


    eqs = [T11, T21, T31, T12, T22, T32, T13, T23, T33, T14, T24, T34]
    v = [q1, q2, q3, q4]


    # eq = cos(q1)*cos(q2 + pi/2)

    # eqs = [eq]
    # v = [q1, q2, q3, q4, q5, q6]
    # v = [q1, q2]



    S = solve(eqs, v, dict=True, minimal=True)
    print("{}".format(S))


if 1:
    # IK with Jacobian
    #   - 


    def compute_jacobian():
        pass