import numpy as np
import open3d as o3d
from enum import Enum
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


# --------------------------------------------------------------------------------------------------
# IK with Jacobian
#   - Get transformation matrices of current position
#   - Form Jacobian from T0_X matrices
#   - Get Jacobian pseudo-inverse
#   - Get difference between target position and current position (d_p)
#   - Get difference between target orientation and current orientation (or set to to a small value?)
#   - Get delta theta (d_theta = alpha * J_pinv * d_x)
#   - Update theta of arm
#   - Repeat




class JointType(Enum):
    UNSPECIFIED = 0
    REVOLUTE = 1
    PRISMATIC = 2


class Link():
    def __init__(self, theta=0, d=0, a=0, alpha=0, joint_type=JointType.UNSPECIFIED):
        self.theta_init = theta
        self.d_init = d

        self.theta = self.theta_init
        self.d = self.d_init
        self.a = a
        self.alpha = alpha

        self.q = 0
        self.joint_type = joint_type
        self.T = self.build_T_from_home(q=0)

    def build_T_from_home(self, q=None):
        if q:
            self.q = q

        if self.joint_type == JointType.UNSPECIFIED:
            self.T = np.eye(4)
            
        elif self.joint_type == JointType.REVOLUTE:
            self.theta = self.theta_init + self.q
        
        elif self.joint_type == JointType.PRISMATIC:
            self.d = self.d_init + self.q

        self.T = np.array([[cos(self.theta), -cos(self.alpha)*sin(self.theta), sin(self.alpha)*sin(self.theta),  self.a*cos(self.theta)],
                           [sin(self.theta), cos(self.alpha)*cos(self.theta),  -sin(self.alpha)*cos(self.theta), self.a*sin(self.theta)],
                           [0,               sin(self.alpha),                  cos(self.alpha),                  self.d                ],
                           [0,               0,                                0,                                1                     ]], dtype='float')
        
        return self.T

    def update_T(self, d_theta):
        self.theta += d_theta

        self.T = np.array([[cos(self.theta), -cos(self.alpha)*sin(self.theta), sin(self.alpha)*sin(self.theta),  self.a*cos(self.theta)],
                           [sin(self.theta), cos(self.alpha)*cos(self.theta),  -sin(self.alpha)*cos(self.theta), self.a*sin(self.theta)],
                           [0,               sin(self.alpha),                  cos(self.alpha),                  self.d                ],
                           [0,               0,                                0,                                1                     ]], dtype='float')
        
        return self.T
    

# Find the inverse transformation matrix
def find_T_inv(T:np.ndarray):
    T_inv = np.eye(4)
    T_inv[:3,:3] = T[:3,:3].T
    T_inv[:3,-1] = (-T[:3,:3].T) @ T[:3,-1]

    return T_inv



# T_target = np.array([[0,    1,      0,      0],
#                      [-1,   0,      0,      -8.5],
#                      [0,    0,      1,      7.0],
#                      [0,    0,      0,      1]], dtype='float')


# L1 = 2
# L2 = 4
# L3 = 2
# L4 = 3
# L5 = 1.5
# L5_prime = 1
# L6 = 5

# alpha = 0.01

# # Build links with home config
# link1 = Link(0, L1, 0, pi/2, JointType.REVOLUTE)
# link2 = Link(pi/2, 0, L2, 0, JointType.REVOLUTE)
# link3 = Link(-pi/2, 0, 0, -pi/2, JointType.REVOLUTE)
# link4 = Link(0, L3+L4, 0, pi/2, JointType.REVOLUTE)
# link5 = Link(-pi/2, 0, L5, -pi/2, JointType.REVOLUTE)
# link6 = Link(pi, L5_prime, L6, 0, JointType.REVOLUTE)

# T1 = link1.build_T_from_home(0)
# T2 = link2.build_T_from_home(0)
# T3 = link3.build_T_from_home(0)
# T4 = link4.build_T_from_home(0)
# T5 = link5.build_T_from_home(0)
# T6 = link6.build_T_from_home(0)

# # p0 = np.array([0, 0, 0, 1], dtype='float')

# T0_1 = T1
# T0_2 = T1 @ T2
# T0_3 = T1 @ T2 @ T3
# T0_4 = T1 @ T2 @ T3 @ T4
# T0_5 = T1 @ T2 @ T3 @ T4 @ T5
# T0_6 = T1 @ T2 @ T3 @ T4 @ T5 @ T6

# P0 = np.array([0, 0, 0], dtype='float')
# P1 = T0_1[0:3,3]
# P2 = T0_2[0:3,3]
# P3 = T0_3[0:3,3]
# P4 = T0_4[0:3,3]
# P5 = T0_5[0:3,3]
# P6 = T0_6[0:3,3]

# Z0 = np.array([0, 0, 1], dtype='float')
# Z1 = T0_1[0:3,2]
# Z2 = T0_2[0:3,2]
# Z3 = T0_3[0:3,2]
# Z4 = T0_4[0:3,2]
# Z5 = T0_5[0:3,2]
# Z6 = T0_6[0:3,2]


# J00 = np.cross(Z0, (P6 - P0))
# J10 = Z0
# J0 = np.concatenate((J00, J10), dtype='float')

# J01 = np.cross(Z1, (P6 - P1))
# J11 = Z1
# J1 = np.concatenate((J01, J11), dtype='float')

# J02 = np.cross(Z2, (P6 - P2))
# J12 = Z2
# J2 = np.concatenate((J02, J12), dtype='float')

# J03 = np.cross(Z3, (P6 - P3))
# J13 = Z3
# J3 = np.concatenate((J03, J13), dtype='float')

# J04 = np.cross(Z4, (P6 - P4))
# J14 = Z4
# J4 = np.concatenate((J04, J14), dtype='float')

# J05 = np.cross(Z5, (P6 - P5))
# J15 = Z5
# J5 = np.concatenate((J05, J15), dtype='float')

# J = np.column_stack((J0, J1, J2, J3, J4, J5))

# print("J00: {}".format(J00))
# print("J10: {}".format(J10))
# print("J1: {}".format(J0))
# print("J:\n{}".format(J))

# J_pinv = np.linalg.pinv(J)

# T6_0 = find_T_inv(T0_6)

# Te_t = T6_0 @ T_target

# print("{}".format(Te_t[0:3,3]))

# d_x = np.concatenate([Te_t[0:3,3], np.array([0.01, 0.01, 0.01])])

# print("d_x:\n{}".format(d_x))

# d_theta = alpha * J_pinv @ d_x

# print("d_theta:\n{}".format(d_theta))

# T1 = link1.update_T(d_theta[0])
# T2 = link2.update_T(d_theta[1])
# T3 = link3.update_T(d_theta[2])
# T4 = link4.update_T(d_theta[3])
# T5 = link5.update_T(d_theta[4])
# T6 = link6.update_T(d_theta[5])

# T0_6 = T1 @ T2 @ T3 @ T4 @ T5 @ T6

# err = np.linalg.norm(T0_6 - T_target)
# print("err: {}".format(err))




# Target location / orientation
T_target = np.array([[0,    1,      0,      0],
                     [-1,   0,      0,      -8.5],
                     [0,    0,      1,      7.0],
                     [0,    0,      0,      1]], dtype='float')

# Link lengths
L1 = 2
L2 = 4
L3 = 2
L4 = 3
L5 = 1.5
L5_prime = 1
L6 = 5

# Create links
link1 = Link(0, L1, 0, pi/2, JointType.REVOLUTE)
link2 = Link(pi/2, 0, L2, 0, JointType.REVOLUTE)
link3 = Link(-pi/2, 0, 0, -pi/2, JointType.REVOLUTE)
link4 = Link(0, L3+L4, 0, pi/2, JointType.REVOLUTE)
link5 = Link(-pi/2, 0, L5, -pi/2, JointType.REVOLUTE)
link6 = Link(pi, L5_prime, L6, 0, JointType.REVOLUTE)


# Parameter to update theta by
alpha = 0.2
# alpha = 1

threshold = 0.2
max_iter = 1000
itr = 0

err = 1000
best_err = err

T_best = None

while err > threshold:
    print("Iteration: {}".format(itr))
    if itr >= max_iter:
        print("Reached max iterations {}, breaking".format(max_iter))
        break

    # Get transformation matrices of current position
    T1 = link1.T
    T2 = link2.T
    T3 = link3.T
    T4 = link4.T
    T5 = link5.T
    T6 = link6.T

    # Form Jacobian
    T0_1 = T1
    T0_2 = T1 @ T2
    T0_3 = T1 @ T2 @ T3
    T0_4 = T1 @ T2 @ T3 @ T4
    T0_5 = T1 @ T2 @ T3 @ T4 @ T5
    T0_6 = T1 @ T2 @ T3 @ T4 @ T5 @ T6

    P0 = np.array([0, 0, 0], dtype='float')
    P1 = T0_1[0:3,3]
    P2 = T0_2[0:3,3]
    P3 = T0_3[0:3,3]
    P4 = T0_4[0:3,3]
    P5 = T0_5[0:3,3]
    P6 = T0_6[0:3,3]

    Z0 = np.array([0, 0, 1], dtype='float')
    Z1 = T0_1[0:3,2]
    Z2 = T0_2[0:3,2]
    Z3 = T0_3[0:3,2]
    Z4 = T0_4[0:3,2]
    Z5 = T0_5[0:3,2]
    Z6 = T0_6[0:3,2]


    J00 = np.cross(Z0, (P6 - P0))
    J10 = Z0
    J0 = np.concatenate((J00, J10), dtype='float')

    J01 = np.cross(Z1, (P6 - P1))
    J11 = Z1
    J1 = np.concatenate((J01, J11), dtype='float')

    J02 = np.cross(Z2, (P6 - P2))
    J12 = Z2
    J2 = np.concatenate((J02, J12), dtype='float')

    J03 = np.cross(Z3, (P6 - P3))
    J13 = Z3
    J3 = np.concatenate((J03, J13), dtype='float')

    J04 = np.cross(Z4, (P6 - P4))
    J14 = Z4
    J4 = np.concatenate((J04, J14), dtype='float')

    J05 = np.cross(Z5, (P6 - P5))
    J15 = Z5
    J5 = np.concatenate((J05, J15), dtype='float')

    J = np.column_stack((J0, J1, J2, J3, J4, J5))

    # Get Jacobian pseudo-inverse
    J_pinv = np.linalg.pinv(J)

    # Get difference between target position and current position
    T6_0 = find_T_inv(T0_6)
    Te_t = T6_0 @ T_target
    # d_x = np.concatenate([Te_t[0:3,3], np.array([0.001, 0.001, 0.001])])

    # print("T_target:\n{}".format(T_target))
    # print("T0_6:\n{}".format(T0_6))
    # print("Te_t:\n{}".format(Te_t))
    d_x = T_target[0:3,3] - T0_6[0:3,3]
    # d_w = np.array([T_target[0,0] - T0_6[0,0], T_target[1,1] - T0_6[1,1], T_target[2,2] - T0_6[2,2]])
    d_w = np.array([1-Te_t[0,0], 1-Te_t[1,1], 1-Te_t[2,2]]) * 0.1
    # d_x = np.concatenate((d_x, np.array([0.001, 0.001, 0.001])), dtype='float')
    d_x = np.concatenate((d_x, d_w), dtype='float')
    # print("d_x:\n{}".format(d_x))


    # Calculate change in theta
    d_theta = alpha * J_pinv @ d_x

    # Update T matrices with new thetas
    T1 = link1.update_T(d_theta[0])
    T2 = link2.update_T(d_theta[1])
    T3 = link3.update_T(d_theta[2])
    T4 = link4.update_T(d_theta[3])
    T5 = link5.update_T(d_theta[4])
    T6 = link6.update_T(d_theta[5])

    # Get new end effector position / orientation
    # T_orig = T0_6
    T0_6 = T1 @ T2 @ T3 @ T4 @ T5 @ T6

    # Get error (norm of distance between new end effector location and target)
    err = np.linalg.norm(T0_6 - T_target)
    print("err: {}".format(err))

    if err < best_err:
        best_err = err
        T_best = T0_6
        best_angles = [link1.theta, link2.theta, link3.theta, link4.theta, link5.theta, link6.theta]

    itr += 1

print("Best error: {}".format(best_err))
print("Best angles: {}".format(best_angles))
print("Final T:\n{}".format(T_best))


frame_end = o3d.geometry.TriangleMesh.create_coordinate_frame().transform(T_best)
frame_target = o3d.geometry.TriangleMesh.create_coordinate_frame().transform(T_target)

o3d.visualization.draw_geometries([frame_end, frame_target])