import time
import math
import numpy as np
import open3d as o3d
from enum import Enum
from math import pi, cos, sin


# Joint type enumerator
class JointType(Enum):
    UNSPECIFIED = 0
    REVOLUTE = 1
    PRISMATIC = 2


# Link descriptor class
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

    # Build transformation matrix (T) from home configuration
    def build_T_from_home(self, q=None):
        if q:
            self.q = q

        if self.joint_type == JointType.UNSPECIFIED:
            self.T = np.eye(4)
            
        elif self.joint_type == JointType.REVOLUTE:
            self.theta = self.theta_init + self.q
        
        elif self.joint_type == JointType.PRISMATIC:
            self.d = self.d_init + self.q

        if self.theta > pi:
            self.theta -= 2*pi
        elif self.theta <= -pi:
            self.theta += 2*pi

        self.T = np.array([[cos(self.theta), -cos(self.alpha)*sin(self.theta), sin(self.alpha)*sin(self.theta),  self.a*cos(self.theta)],
                           [sin(self.theta), cos(self.alpha)*cos(self.theta),  -sin(self.alpha)*cos(self.theta), self.a*sin(self.theta)],
                           [0,               sin(self.alpha),                  cos(self.alpha),                  self.d                ],
                           [0,               0,                                0,                                1                     ]], dtype='float')
        
        return self.T

    # Update transformation matrix (T) from current configuration
    def update_T(self, d_theta):
        self.theta += d_theta

        if self.theta > pi:
            self.theta -= 2*pi
        elif self.theta <= -pi:
            self.theta += 2*pi

        self.q = self.theta - self.theta_init

        self.T = np.array([[cos(self.theta), -cos(self.alpha)*sin(self.theta), sin(self.alpha)*sin(self.theta),  self.a*cos(self.theta)],
                           [sin(self.theta), cos(self.alpha)*cos(self.theta),  -sin(self.alpha)*cos(self.theta), self.a*sin(self.theta)],
                           [0,               sin(self.alpha),                  cos(self.alpha),                  self.d                ],
                           [0,               0,                                0,                                1                     ]], dtype='float')
        
        return self.T
    
    # Get q value (difference between current theta and home position)
    def get_q(self):
        self.q = self.theta - self.theta_init
        return self.q
    
    # Reset theta to home position
    def reset_theta(self):
        self.theta = self.theta_init
        self.q = 0


# Find the inverse transformation matrix
def find_T_inv(T:np.ndarray):
    T_inv = np.eye(4)
    T_inv[:3,:3] = T[:3,:3].T
    T_inv[:3,-1] = (-T[:3,:3].T) @ T[:3,-1]

    return T_inv


# --------------------------------------------------------------------------------------------
# Forward Kinematics

# Forward kinematics - return link frame positions
def forward_kinematics(q1, q2, q3, q4, q5, q6):
    # Link lengths
    L1 = 2
    L2 = 4
    L3 = 2
    L4 = 3
    L5 = 1.5
    L5_prime = 1
    L6 = 5

    # Build links
    link1 = Link(0, L1, 0, pi/2, JointType.REVOLUTE)
    link2 = Link(pi/2, 0, L2, 0, JointType.REVOLUTE)
    link3 = Link(-pi/2, 0, 0, -pi/2, JointType.REVOLUTE)
    link4 = Link(0, L3+L4, 0, pi/2, JointType.REVOLUTE)
    link5 = Link(-pi/2, 0, L5, -pi/2, JointType.REVOLUTE)
    link6 = Link(pi, L5_prime, L6, 0, JointType.REVOLUTE)

    # Build link transformation matrices
    T1 = link1.build_T_from_home(q1)
    T2 = link2.build_T_from_home(q2)
    T3 = link3.build_T_from_home(q3)
    T4 = link4.build_T_from_home(q4)
    T5 = link5.build_T_from_home(q5)
    T6 = link6.build_T_from_home(q6)

    # Define origin as (0, 0, 0)
    p0 = np.array([0, 0, 0, 1])

    # Build T matrices from frame 0
    T0_1 = T1
    T0_2 = T0_1 @ T2
    T0_3 = T0_2 @ T3
    T0_4 = T0_3 @ T4
    T0_5 = T0_4 @ T5
    T0_6 = T0_5 @ T6

    # Calculate points
    p1 = T0_1 @ p0
    p2 = T0_2 @ p0
    p3 = T0_3 @ p0
    p4 = T0_4 @ p0
    p5 = T0_5 @ p0
    p6 = T0_6 @ p0

    points = [p0, p1, p2, p3, p4, p5, p6]
    
    # Remove last index of points (from 1x4 vector to 1x3 vector)
    for i, p in enumerate(points):
        points[i] = p[:3]

    return points


# Forward kinematics - return link frame positions and transformation matrices
def forward_kinematics_with_orientation(q1, q2, q3, q4, q5, q6):
    # Link lengths
    L1 = 2
    L2 = 4
    L3 = 2
    L4 = 3
    L5 = 1.5
    L5_prime = 1
    L6 = 5

    # Build links
    link1 = Link(0, L1, 0, pi/2, JointType.REVOLUTE)
    link2 = Link(pi/2, 0, L2, 0, JointType.REVOLUTE)
    link3 = Link(-pi/2, 0, 0, -pi/2, JointType.REVOLUTE)
    link4 = Link(0, L3+L4, 0, pi/2, JointType.REVOLUTE)
    link5 = Link(-pi/2, 0, L5, -pi/2, JointType.REVOLUTE)
    link6 = Link(pi, L5_prime, L6, 0, JointType.REVOLUTE)

    # Build link transformation matrices
    T1 = link1.build_T_from_home(q1)
    T2 = link2.build_T_from_home(q2)
    T3 = link3.build_T_from_home(q3)
    T4 = link4.build_T_from_home(q4)
    T5 = link5.build_T_from_home(q5)
    T6 = link6.build_T_from_home(q6)

    # Define origin as (0, 0, 0)
    p0 = np.array([0, 0, 0, 1])

    # Build T matrices from frame 0
    T0_1 = T1
    T0_2 = T0_1 @ T2
    T0_3 = T0_2 @ T3
    T0_4 = T0_3 @ T4
    T0_5 = T0_4 @ T5
    T0_6 = T0_5 @ T6

    # Calculate points
    p1 = T0_1 @ p0
    p2 = T0_2 @ p0
    p3 = T0_3 @ p0
    p4 = T0_4 @ p0
    p5 = T0_5 @ p0
    p6 = T0_6 @ p0

    points = [p0, p1, p2, p3, p4, p5, p6]
    
    # Remove last index of points (from 1x4 vector to 1x3 vector)
    for i, p in enumerate(points):
        points[i] = p[:3]

    return points, T0_1, T0_2, T0_3, T0_4, T0_5, T0_6





# --------------------------------------------------------------------------------------------
# Inverse Kinematics

# Inverse kinematics - return angles for given transformtion matrix T, or None if no solution found
def inverse_kinematics(T, err_tol=0.01, max_iter=100000, valid_err_thresh=1.0):

    T_target = T

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

    # Initialize origin frame at (0,0,0)
    p0 = np.array([0, 0, 0, 1])

    # Parameter to scale theta by
    alpha = 0.24
    beta_threshold = 0.0872

    # User adjustable parameters
    threshold = err_tol                     # Error tolerance for solution
    max_iter = max_iter                     # Max number of iterations to find solution
    valid_err_thresh = valid_err_thresh     # Threshold to consider T unsolvable if max iterations reached

    # Initialize variables
    err = 1000
    best_err = err
    T_best = None

    print("Finding angles...")

    itr = 0
    while err > threshold:
        # print("Iteration: {}".format(itr))
        if itr >= max_iter:
            # print("Reached max iterations {}, breaking".format(max_iter))
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

        # Get P components (location of frame origins) from transformation matrices
        P0 = np.array([0, 0, 0], dtype='float')
        P1 = T0_1[0:3,3]
        P2 = T0_2[0:3,3]
        P3 = T0_3[0:3,3]
        P4 = T0_4[0:3,3]
        P5 = T0_5[0:3,3]
        P6 = T0_6[0:3,3]

        # Get Z components from transformation matrices (3rd column)
        Z0 = np.array([0, 0, 1], dtype='float')
        Z1 = T0_1[0:3,2]
        Z2 = T0_2[0:3,2]
        Z3 = T0_3[0:3,2]
        Z4 = T0_4[0:3,2]
        Z5 = T0_5[0:3,2]
        Z6 = T0_6[0:3,2]

        # Create Jacobian from Z and P components
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

        # Calculate difference between target position and current position
        T6_0 = find_T_inv(T0_6)
        Te_t = T6_0 @ T_target
        d_x = T_target[0:3,3] - T0_6[0:3,3]

        # Calculate difference in orientation with quaternion
        try:
            qw = math.sqrt(1 + Te_t[0,0] + Te_t[1,1] + Te_t[2,2]) / 2
            qx = (Te_t[2,1] - Te_t[1,2]) / (4*qw)
            qy = (Te_t[0,2] - Te_t[2,0]) / (4*qw)
            qz = (Te_t[1,0] - Te_t[0,1]) / (4*qw)

            tx = 2 * math.asin(qx)
            ty = 2 * math.asin(qy)
            tz = 2 * math.asin(qz)

            d_w = np.array([tx, ty, tz])
        except:
            print("Exception while calculating difference in angle")
            print("Desired end effector position and orientation our of workspace of robot")
            print("Completed in {} iterations".format(itr))
            return None
        
        # Form dx (delta of location and orientation)
        d_x = np.concatenate((d_x, d_w), dtype='float')

        # Calculate change in theta
        d_theta = J_pinv @ (alpha * d_x)
        # d_theta = J.T @ (alpha * d_x)

        # Update T matrices with new thetas
        T1 = link1.update_T(d_theta[0])
        T2 = link2.update_T(d_theta[1])
        T3 = link3.update_T(d_theta[2])
        T4 = link4.update_T(d_theta[3])
        T5 = link5.update_T(d_theta[4])
        T6 = link6.update_T(d_theta[5])

        # Build T matrices from frame 0
        T0_1 = T1
        T0_2 = T0_1 @ T2
        T0_3 = T0_2 @ T3
        T0_4 = T0_3 @ T4
        T0_5 = T0_4 @ T5
        T0_6 = T0_5 @ T6

        # Calculate link frame points
        p1 = T0_1 @ p0
        p2 = T0_2 @ p0
        p3 = T0_3 @ p0
        p4 = T0_4 @ p0
        p5 = T0_5 @ p0
        p6 = T0_6 @ p0

        # If a point intersets with the wall, reset link 2 to home position
        if (p2[2] < 0) or (p3[2] < 0) or (p4[2] < 0) or (p5[2] < 0) or (p6[2] < 0):
            link2.reset_theta()

        # Calculate error (norm of distance between new end effector location and target)
        err = np.linalg.norm(T0_6 - T_target)

        # Get angles
        angles = [link1.get_q(), link2.get_q(), link3.get_q(), link4.get_q(), link5.get_q(), link6.get_q()]

        if err < best_err:
            best_err = err
            T_best = T0_6
            best_angles = angles

        itr += 1
    
    # If error too high, couldn't find correct angles
    if best_err > valid_err_thresh:
        print("Desired end effector position and orientation our of workspace of robot")
        print("Completed in {} iterations".format(itr))
        return None

    # 
    for i, a in enumerate(angles):
        if a > pi:
            mult = abs(a / (2*pi))
            angles[i] = a - round(mult)*2*pi
        elif a <= -pi:
            mult = abs(a / (2*pi))
            angles[i] = a + round(mult)*2*pi

    print("Error: {}".format(best_err))
    print("Angles: {}".format(best_angles))
    print("Final T:\n{}".format(T_best))
    print("Completed in {} iterations".format(itr))

    return best_angles
    # return angles
    # return T_best



# --------------------------------------------------------------------------------------------
# Visualization

def visualize_arm(T_matrices, T_verify=None):
    frames = []
    points = []
    lines = []
    p0 = np.array([0, 0, 0, 1])
    points.append(p0[:3])

    frames.append(o3d.geometry.TriangleMesh.create_coordinate_frame())

    for i, T in enumerate(T_matrices):
        f = o3d.geometry.TriangleMesh.create_coordinate_frame().transform(T)
        frames.append(f)
        p = T @ p0
        points.append(p[:3])

        if i > 0:
            lines.append([i-1, i])


    lines.append([len(T_matrices)-1, len(T_matrices)])

    axes_pcd = o3d.geometry.PointCloud()
    axes_pcd.points = o3d.utility.Vector3dVector(np.asarray([[0,0,0],[10,0,0],[0,10,0]]))
    axes_l = [[0,1],
              [0,2]]
    axes_lines = o3d.geometry.LineSet(points=axes_pcd.points,
                                      lines=o3d.utility.Vector2iVector(axes_l))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    line_set = o3d.geometry.LineSet(points=pcd.points,
                                    lines=o3d.utility.Vector2iVector(lines))

    visuals = []
    visuals.extend(frames)
    visuals.append(pcd)
    visuals.append(line_set)
    visuals.append(axes_lines)

    if T_verify:
        frames_v = []
        points_v = []
        lines_v = []

        points_v.append(p0[:3])
        for i, T in enumerate(T_verify):
            f = o3d.geometry.TriangleMesh.create_coordinate_frame().transform(T)
            frames_v.append(f)
            p = T @ p0
            points_v.append(p[:3])

            if i > 0:
                lines_v.append([i-1, i])
        
        lines_v.append([len(T_verify)-1, len(T_verify)])

        pcd_v = o3d.geometry.PointCloud()
        pcd_v.points = o3d.utility.Vector3dVector(points_v)

        line_set_v = o3d.geometry.LineSet(points=pcd_v.points,
                                        lines=o3d.utility.Vector2iVector(lines_v))
        
        visuals.extend(frames_v)
        visuals.append(pcd_v)
        visuals.append(line_set_v)

    o3d.visualization.draw_geometries(visuals)

