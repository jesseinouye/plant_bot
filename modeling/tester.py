from math import pi, cos, sin
import open3d as o3d
import numpy as np
import time

from enum import Enum


# Notes:
# Inverse Kinematics
#   - Treat arm as two separate manipulators: shoulder/elbow and wrist
#       - Shoulder/elbow gets arm to correct position
#       - Wrist rotates end effector to correct orientation
# 
#   - In terms of our arm:
#       - Get frame 6 to correct orientation w.r.t. frame 4 (position and orientation?)
#           - Find all posible solutions?
#           - Filter out bad solutions using some reachable workspace condition (all posible points that frame 4 can reach)
#       - Get frame 4 into correct position using 3R IK w.r.t. frame 0 (position of frame 4 only, not orientation - find necessary orientation as angle q4)
#           - Do this for all posible solutions of the above


# t_4 = x_D - T_4_D * x_4
#       t_4 = frame 4 position
#       x_D = target point
#       T_4_D = transformation from frame 4 to target point
#       x_4 = frame 4 point


# Thoughts:
#   - Position / orientation of end effector (F6) defines position and z-axis orientation of F5
#       - Which defines position for F4, and z-axis orientation of F4
# 
#   - Find position 



p1 = [0.0, 0.0, 0.0]
p2 = [0.0, 0.0, 1.0]
p3 = [0.0, 1.0, 1.0]


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
        self.T = None

    def build_T(self, q=None):
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
                           [0,               0,                                0,                                1                     ]])
        
        return self.T


# L1 = 1
# L2 = 2
# L3 = 3
# L4 = 4
# L5 = 5
# L5_prime = 0.1
# L6 = 6

L1 = 2
L2 = 4
L3 = 2
L4 = 3
L5 = 1.5
L5_prime = 1
L6 = 5


link1 = Link(0, L1, 0, pi/2, JointType.REVOLUTE)
link2 = Link(pi/2, 0, L2, 0, JointType.REVOLUTE)
link3 = Link(-pi/2, 0, 0, -pi/2, JointType.REVOLUTE)
link4 = Link(0, L3+L4, 0, pi/2, JointType.REVOLUTE)
link5 = Link(-pi/2, 0, L5, -pi/2, JointType.REVOLUTE)
link6 = Link(pi, L5_prime, L6, 0, JointType.REVOLUTE)


T1 = link1.build_T(pi/2)
T2 = link2.build_T(0)
T3 = link3.build_T(pi/2)
T4 = link4.build_T(0)
T5 = link5.build_T(0)
T6 = link6.build_T(0)

p0 = np.array([0, 0, 0, 1])
# p1 = T1 @ p0
# p2 = T2 @ p1
# p3 = T3 @ p2
# p4 = T4 @ p3
# p5 = T5 @ p4
# p6 = T6 @ p5

p1 = T1 @ p0
p2 = T1 @ T2 @ p0
p3 = T1 @ T2 @ T3 @ p0
p4 = T1 @ T2 @ T3 @ T4 @ p0
p5 = T1 @ T2 @ T3 @ T4 @ T5 @ p0
p6 = T1 @ T2 @ T3 @ T4 @ T5 @ T6 @ p0

T0_1 = T1
T0_2 = T1 @ T2
T0_3 = T1 @ T2 @ T3
T0_4 = T1 @ T2 @ T3 @ T4
T0_5 = T1 @ T2 @ T3 @ T4 @ T5
T0_6 = T1 @ T2 @ T3 @ T4 @ T5 @ T6


print("p1: {}, p2: {}, p3: {}, p4: {}, p5: {}, p6: {}".format(p1, p2, p3, p4, p5, p6))
print("T1:\n{}\nT2:\n{}\nT3:\n{}\nT4:\n{}\nT5:\n{}\nT6:\n{}".format(T1, T2, T3, T4, T5, T6))


print("T0_1:\n{}".format(T0_1))
print("T0_2:\n{}".format(T0_2))
print("T0_3:\n{}".format(T0_3))
print("T0_4:\n{}".format(T0_4))
print("T0_5:\n{}".format(T0_5))
print("T0_6:\n{}".format(T0_6))

points = [p0, p1, p2, p3, p4, p5, p6]

for i, p in enumerate(points):
    points[i] = p[:3] + np.array([0.0, 2.0, 0.0])
    # points[i] = p[:3]







# points = [p1, p2, p3]
points = np.asarray(points)

l = [[0, 1],
     [1, 2],
     [2, 3],
     [3, 4],
     [4, 5],
     [5, 6]]



frame0 = o3d.geometry.TriangleMesh.create_coordinate_frame()
frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame().transform(T0_1)
frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame().transform(T0_2)
frame3 = o3d.geometry.TriangleMesh.create_coordinate_frame().transform(T0_3)
frame4 = o3d.geometry.TriangleMesh.create_coordinate_frame().transform(T0_4)
frame5 = o3d.geometry.TriangleMesh.create_coordinate_frame().transform(T0_5)
frame6 = o3d.geometry.TriangleMesh.create_coordinate_frame().transform(T0_6)





if 1:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    line_set = o3d.geometry.LineSet(points=pcd.points,
                                    lines=o3d.utility.Vector2iVector(l))

    np_colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    pcd.colors = o3d.utility.Vector3dVector(np_colors)

    print("colors? {}".format(pcd.has_colors()))

    # o3d.visualization.draw_geometries([line_set, pcd])
    # o3d.visualization.draw_geometries([frame0, frame1, frame2, frame3, frame4, frame5, frame6, pcd])
    o3d.visualization.draw_geometries([frame0, frame1, frame2, frame3, frame4, frame5, frame6, pcd, line_set])


if 0:
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(frame0)
    vis.add_geometry(frame1)

    segments = 10

    angles = [(pi/2)/segments * (i+1) for i in range(segments)]

    for q in angles:
        T1 = link1.build_T((pi/2)/segments)
        T0_1 = T1
        print("Using q: {}\nT0_1:\n{}".format(q, T0_1))
        frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame().transform(T0_1)
        # frame1 = frame1.transform(T0_1)
        vis.update_geometry(frame0)
        vis.update_geometry(frame1)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.5)















# TODO: figure out how to do this as a "nice to have"
def matmul_str(M1, M2):
    M1_rows = len(M1)
    M1_cols = len(M1[0])
    M2_rows = len(M2)
    M2_cols = len(M2[0])

    if (M1_rows != M2_cols) or (M1_cols != M2_rows):
        print("ERROR: matrix of size [{},{}] doesn't match [{},{}] for matrix multiplication".format(M1_rows, M1_cols, M2_rows, M2_cols))

    O = []

    for k, M2_col in M2[0]:
        for i, M1_row in M1:
            out_row = []
            for j, M1_col in M1_row:
                out_row.append(M1[i][i])
                T1 = link1.build_T(0)
            