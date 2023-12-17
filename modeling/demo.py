import numpy as np
import open3d as o3d
from enum import Enum
from math import pi, cos, sin

from kinematics import forward_kinematics, forward_kinematics_with_orientation, inverse_kinematics, visualize_arm

if 1:
    # Define joint angles
    # Test 1
    q1 = pi/2
    q2 = 0
    q3 = pi/2
    q4 = 0
    q5 = 3*pi/2
    q6 = pi/2

    # Compute forward kinematics using package
    # points = forward_kinematics(q1, q2, q3, q4, q5, q6)
    points, T0_1, T0_2, T0_3, T0_4, T0_5, T0_6 = forward_kinematics_with_orientation(q1, q2, q3, q4, q5, q6)
    print("Visualizing FK test 1 - press escape to move to next test")
    visualize_arm([T0_1, T0_2, T0_3, T0_4, T0_5, T0_6])

    # Test 2
    q1 = 0.34
    q2 = -1.2
    q3 = 1.0
    q4 = pi
    q5 = 2.24
    q6 = pi/7

    # Compute forward kinematics using package
    # points = forward_kinematics(q1, q2, q3, q4, q5, q6)
    points, T0_1, T0_2, T0_3, T0_4, T0_5, T0_6 = forward_kinematics_with_orientation(q1, q2, q3, q4, q5, q6)
    print("Visualizing FK test 2 - press escape to move to next test")
    visualize_arm([T0_1, T0_2, T0_3, T0_4, T0_5, T0_6])

    # Test 3
    q1 = 0
    q2 = 0
    q3 = 0
    q4 = 0
    q5 = pi
    q6 = pi

    # Compute forward kinematics using package
    # points = forward_kinematics(q1, q2, q3, q4, q5, q6)
    points, T0_1, T0_2, T0_3, T0_4, T0_5, T0_6 = forward_kinematics_with_orientation(q1, q2, q3, q4, q5, q6)
    print("Visualizing FK test 3 - press escape to move to next test")
    visualize_arm([T0_1, T0_2, T0_3, T0_4, T0_5, T0_6])



if 1:
    # ------------------------------------------------------------------------
    # Test 1
    T_target = np.array([[1,    0,      0,      5],
                        [0,    0,      1,      -4],
                        [0,    -1,     0,      4.5],
                        [0,    0,      0,      1]], dtype='float')
    
    angles = inverse_kinematics(T_target)

    if not angles:
        print("ERROR: no angles found for target T matrix\n{}".format(T_target))
        exit()

    # Angles from IK
    q1 = angles[0]
    q2 = angles[1]
    q3 = angles[2]
    q4 = angles[3]
    q5 = angles[4]
    q6 = angles[5]

    points, T0_1, T0_2, T0_3, T0_4, T0_5, T0_6 = forward_kinematics_with_orientation(q1, q2, q3, q4, q5, q6)

    # Verification angles
    q1_v = pi/2
    q2_v = 0
    q3_v = pi/2
    q4_v = 0
    q5_v = 3*pi/2
    q6_v = pi/2

    points_v, T0_1_v, T0_2_v, T0_3_v, T0_4_v, T0_5_v, T0_6_v = forward_kinematics_with_orientation(q1_v, q2_v, q3_v, q4_v, q5_v, q6_v)

    T_found = [T0_1, T0_2, T0_3, T0_4, T0_5, T0_6]
    T_verify = [T0_1_v, T0_2_v, T0_3_v, T0_4_v, T0_5_v, T0_6_v]

    print("Visualizing IK test 1 - press escape to move to next test")
    visualize_arm(T_found, T_verify)

    # ------------------------------------------------------------------------
    # Test 2
    T_target = np.array([[0.66425986,       -0.31715346,        0.67688442,         -3.30047602],
                        [0.74336259,       0.3754367,         -0.5535877,          2.37045682],
                        [-0.078555,        0.87089665,         0.48514755,         7.85705344],
                        [0,                0,                  0,                  1]], dtype='float')
    
    angles = inverse_kinematics(T_target)

    if not angles:
        print("ERROR: no angles found for target T matrix\n{}".format(T_target))
        exit()

    # Angles from IK
    q1 = angles[0]
    q2 = angles[1]
    q3 = angles[2]
    q4 = angles[3]
    q5 = angles[4]
    q6 = angles[5]

    points, T0_1, T0_2, T0_3, T0_4, T0_5, T0_6 = forward_kinematics_with_orientation(q1, q2, q3, q4, q5, q6)

    # Verification angles
    q1_v = 0.124
    q2_v = 1.52
    q3_v = -1
    q4_v = 2.45
    q5_v = 3.02
    q6_v = 1.11

    points_v, T0_1_v, T0_2_v, T0_3_v, T0_4_v, T0_5_v, T0_6_v = forward_kinematics_with_orientation(q1_v, q2_v, q3_v, q4_v, q5_v, q6_v)

    T_found = [T0_1, T0_2, T0_3, T0_4, T0_5, T0_6]
    T_verify = [T0_1_v, T0_2_v, T0_3_v, T0_4_v, T0_5_v, T0_6_v]

    print("Visualizing IK test 2 - press escape to move to next test")
    # visualize_arm(T_found, T_verify)
    visualize_arm(T_found, T_verify)


    # ------------------------------------------------------------------------
    # Test 3
    T_target = np.array([[0.78834994,       0.25530932,         0.55975131,         3.94934469],
                        [0.23199901,       0.71929276,         -0.65482394,        1.18015784],
                        [-0.56980772,      0.64609216,         0.50782289,         1.11017123],
                        [0,                0,                  0,                  1]], dtype='float')
    
    angles = inverse_kinematics(T_target)

    if not angles:
        print("ERROR: no angles found for target T matrix\n{}".format(T_target))
        exit()

    # Angles from IK
    q1 = angles[0]
    q2 = angles[1]
    q3 = angles[2]
    q4 = angles[3]
    q5 = angles[4]
    q6 = angles[5]

    points, T0_1, T0_2, T0_3, T0_4, T0_5, T0_6 = forward_kinematics_with_orientation(q1, q2, q3, q4, q5, q6)

    # Verification angles
    q1_v = 2.3
    q2_v = 0.987
    q3_v = -2.45
    q4_v = 3.11
    q5_v = -0.93
    q6_v = -2.33

    points_v, T0_1_v, T0_2_v, T0_3_v, T0_4_v, T0_5_v, T0_6_v = forward_kinematics_with_orientation(q1_v, q2_v, q3_v, q4_v, q5_v, q6_v)

    T_found = [T0_1, T0_2, T0_3, T0_4, T0_5, T0_6]
    T_verify = [T0_1_v, T0_2_v, T0_3_v, T0_4_v, T0_5_v, T0_6_v]

    print("Visualizing IK test 3 - press escape to move to next test")
    visualize_arm(T_found, T_verify)

