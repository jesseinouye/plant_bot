clear;

% 
% syms q1 q2 q3 q4 q5 q6
% 
% 
% 
% 
% L1 = 2;
% L2 = 4;
% L3 = 2;
% L4 = 3;
% L5 = 1.5;
% L5p = 1;
% L6 = 5;
% 
% R11 = 0;
% R21 = 0;
% R31 = 1;
% R12 = 0;
% R22 = 1;
% R32 = 0;
% R13 = 1;
% R23 = 0;
% R33 = 0;
% T1 = 1;
% T2 = 0;
% T3 = 14.5;
% 
% 
% % Home configuration (all q's 0)
% % T0_6:
% % [[-6.12323400e-17 -6.12323400e-17  1.00000000e+00  1.00000000e+00]
% %  [ 1.83697020e-16 -1.00000000e+00 -6.12323400e-17  1.01033361e-15]
% %  [ 1.00000000e+00  1.83697020e-16  6.12323400e-17  1.45000000e+01]
% %  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
% 
% 
% t1 = q1;
% t2 = (pi/2)+q2;
% t3 = (-pi/2)+q3;
% t4 = q4;
% t5 = (-pi/2)+q5;
% t6 = pi+q6;
% 
% T11 = cos(t1)*cos(t4)*cos(t5)*cos(t6)*cos(t2+t3) - sin(t1)*sin(t4)*cos(t5)*cos(t6) - cos(t1)*sin(t5)*cos(t6)*sin(t2+t3) - cos(t1)*sin(t4)*sin(t6)*cos(t2+t3) - sin(t1)*cos(t4)*sin(t6) == R11;
% T21 = sin(t1)*cos(t4)*cos(t5)*cos(t6)*cos(t2+t3) + cos(t1)*sin(t4)*cos(t5)*cos(t6) - sin(t1)*sin(t5)*cos(t6)*sin(t2+t3) - sin(t1)*sin(t4)*sin(t6)*cos(t2+t3) + cos(t1)*cos(t4)*sin(t6) == R21;
% T31 = cos(t4)*cos(t5)*cos(t6)*sin(t2+t3) + sin(t5)*cos(t6)*cos(t2+t3) - sin(t4)*sin(t6)*sin(t2+t3) == R31;
% 
% T12 = -cos(t1)*cos(t4)*cos(t5)*sin(t6)*cos(t2+t3) + sin(t1)*sin(t4)*cos(t5)*sin(t6) + cos(t1)*sin(t5)*sin(t6)*sin(t2+t3) - cos(t1)*sin(t4)*cos(t6)*cos(t2+t3) - sin(t1)*cos(t4)*cos(t6) == R12;
% T22 = -sin(t1)*cos(t4)*cos(t5)*cos(t6)*cos(t2+t3) - cos(t1)*sin(t4)*cos(t5)*sin(t6) + sin(t1)*sin(t5)*sin(t6)*sin(t2+t3) - sin(t1)*sin(t4)*sin(t6)*cos(t2+t3) + cos(t1)*cos(t4)*cos(t6) == R22;
% T32 = -cos(t4)*cos(t5)*sin(t6)*sin(t2+t3) - sin(t5)*sin(t6)*cos(t2+t3) - sin(t4)*cos(t6)*sin(t2+t3) == R32;
% 
% T13 = -cos(t1)*cos(t4)*sin(t5)*cos(t2+t3) + sin(t1)*sin(t4)*sin(t5) - cos(t1)*cos(t5)*sin(t2+t3) == R13;
% T23 = -sin(t1)*cos(t4)*sin(t5)*cos(t2+t3) - cos(t1)*sin(t4)*sin(t5) - sin(t1)*cos(t5)*sin(t2+t3) == R23;
% T33 = -cos(t4)*sin(t5)*sin(t2+t3) + cos(t5)*cos(t2+t3) == R33;
% 
% T14 = L6*(cos(t1)*cos(t4)*cos(t5)*cos(t6)*cos(t2+t3) - sin(t1)*sin(t4)*cos(t5)*cos(t6) - cos(t1)*sin(t5)*cos(t6)*sin(t2+t3) - cos(t1)*sin(t4)*sin(t6)*cos(t2+t3) - sin(t1)*cos(t4)*sin(t6))...
%       + L5p*(-cos(t1)*cos(t4)*cos(t5)*cos(t2+t3) + sin(t1)*sin(t4)*sin(t5) - cos(t1)*cos(t5)*sin(t2+t3)) + L5*(cos(t1)*cos(t4)*cos(t5)*cos(t2+t3) - sin(t1)*sin(t4)*sin(t5) - cos(t1)*sin(t5)*sin(t2+t3))...
%       + (L3+L4)*(-cos(t1)*sin(t2+t3)) + L2*(cos(t1)*cos(t2)) == T1;
% 
% T24 = L6*(sin(t1)*cos(t4)*cos(t5)*cos(t6)*cos(t2+t3) + cos(t1)*sin(t4)*cos(t5)*cos(t6) - sin(t1)*sin(t5)*cos(t6)*sin(t2+t3) - sin(t1)*sin(t4)*sin(t6)*cos(t2+t3) + cos(t1)*cos(t4)*cos(t6))...
%       + L5p*(-sin(t1)*cos(t4)*sin(t5)*cos(t2+t3) - cos(t1)*sin(t4)*sin(t5) - sin(t1)*cos(t5)*sin(t2+t3)) + L5*(sin(t1)*cos(t4)*cos(t5)*cos(t2+t3) + cos(t1)*sin(t4)*cos(t5) - sin(t1)*sin(t5)*sin(t2+t3))...
%       + (L3+L4)*(-sin(t1)*sin(t2+t3)) + L2*(sin(t1)*cos(t2)) == T2;
% 
% T34 = L6*(cos(t4)*cos(t5)*cos(t6)*sin(t2+t3) + sin(t5)*cos(t6)*cos(t2+t3) - sin(t4)*sin(t6)*sin(t2+t3)) + L5p*(-cos(t4)*sin(t5)*sin(t2+t3) + cos(t5)*cos(t2+t3))...
%       + L5*(cos(t4)*cos(t5)*sin(t2+t3) + sin(t5)*cos(t2+t3)) + (L3+L4)*cos(t2+t3) + L2*sin(t2) + L1 == T3;
% 
% eqns = [T11, T21, T31, T12, T22, T32, T13, T23, T33, T14, T24, T34];
% 
% S = solve(eqns, [q1, q2, q3, q4, q5, q6])



% ---------------------------------------------------------------------------------------

% syms q1;
% 
% L1 = 2;
% 
% % Home configuration
% % R11 = 1;
% % R21 = 0;
% % R31 = 0;
% % R12 = 0;
% % R22 = 0;
% % R32 = 1;
% % R13 = 0;
% % R23 = -1;
% % R33 = 0;
% % T1 = 0;
% % T2 = 0;
% % T3 = 2;
% 
% 
% % Not home config -> q1 = pi/4
% R11 = 0.7071067812;
% R21 = 0.7071067812;
% R31 = 0;
% R12 = 0;
% R22 = 0;
% R32 = 1;
% R13 = 0.7071067812;
% R23 = -0.7071067812;
% R33 = 0;
% T1 = 0;
% T2 = 0;
% T3 = 2;
% 
% 
% t1 = q1;
% 
% T11 = cos(t1) == R11;
% T21 = sin(t1) == R21;
% T31 = 0 == R31;
% T12 = 0 == R12;
% T22 = 0 == R22;
% T32 = 1 == R32;
% T13 = sin(t1) == R13;
% T23 = -cos(t1) == R23;
% T33 = 0 == R33;
% T14 = 0 == T1;
% T24 = 0 == T2;
% T34 = L1 == T3;
% 
% % eqns = [T11, T21, T31, T12, T22, T32, T13, T23, T33, T14, T24, T34];
% eqns = [T11, T21, T13, T23];
% 
% S = solve(eqns, q1)




% ---------------------------------------------------------------------------------------

% syms q1;
% 
% R11 = 0.5;
% 
% t1 = q1;
% 
% T11 = cos(t1) == R11;
% 
% eqns = [T11];
% 
% S = solve(eqns, q1)




% ---------------------------------------------------------------------------------------






% % syms t1 t2 t3 t4 t5 t6;
% syms q1 q2 q3 q4 q5 q6
% 
% 
% 
% 
% L1 = 2;
% L2 = 4;
% L3 = 2;
% L4 = 3;
% L5 = 1.5;
% L5p = 1;
% L6 = 5;
% 
% R11 = 0;
% R21 = 0;
% R31 = 1;
% R12 = -1;
% R22 = 0;
% R32 = 0;
% R13 = 0;
% R23 = -1;
% R33 = 0;
% T1 = 0;
% T2 = 0;
% T3 = 6;
% 
% 
% % Home configuration (all q's 0)
% % T0_6:
% % [[-6.12323400e-17 -6.12323400e-17  1.00000000e+00  1.00000000e+00]
% %  [ 1.83697020e-16 -1.00000000e+00 -6.12323400e-17  1.01033361e-15]
% %  [ 1.00000000e+00  1.83697020e-16  6.12323400e-17  1.45000000e+01]
% %  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
% 
% 
% t1 = q1;
% t2 = (pi/2)+q2;
% t3 = (-pi/2)+q3;
% t4 = q4;
% t5 = (-pi/2)+q5;
% t6 = pi+q6;
% 
% T11 = cos(t1)*cos(t2) == R11;
% T21 = sin(t1)*cos(t2) == R21;
% T31 = sin(t2) == R31;
% T12 = -cos(t1)*sin(t2) == R12;
% T22 = -sin(t1)*sin(t2) == R22;
% T32 = cos(t2) == R32;
% T13 = sin(t1) == R13;
% T23 = -cos(t1) == R23;
% T14 = L2*cos(t1)*cos(t2) == T1;
% T24 = L2*sin(t1)*cos(t2) == T2;
% T34 = L2*sin(t2) + L1 == T3;
% 
% 
% eqns = [T11, T21, T31, T12, T22, T32, T13, T23, T14, T24, T34];
% 
% % S = solve(T11, T21, T31, T12, T22, T32, T13, T23, T33, T14, T24, T34, t1, t2, t3, t4, t5, t6)
% 
% S = solve(eqns, [q1, q2])
% % 
% % S.q1










% ---------------------------------------------------------------------------------------





syms q1 q2 q3 q4 q5 q6;

L1 = 2;
L2 = 4;
L3 = 2;
L4 = 3;
L5 = 1.5;
L5p = 1;
L6 = 5;

R11 = 0;
R21 = -1;
R31 = 0;
R12 = 1;
R22 = 0;
R32 = 0;
R13 = 0;
R23 = 0;
R33 = 1;
T1 = 0;
T2 = -8.5;
T3 = 7;


% Home configuration (all q's 0)
% T0_6:
% [[-6.12323400e-17 -6.12323400e-17  1.00000000e+00  1.00000000e+00]
%  [ 1.83697020e-16 -1.00000000e+00 -6.12323400e-17  1.01033361e-15]
%  [ 1.00000000e+00  1.83697020e-16  6.12323400e-17  1.45000000e+01]
%  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]


t1 = q1;
t2 = (pi/2)+q2;
t3 = (-pi/2)+q3;
t4 = q4;
t5 = (-pi/2)+q5;
t6 = pi+q6;

T0_1 = [cos(t1) 0       sin(t1)  0
        sin(t1) 0       -cos(t1) 0
        0       1       0        L1
        0       0       0        1];

T1_2 = [cos(t2) -sin(t2) 0      L2*cos(t2)
        sin(t2) cos(t2)  0      L2*sin(t2)
        0       0        1      0
        0       0        0      1];

T2_3 = [cos(t3) 0       -sin(t3) 0
        sin(t3) 0       cos(t3)  0
        0       -1      0        0
        0       0       0        1];

T3_4 = [cos(t4) 0       sin(t4)  0
        sin(t4) 0       -cos(t4) 0
        0       1       0        L3+L4
        0       0       0        1];

T4_5 = [cos(t5) 0       -sin(t5) L5*cos(t5)
        sin(t5) 0       cos(t5)  L5*sin(t5)
        0       -1      0        0
        0       0       0        1];

T5_6 = [cos(t6) -sin(t6) 0      L6*cos(t6)
        sin(t6) cos(t6)  0      L6*sin(t6)
        0       0        1      L5p
        0       0        0      1];



T0_2 = T0_1 * T1_2
% T0_5 = T0_1 * T1_2 * T2_3 * T3_4 * T4_5
% T0_6 = T0_1 * T1_2 * T2_3 * T3_4 * T4_5 * T5_6


T11 = T0_6(1,1) == R11;
T21 = T0_6(2,1) == R21;
T31 = T0_6(3,1) == R31;

T12 = T0_6(1,2) == R12;
T22 = T0_6(2,2) == R22;
T32 = T0_6(3,2) == R32;

T13 = T0_6(1,3) == R13;
T23 = T0_6(2,3) == R23;
T33 = T0_6(3,3) == R33;

T14 = T0_6(1,4) == T1;
T24 = T0_6(2,4) == T2;
T34 = T0_6(3,4) == T3;


% T11 = T0_6(1,1) == R11;
% T21 = T0_6(2,1) == R21;
% T31 = T0_6(3,1) == R31;
% 
% T12 = T0_6(1,2) == R12;
% T22 = T0_6(2,2) == R22;
% T32 = T0_6(3,2) == R32;
% 
% T13 = T0_6(1,3) == R13;
% T23 = T0_6(2,3) == R23;
% T33 = T0_6(3,3) == R33;
% 
% T14 = T0_6(1,4) == T1;
% T24 = T0_6(2,4) == T2;
% T34 = T0_6(3,4) == T3;




% T11 = cos(t1)*cos(t4)*cos(t5)*cos(t6)*cos(t2+t3) - sin(t1)*sin(t4)*cos(t5)*cos(t6) - cos(t1)*sin(t5)*cos(t6)*sin(t2+t3) - cos(t1)*sin(t4)*sin(t6)*cos(t2+t3) - sin(t1)*cos(t4)*sin(t6) == R11;
% T21 = sin(t1)*cos(t4)*cos(t5)*cos(t6)*cos(t2+t3) + cos(t1)*sin(t4)*cos(t5)*cos(t6) - sin(t1)*sin(t5)*cos(t6)*sin(t2+t3) - sin(t1)*sin(t4)*sin(t6)*cos(t2+t3) + cos(t1)*cos(t4)*sin(t6) == R21;
% T31 = cos(t4)*cos(t5)*cos(t6)*sin(t2+t3) + sin(t5)*cos(t6)*cos(t2+t3) - sin(t4)*sin(t6)*sin(t2+t3) == R31;
% 
% T12 = -cos(t1)*cos(t4)*cos(t5)*sin(t6)*cos(t2+t3) + sin(t1)*sin(t4)*cos(t5)*sin(t6) + cos(t1)*sin(t5)*sin(t6)*sin(t2+t3) - cos(t1)*sin(t4)*cos(t6)*cos(t2+t3) - sin(t1)*cos(t4)*cos(t6) == R12;
% T22 = -sin(t1)*cos(t4)*cos(t5)*cos(t6)*cos(t2+t3) - cos(t1)*sin(t4)*cos(t5)*sin(t6) + sin(t1)*sin(t5)*sin(t6)*sin(t2+t3) - sin(t1)*sin(t4)*sin(t6)*cos(t2+t3) + cos(t1)*cos(t4)*cos(t6) == R22;
% T32 = -cos(t4)*cos(t5)*sin(t6)*sin(t2+t3) - sin(t5)*sin(t6)*cos(t2+t3) - sin(t4)*cos(t6)*sin(t2+t3) == R32;
% 
% T13 = -cos(t1)*cos(t4)*sin(t5)*cos(t2+t3) + sin(t1)*sin(t4)*sin(t5) - cos(t1)*cos(t5)*sin(t2+t3) == R13;
% T23 = -sin(t1)*cos(t4)*sin(t5)*cos(t2+t3) - cos(t1)*sin(t4)*sin(t5) - sin(t1)*cos(t5)*sin(t2+t3) == R23;
% T33 = -cos(t4)*sin(t5)*sin(t2+t3) + cos(t5)*cos(t2+t3) == R33;
% 
% T14 = L6*(cos(t1)*cos(t4)*cos(t5)*cos(t6)*cos(t2+t3) - sin(t1)*sin(t4)*cos(t5)*cos(t6) - cos(t1)*sin(t5)*cos(t6)*sin(t2+t3) - cos(t1)*sin(t4)*sin(t6)*cos(t2+t3) - sin(t1)*cos(t4)*sin(t6))...
%       + L5p*(-cos(t1)*cos(t4)*cos(t5)*cos(t2+t3) + sin(t1)*sin(t4)*sin(t5) - cos(t1)*cos(t5)*sin(t2+t3)) + L5*(cos(t1)*cos(t4)*cos(t5)*cos(t2+t3) - sin(t1)*sin(t4)*sin(t5) - cos(t1)*sin(t5)*sin(t2+t3))...
%       + (L3+L4)*(-cos(t1)*sin(t2+t3)) + L2*(cos(t1)*cos(t2)) == T1;
% 
% T24 = L6*(sin(t1)*cos(t4)*cos(t5)*cos(t6)*cos(t2+t3) + cos(t1)*sin(t4)*cos(t5)*cos(t6) - sin(t1)*sin(t5)*cos(t6)*sin(t2+t3) - sin(t1)*sin(t4)*sin(t6)*cos(t2+t3) + cos(t1)*cos(t4)*cos(t6))...
%       + L5p*(-sin(t1)*cos(t4)*sin(t5)*cos(t2+t3) - cos(t1)*sin(t4)*sin(t5) - sin(t1)*cos(t5)*sin(t2+t3)) + L5*(sin(t1)*cos(t4)*cos(t5)*cos(t2+t3) + cos(t1)*sin(t4)*cos(t5) - sin(t1)*sin(t5)*sin(t2+t3))...
%       + (L3+L4)*(-sin(t1)*sin(t2+t3)) + L2*(sin(t1)*cos(t2)) == T2;
% 
% T34 = L6*(cos(t4)*cos(t5)*cos(t6)*sin(t2+t3) + sin(t5)*cos(t6)*cos(t2+t3) - sin(t4)*sin(t6)*sin(t2+t3)) + L5p*(-cos(t4)*sin(t5)*sin(t2+t3) + cos(t5)*cos(t2+t3))...
%       + L5*(cos(t4)*cos(t5)*sin(t2+t3) + sin(t5)*cos(t2+t3)) + (L3+L4)*cos(t2+t3) + L2*sin(t2) + L1 == T3;
% 



% eqns = [T11, T21, T31, T12, T22, T32, T13, T23, T33, T14, T24, T34];
% eqns = [T11, T21, T31, T12, T22, T32];
% S = solve(eqns, [q1, q2, q3, q4, q5, q6])

% eqns = [T11, T21, T31, T12, T22, T32, T13, T23, T33, T14, T24, T34];
% S = solve(eqns, [q1, q2, q3, q4, q5, q6], 'IgnoreAnalyticConstraints', true)




% TEST1 = sin(q4)*sin(q6)*(cos(q2 + pi/2)*sin(q3 - pi/2) + cos(q3 - pi/2)*sin(q2 + pi/2))...
%         - cos(q6)*(sin(q5 - pi/2)*(cos(q2 + pi/2)*cos(q3 - pi/2) - sin(q2 + pi/2)*sin(q3 - pi/2))...
%         + cos(q4)*cos(q5 - pi/2)*(cos(q2 + pi/2)*sin(q3 - pi/2) + cos(q3 - pi/2)*sin(q2 + pi/2))) == R31;
% 
% eqns = [TEST1];
% S = solve(eqns, [q1, q2, q3, q4, q5, q6])



























