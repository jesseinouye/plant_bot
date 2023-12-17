#include <mbed.h>
#include <Matrix.h>
#include <stdio.h>
#include <Servo.h>

#define N_DOF 6



int main(void) {

    float theta[N_DOF];

    Matrix T_0_1(4, 4, {
        0, 0, 0, 0,
        0, 

    });

    //// Forward Kinematics
    /*
    // Inputs
    
        Vector Theta, 6x1
    // Outputs
        -> Move Servos

        Matrix Position, 6x3
        Vector Force, 6x1
        Vector Tau, 6x1
        Vector Omega, 6x1 
    */

   //// Inverse Kinematics
   /*
   // Inputs
    
        Matrix Position, 6x3
    // Outputs
        Vector Theta, 6x1
        Vector Force, 6x1
        Vector Tau, 6x1
        Vector Omega, 6x1

        -> Move Servos
    */

    // Matrix J(6, N_DOF);

    Servo s1(PA_8, 0.02f, 560, 2440, 90);
    Servo s2(PC_6, 0.02f, 580, 2570, 90);

    s1.print();
    s2.print();

    bool toggle = true;
    int angle = 90;
    thread_sleep_for(2000);
    s1.set_angle(90);
    s2.set_angle(90);
    while(true) {
        // if (toggle) {
        //     for (angle; angle < 180; angle += 1) {
        //         s1.set_angle(angle);
        //         s2.set_angle(angle);
        //         thread_sleep_for(50);
        //     }
        //     toggle = false;
        //     thread_sleep_for(1000);
        // } 
        // else {
        //     for (angle; angle > 0; angle -= 1) {
        //         s1.set_angle(angle);
        //         s2.set_angle(angle);
        //         thread_sleep_for(50);
        //     }
        //     toggle = true;
        //     thread_sleep_for(1000);
        // }
    }

    return 0;
}