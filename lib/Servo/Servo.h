#include <mbed.h>

class Servo {
private:
    // PWM output
    PinName pin; 
    PwmOut* pwm;
    const float period = 0.02f;
    int min_pulse_width = 530;
    int max_pulse_width = 2400;
    
    // Angle range (in degrees)
    const int min_angle = 0;
    const int max_angle = 180;
    const int home_angle = 90;
    int angle = home_angle;

public:
    // Constructors
    Servo(PinName _pin);
    Servo(PinName _pin, float _period, int min_pulse, int max_pulse, int _home_angle);
    Servo(PinName _pin, float _period, int min_pulse, int max_pulse, int _home_angle, int _min_angle, int _max_angle);
    // Destructor
    ~Servo();

    // Move servo to appropriate angle
    void set_angle(int degrees);

    void print(void);

    // user-input
    //void calibrate(void);
};