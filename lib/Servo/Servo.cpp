#include <Servo.h>
#include <mbed.h>
#include <stdio.h>

// Constructors
Servo::Servo(PinName _pin, float _period, int min_pulse, int max_pulse, int _home_angle, int _min_angle, int _max_angle) : pin(_pin), period(_period), home_angle(_home_angle), min_angle(_min_angle), max_angle(_max_angle) {
    min_pulse_width = min_pulse;
    max_pulse_width = max_pulse;
    
    pwm = new PwmOut(pin);
    pwm->period(period);
    set_angle(home_angle);
}
Servo::Servo(PinName _pin, float _period, int min_pulse, int max_pulse, int _home_angle) : Servo::Servo(_pin, _period, min_pulse, max_pulse, _home_angle, 0, 180) { }
Servo::Servo(PinName _pin) : Servo::Servo(_pin, 0.02f, 600, 2300, 90) { }

// Destructor
Servo::~Servo() {
    pwm->suspend();
    delete pwm;
}

// Mutator Functions
void Servo::set_angle(int degrees) {
    if (degrees > max_angle || degrees < min_angle) {
        printf("Error: %d degrees is out of servo range [%d, %d]\n", degrees, min_angle, max_angle);
        return;
    }

    // Mapping from pulse range [min_pulse, max_pulse] to degree range [min_angle, max_angle]
    double slope = static_cast<double>(max_pulse_width - min_pulse_width) / (max_angle - min_angle);
    int pulse = round(min_pulse_width + slope * (degrees - min_angle)); 

    // move servo
    pwm->pulsewidth_us(pulse);
    angle = degrees;
}

void Servo::print(void) {
    printf("Servo: pin %d\n", pin);
    printf("- pulse [%d, %d] us\n", min_pulse_width, max_pulse_width);
    printf("- angle [%d, %d] degrees\n", min_angle, max_angle);
    printf("- current angle: %d\n", angle);
}