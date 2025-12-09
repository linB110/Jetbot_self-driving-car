#pragma once
#include <Arduino.h>
#include <stdint.h>

// ---------- Motor pins ----------
extern const int ENA; // left wheel (PWM)
extern const int IN1;
extern const int IN2;
extern const int ENB; // right wheel (PWM)
extern const int IN3;
extern const int IN4;

// ---------- Encoder pins ----------
extern const int ENCODER_R_A;
extern const int ENCODER_R_B;
extern const int ENCODER_L_A;
extern const int ENCODER_L_B;

// ---------- Params ----------
#ifndef PPR   // pulse per revolution
#define PPR 960.0f
#endif

#ifndef SPEED_WINDOW_MS
#define SPEED_WINDOW_MS 500
#endif

// initialization
void motor_encoder_begin();

// read RPM
float read_left_rpm();
float read_right_rpm();

// read imuplse
int read_right_pulse();
int read_left_pulse();
long get_diff_ticks();
void reset_ticks();

// update ticks
void update_right_ticks();
void update_left_ticks();

// motor control (abstract)
void motor_control(int pwm, int inA, int inB, int enable);

// motor control (closed loop)
void move_forward();
void move_backward();
void turn_right();
void turn_left();
void rotate_cw(int pwm = 125);
void rotate_ccw(int pwm = 125);
void stop_motors();
