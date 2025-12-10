#include "motor_motion.h"

// ---------- Motor pins ----------
const int ENA = 6; // right wheel (PWM)
const int IN1 = 7;
const int IN2 = 8;
const int ENB = 9; // left wheel (PWM)
const int IN3 = 10;
const int IN4 = 11;

// ---------- Encoder pins ----------
const int ENCODER_R_A = 2;
const int ENCODER_R_B = 4;
const int ENCODER_L_A = 3;
const int ENCODER_L_B = 5;

// ---------- Encoder counters ----------
volatile long right_ticks = 0;
volatile long left_ticks  = 0;

// ---------- Speed tracking ----------
unsigned long last_left_speed_ts  = 0;
unsigned long last_right_speed_ts = 0;
long last_right_ticks = 0;
long last_left_ticks  = 0;

// closed-loop control variable
const int ticks_tolerance = 10;
const int ticks_turning_diff = 300;
const float Kp = 0.8f;

// ==================================================
// initialization
// ==================================================
void motor_encoder_begin()
{
  // Motor pins
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  // Encoder pins
  pinMode(ENCODER_R_A, INPUT_PULLUP);
  pinMode(ENCODER_R_B, INPUT_PULLUP);
  pinMode(ENCODER_L_A, INPUT_PULLUP);
  pinMode(ENCODER_L_B, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(ENCODER_R_A), update_right_ticks, RISING);
  attachInterrupt(digitalPinToInterrupt(ENCODER_L_A), update_left_ticks,  RISING);

  last_left_speed_ts  = millis();
  last_right_speed_ts = millis();
}


void update_right_ticks()
{
  if (digitalRead(ENCODER_R_B) == HIGH) {
    right_ticks++;
  } else {
    right_ticks--;
  }
}

void update_left_ticks()
{
  if (digitalRead(ENCODER_L_B) == HIGH) {
    left_ticks++;
  } else {
    left_ticks--;
  }
}

// ==================================================
// read RPM data
// ==================================================
float read_left_rpm()
{
  noInterrupts();
  long l_ticks = left_ticks;
  interrupts();

  long dl = l_ticks - last_left_ticks;
  last_left_ticks = l_ticks;

  const float window_s = float(SPEED_WINDOW_MS) / 1000.0f;
  const float l_cps = dl / window_s;
  const float left_rpm = (l_cps / PPR) * 60.0f;

  last_left_speed_ts = millis();
  return left_rpm;
}

float read_right_rpm()
{
  noInterrupts();
  long r_ticks = right_ticks;
  interrupts();

  long dr = r_ticks - last_right_ticks;
  last_right_ticks = r_ticks;

  const float window_s = float(SPEED_WINDOW_MS) / 1000.0f;
  const float r_cps = dr / window_s;
  const float right_rpm = (r_cps / PPR) * 60.0f;

  last_right_speed_ts = millis();
  return right_rpm;
}

// ==================================================
// read impulse of encoder
// ==================================================

int read_right_pulse()
{
  noInterrupts();
  long r_ticks = right_ticks;
  interrupts();

  return r_ticks;
}

int read_left_pulse()
{
  noInterrupts();
  long l_ticks = left_ticks;
  interrupts();

  return l_ticks;
}

long get_diff_ticks() 
{
  long r = read_right_pulse();
  long l = read_left_pulse();
  return r - l;   
}

void reset_ticks()
{
  right_ticks = 0;
  left_ticks = 0;
}

// ==================================================
// motor control (open loop)
// ==================================================
void motor_control(int pwm, int inA, int inB, int enable)
{
  const bool isForward = (pwm >= 0);
  int pwmVal = constrain(abs(pwm), 0, 255);

  digitalWrite(inA, isForward ? LOW : HIGH);
  digitalWrite(inB, isForward ? HIGH : LOW);

  analogWrite(enable, pwmVal);
}

void stop_motors()
{
  motor_control(0, IN1, IN2, ENA);
  motor_control(0, IN3, IN4, ENB);
}

// ==================================================
// motor control (closed loop)
// ==================================================
void move_forward()
{
  long r_pulse = read_right_pulse();
  long l_pulse = read_left_pulse();
  
  const int diff = r_pulse - l_pulse;
  int error = diff;
  if (abs(error) <= ticks_tolerance) 
    error = 0;

  const int max_comp = 60;
  const int min_pwm = 110;
  int comp = (int)(Kp * error);
  comp = constrain(comp, -max_comp, max_comp);

  int pwm_left  = min_pwm + comp;
  int pwm_right = min_pwm - comp;

  pwm_left  = max(pwm_left,  min_pwm);
  pwm_right = max(pwm_right, min_pwm);

  motor_control(pwm_right, IN1, IN2, ENA);  
  motor_control(pwm_left,  IN3, IN4, ENB);
}

void move_backward()
{
  long r_pulse = read_right_pulse();
  long l_pulse = read_left_pulse();

  const int diff = r_pulse - l_pulse;  
  int error = diff;                     
  if (abs(error) <= ticks_tolerance) error = 0;

  const int max_comp = 60;
  const int min_pwm  = 80;
  int comp = (int)(Kp * error);
  comp = constrain(comp, -max_comp, max_comp);

  int pwm_left_mag  = min_pwm - comp;   
  int pwm_right_mag = min_pwm + comp;

  pwm_left_mag  = max(pwm_left_mag,  min_pwm);
  pwm_right_mag = max(pwm_right_mag, min_pwm);

  motor_control(-pwm_right_mag, IN1, IN2, ENA);
  motor_control(-pwm_left_mag,  IN3, IN4, ENB);
}

void turn_right(int base_pwm, int diff)
{
  motor_control(base_pwm + diff, IN1, IN2, ENA);
  motor_control(base_pwm,  IN3, IN4, ENB);
}

void turn_left(int base_pwm, int diff)
{
  motor_control(base_pwm, IN1, IN2, ENA);
  motor_control(base_pwm + diff, IN3, IN4, ENB);
}

//void turn_right(int base_pwm)
//{
//  long r_pulse = read_right_pulse();
//  long l_pulse = read_left_pulse();
//
//  long diff   = r_pulse - l_pulse;           
//  long target = -ticks_turning_diff;          
//  long error  = diff - target;
//  if (abs(error) <= ticks_tolerance) error = 0;
//
//  const int max_comp = 60;
//
//  int comp = (int)(Kp * error);
//  comp = constrain(comp, -max_comp, max_comp);
//
//  int pwm_left  = base_pwm + comp;  
//  int pwm_right = base_pwm - comp;
//
//  pwm_left  = max(pwm_left,  base_pwm);
//  pwm_right = max(pwm_right, base_pwm);
//
//  motor_control(pwm_right, IN1, IN2, ENA);
//  motor_control(pwm_left,  IN3, IN4, ENB);
//}

//void turn_left(int base_pwm)
//{
//  long r_pulse = read_right_pulse();
//  long l_pulse = read_left_pulse();
//
//  long diff   = r_pulse - l_pulse;           
//  long target = +ticks_turning_diff;          
//  long error  = diff - target;
//  if (abs(error) <= ticks_tolerance) error = 0;
//
//  const int max_comp = 60;
//
//  int comp = (int)(Kp * error);
//  comp = constrain(comp, -max_comp, max_comp);
//
//  int pwm_left  = base_pwm + comp;
//  int pwm_right = base_pwm - comp;
//
//  pwm_left  = max(pwm_left,  base_pwm);
//  pwm_right = max(pwm_right, base_pwm);
//
//  motor_control(pwm_right, IN1, IN2, ENA);
//  motor_control(pwm_left,  IN3, IN4, ENB);
//}

void rotate_cw(int pwm)
{
  const int base_pwm = 85;
  pwm = constrain(pwm, base_pwm, 255);

  motor_control(-pwm, IN1, IN2, ENA);
  motor_control(pwm, IN3, IN4, ENB);
}

void rotate_ccw(int pwm)
{
  const int base_pwm = 85;
  pwm = constrain(pwm, base_pwm, 255);

  motor_control(pwm, IN1, IN2, ENA);
  motor_control(-pwm, IN3, IN4, ENB);
}

void rotate_angle(float angle)
{
  if (fabs(angle) < 0.5f)
    return;

  const long target_ticks = (long)(fabs(angle) * PPA);
  reset_ticks();

  const bool cw = (angle > 0);
  const int fast_pwm = 105;   
  const int slow_pwm = 85;    
  const long slow_region = (long)(0.15f * target_ticks); 

  while (true) {
    long d = labs(get_diff_ticks());
    long remaining = target_ticks - d;

    if (remaining <= 0) {
      break;   
    }

    int pwm;
    if (remaining > slow_region) {
      pwm = fast_pwm;
    } else {
      float ratio = (float)remaining / (float)slow_region;  
      pwm = (int)(slow_pwm + (fast_pwm - slow_pwm) * ratio);
    }

    if (cw) {
      rotate_cw(pwm);
    } else {
      rotate_ccw(pwm);
    }

    delay(3);
  }

  stop_motors();
}
