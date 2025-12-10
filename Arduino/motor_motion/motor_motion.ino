#include <Arduino.h>

#include <ros.h>
#include <std_msgs/String.h>

#include <MsTimer2.h>
#include "motor_motion.h"

ros::NodeHandle nh;

// Use timer interrupt to activate motion close-loop control
const unsigned long Motion_control_period = 50;

// base RPM to drive motor
volatile int base_pwm = 85;

// types od traffic signs
enum TrafficSigns{
  SIGN_AHEAD_ONLY,
  SIGN_GENERAL_CAUTION,
  SIGN_NO_ENTRY,
  SIGN_PEDESTRIANS,
  SIGN_SPEED_LIMIT,
  SIGN_STOP,
  SIGN_TURN_LEFT,
  SIGN_TURN_RIGHT,
  SIGN_UNKNOWN
};

// types of motio modes
enum MotionMode {
  MODE_FORWARD,
  MODE_BACKWARD,
  MODE_STOP,
  MODE_TURN_LEFT,
  MODE_TURN_RIGHT
};

volatile MotionMode motion_mode = MODE_FORWARD;
volatile MotionMode previous_mode = MODE_FORWARD;

const unsigned long NO_SIGN_TIMEOUT_MS = 7000;
volatile unsigned long last_sign_ts = 0;

TrafficSigns classify_sign(const String& s) 
{
  if (s == "Stop") return SIGN_STOP;
  if (s == "Turn-left-ahead") return SIGN_TURN_LEFT;
  if (s == "Turn-right-ahead") return SIGN_TURN_RIGHT;
  if (s == "Pedestrians") return SIGN_PEDESTRIANS;
  if (s == "Ahead-only") return SIGN_AHEAD_ONLY;
  if (s == "General-caution") return SIGN_GENERAL_CAUTION;
  if (s == "No-entry") return SIGN_NO_ENTRY;

  // Speed limit: 20 - 120
  if (s.startsWith("Speed Limit"))
      return SIGN_SPEED_LIMIT;

  return SIGN_UNKNOWN;
}

int parse_speed_limit(const String& s)
{
  int last_space = s.lastIndexOf(' ');
  if (last_space < 0)
    return 0;
    
  return s.substring(last_space + 1).toInt();
}

void motion_callback(const std_msgs::String& msg)
{
  TrafficSigns sign = classify_sign(msg.data);

  last_sign_ts = millis(); 
  
  switch (sign) {
    case SIGN_STOP:
      motion_mode = MODE_STOP;
      base_pwm = 0;
      break;

    case SIGN_TURN_LEFT:
      motion_mode = MODE_TURN_LEFT;
      break;

    case SIGN_TURN_RIGHT:
      motion_mode = MODE_TURN_RIGHT;
      break;

    case SIGN_SPEED_LIMIT:
      base_pwm = parse_speed_limit(msg.data);
      motion_mode = MODE_FORWARD;
      break;

    default:
      motion_mode = MODE_FORWARD;
      break;
  }
}

ros::Subscriber<std_msgs::String> sub("valid_motion", motion_callback);

void motion_control_ISR()
{
  if (motion_mode != previous_mode)
    reset_ticks();
  
  long r = read_right_pulse();
  long l = read_left_pulse();

  const int max_comp = 80;
  const float Kp = 0.5f;
  const int ticks_tolerance = 5;

  int pwmR = 0, pwmL = 0;

  switch (motion_mode) {

    case MODE_STOP:
      pwmL = 0;
      pwmR = 0;
      break;

    case MODE_FORWARD: {
      long error = r - l;
      if (abs(error) <= ticks_tolerance) error = 0;

      int comp = (int)(Kp * error);
      comp = constrain(comp, -max_comp, max_comp);

      int left  = base_pwm + comp;
      int right = base_pwm - comp;

      left  = constrain(left,  0, 255);
      right = constrain(right, 0, 255);

      pwmL = left;
      pwmR = right;
      motor_control(pwmR, IN1, IN2, ENA);
       motor_control(pwmL, IN3, IN4, ENB);
  
      break;
    }

    case MODE_TURN_LEFT:
      turn_left(90, 15);
      break;

    case MODE_TURN_RIGHT:
      turn_right(90, 15);
      break;
  }
}



void setup() 
{
  motor_encoder_begin();

  MsTimer2::set(Motion_control_period, motion_control_ISR);
  MsTimer2::start();

  Serial.begin(19200);
  nh.initNode();
  nh.subscribe(sub);
}

void loop()
{
  nh.spinOnce();

  unsigned long now = millis();
  if (now - last_sign_ts > NO_SIGN_TIMEOUT_MS) {
    motion_mode = MODE_FORWARD;   
    base_pwm = 85;                
  }

  delay(100);
}
