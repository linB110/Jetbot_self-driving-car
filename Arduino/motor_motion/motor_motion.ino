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

  
  
  switch (sign) {

    case SIGN_STOP:
      base_pwm = 0;
      stop_motors();
      break;

    case SIGN_TURN_LEFT:
      turn_left();
      break;

    case SIGN_TURN_RIGHT:
      turn_right();
      break;

    case SIGN_SPEED_LIMIT:
      // parse speed value
      base_pwm = parse_speed_limit(msg.data);
      break;

    default:
      move_forward();
      break;
  }
}

ros::Subscriber<std_msgs::String> sub("valid_motion", motion_callback);

void motion_control_ISR()
{
  long r = read_right_pulse();
  long l = read_left_pulse();

  int pwmR = 0, pwmL = 0;
  const int max_comp = 80;     
  const float Kp = 0.5f;
  const int ticks_tolerance = 5;

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
  int l  = read_left_rpm();
  int r = read_right_rpm();

  delay(100);
}
