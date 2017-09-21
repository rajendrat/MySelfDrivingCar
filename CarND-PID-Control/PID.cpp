#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}


void PID::Init(double Kp, double Ki, double Kd) {

  //Initializing  Kp, Ki, Kd 
  this->Kp = Kp;
  this->Ki = Ki;
  this->Kd = Kd;

  return;
}

void PID::UpdateError(double cte) {

  //Update error values  
  d_error = cte - p_error; // old CTE error
  p_error = cte;   
  i_error += cte; //Cummulative CTEs

  return;
}

double PID::TotalError() {

  //Total Error
  return (-(Kp * p_error) - (Ki * i_error) - (Kd * d_error));
}
