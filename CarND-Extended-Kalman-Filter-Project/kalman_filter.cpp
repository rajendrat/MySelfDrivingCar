#include "kalman_filter.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

#define PI           3.14159265358979323846  /* pi */

void KalmanFilter::Predict() {
	x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
   float rho = sqrt((x_(0) * x_(0)) + (x_(1) * x_(1)));
   float theta = 0;
   float rho_dot = 0;

  if ((fabs(x_(0) ) < 0.001)) {
     theta = 0;
  } else {
     theta = atan(x_(1) / x_(0));
  }

  if ((fabs(rho) < 0.001)) {
     rho_dot = 0;
  } else {
     rho_dot = ((x_(0) * x_(2)) + (x_(1) * x_(3) )) / rho;
  }

  std::cout << "rho : " << rho;
  std::cout << "theta : " << theta;

  VectorXd  h = VectorXd(3);
  h << rho, theta, rho_dot;

  VectorXd y = z - h;


// Normalization of y(1)

  while (y(1)>PI) 
    y(1) -= 2 * PI;

  while (y(1)<-PI)
    y(1) += 2 * PI;

  MatrixXd Ht  = H_.transpose();
  MatrixXd S   = H_ * P_ * Ht + R_;
  MatrixXd Si  = S.inverse();
  MatrixXd Pht = P_ * Ht;
  MatrixXd K   = Pht * Si;
 

// new estimate 
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;

}

