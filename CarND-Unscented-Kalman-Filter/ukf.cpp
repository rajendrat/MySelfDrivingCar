#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  
  // initialized and timestamp
  is_initialized_ = false;
  previous_timestamp_ = 0;
	
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  x_ = VectorXd(5);

  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate nu psi] in SI units and rad
  x_aug_ = VectorXd(7);

  // state covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  
  // aug state covariance matrix
  P_aug_ = MatrixXd(n_aug_, n_aug_);

  // Sigma Points
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Initialize the other Parameters from ukf.h
  time_us_ = 0.0;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.0;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03; 

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  // Weights of sigma points
  weights_ = VectorXd(2*n_aug_+1);

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    // first measurement
    cout << "UKF: " << endl;
    
    // initialize state vector
    x_ << 0, 0, 0, 0, 0;
	  
    // initialize augmented state
    x_aug_.head(5) = x_;
    x_aug_(5) = 0;
    x_aug_(6) = 0;
		
    // intialized state covariance matrix P 
    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1000, 0, 0,
          0, 0, 0, 1000, 0,
          0, 0, 0, 0, 1;

    // initialized augmented state covariance matrix P
    P_aug_.fill(0.0);
    P_aug_.topLeftCorner(n_x_,n_x_) = P_;
    P_aug_(5,5) = std_a_*std_a_;
    P_aug_(6,6) = std_yawdd_*std_yawdd_;
		
    //Initialize weights
	  weights_(0) = lambda_/(lambda_ + n_aug_);
	  for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
	    weights_(i) = 0.5/(n_aug_+lambda_);
	  }

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

      // Give x_ the RADAR measurement
      //Convert RADAR from polar to cartesian coordinates
      rho_in = meas_package.raw_measurements_(0); 
      phi_in = meas_package.raw_measurements_(1); 
      rho_dot_in = meas_package.raw_measurements_(2);
      double vx_in = rho_dot_in * cos(phi_in);
      double vy_in = rho_dot_in * sin(phi_in);

      x_ << rho_in * cos(phi_in), rho_in * sin(phi_in), sqrt(vx_in*vx_in + vy_in*vy_in), 0, 0; 
	
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      
      //Give x_ the LASER measurement
      x_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 0, 0, 0;
    }
	
    // Check for zeros in the initial values of px and py
    if (fabs(x_(0)) < 0.001 and fabs(x_(1)) < 0.001) {
      x_(0) = 0.001;
      x_(1) = 0.001;
    }
		
		// Set the x_aug_ vector
    x_aug_.head(5) = x_;
    x_aug_(5) = 0;
    x_aug_(6) = 0;
		
    // done initializing, no need to predict or update
    previous_timestamp_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }
	

  // Calculate the timestep between measurements in seconds
  float dt = (meas_package.timestamp_ - previous_timestamp_);
  dt /= 1000000.0; // convert micros to s
  previous_timestamp_ = meas_package.timestamp_;
  
  //Predict
  Prediction(dt);
  cout << "Predict" << endl;
  
  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
     // LASER
     UpdateLidar(meas_package);
     cout << "Lidar" << endl;
  } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
     // RADAR
     UpdateRadar(meas_package);
     cout << "Radar" << endl;
  }

  return;
}

/*******************************************************************************
 * Function to perform state predictions based on CTRV dynamic model
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 ******************************************************************************/
void UKF::Prediction(double delta_t) {

  x_aug_.head(5) = x_;
  x_aug_(5) = 0;
  x_aug_(6) = 0;
	
  P_aug_.fill(0.0);
  P_aug_.topLeftCorner(n_x_,n_x_) = P_;
  P_aug_(5,5) = std_a_*std_a_;
  P_aug_(6,6) = std_yawdd_*std_yawdd_;

  P_.fill(0);
	
  MatrixXd L = P_aug_.llt().matrixL();
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
	
  Xsig_aug.col(0)  = x_aug_;
  x_.fill(0);
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug.col(i + 1) = x_aug_ + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug_ - sqrt(lambda_ + n_aug_) * L.col(i);
  }
	
  //predict sigma points
  for (int i = 0; i < (2 * n_aug_ + 1); i++) {
    double px = Xsig_aug(0,i);
    double py = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yaw_rate = Xsig_aug(4,i);
    double sigma_a = Xsig_aug(5,i);
    double sigma_yaw = Xsig_aug(6,i);
    
    if (fabs(yaw_rate) > 0.001) {
      Xsig_pred_(0,i) = px + (v / yaw_rate)*(sin(yaw + yaw_rate*delta_t) - sin(yaw)) + 0.5*delta_t*delta_t*cos(yaw)*sigma_a;
      Xsig_pred_(1,i) = py + (v / yaw_rate)*(-cos(yaw + yaw_rate*delta_t) + cos(yaw)) + 0.5*delta_t*delta_t*sin(yaw)*sigma_a;
      Xsig_pred_(2,i) = v + delta_t*sigma_a;
      Xsig_pred_(3,i) = yaw + yaw_rate*delta_t + 0.5*delta_t*delta_t*sigma_yaw;
      Xsig_pred_(4,i) = yaw_rate + delta_t*sigma_yaw;
    } else{
      Xsig_pred_(0,i) = px + v*cos(yaw)*delta_t + 0.5*delta_t*delta_t*cos(yaw)*sigma_a;
      Xsig_pred_(1,i) = py + v*sin(yaw)*delta_t + 0.5*delta_t*delta_t*sin(yaw)*sigma_a;
      Xsig_pred_(2,i) = v + delta_t*sigma_a;
      Xsig_pred_(3,i) = yaw + 0.5*delta_t*delta_t*sigma_yaw;
      Xsig_pred_(4,i) = yaw_rate + delta_t*sigma_yaw;
    } 
  }

  x_ = Xsig_pred_ * weights_; // Matrix Calc (5X15) * (15X1) = (5X1)

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}


/*******************************************************************************
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) 
{
  // Laser Updates UKF
  //set measurement dimension, px, py
  int n_z = 2, i = 0;
	
  VectorXd z = meas_package.raw_measurements_;
	
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);
	
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
	
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
	
  //transform sigma points into measurement space
  for (i = 0; i < 2 * n_aug_ + 1; i++) {

    //sigma point predictions in process space
    double px = Xsig_pred_(0,i);
    double py = Xsig_pred_(1,i);

    //sigma point predictions in measurement space
    Zsig(0,i) = px;                   
    Zsig(1,i) = py;                                 
  }


  z_pred = Zsig * weights_; 

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_laspx_*std_laspx_, 0,
          0, std_laspy_*std_laspy_;
  S = S + R;
	
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
	
  //calculate cross correlation matrix
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
	
  //NIS Lidar Update
  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;	

  return;
}



void UKF::UpdateRadar(MeasurementPackage meas_package)
{

  //RADAR update UKF
  double px, py, v, v1, v2, yaw;

  int n_z = 3, i = 0;
  VectorXd z = meas_package.raw_measurements_;

  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0.0);

  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);

  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  //sigma points on to measurement
  for (i = 0; i < 2 * n_aug_ + 1; i++) {
    px = Xsig_pred_(0,i);
    py = Xsig_pred_(1,i);
    v  = Xsig_pred_(2,i);
    yaw = Xsig_pred_(3,i);
    v1 = cos(yaw)*v;
    v2 = sin(yaw)*v;

    if (fabs(px) < 0.001) 
      px = 0.001;
    if (fabs(py) < 0.001) 
      py = 0.001;
  
    //rho, phi and rho_dot
    Zsig(0,i) = sqrt(px*px + py*py);
    Zsig(1,i) = atan2(py,px);  
    Zsig(2,i) = (px*v1 + py*v2 ) / sqrt(px*px + py*py);
  }
  
   z_pred = Zsig * weights_; //Matrix Calc (3X15) * (15X1) = (3X1)
    
  for (i = 0; i < 2*n_aug_+1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  } 

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  R << std_radr_*std_radr_, 0, 0, 0, std_radphi_*std_radphi_, 0,
        0, 0, std_radrd_*std_radrd_;
  S = S + R; 

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
        
  //calculate cross correlation matrix
  for (i = 0; i<2*n_aug_+1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  //NIS Update
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
  return;
}
