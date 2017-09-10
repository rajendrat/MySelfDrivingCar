/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) 
{
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  is_initialized = true;
  num_particles = 100;
  std::default_random_engine generate;

  // normal distribution for x, y, and theta
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  particles.resize(num_particles);

  // Generating particles
  for (auto& p: particles) {
     p.x = dist_x(generate);
     p.y = dist_y(generate);
     p.theta = dist_theta(generate);
     p.weight = 1;
 
  } 

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  
  std::default_random_engine generate;

  // Generating Random gauusian noise 
  std::normal_distribution<double> N_x(0, std_pos[0]);
  std::normal_distribution<double> N_y(0, std_pos[1]);
  std::normal_distribution<double> N_theta(0, std_pos[2]);

  for (auto& p: particles) {
     if (fabs(yaw_rate) < 0.0001) {
        p.x += velocity * delta_t * cos(p.theta);
        p.y += velocity * delta_t * sin(p.theta);
     } else {
        p.x += velocity/yaw_rate * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
        //p.y += velocity/yaw_rate * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
        p.y += velocity/yaw_rate * (cos(p.theta) - cos( p.theta+yaw_rate*delta_t));
        p.theta += yaw_rate * delta_t;
     }

     // added sensor noise 
     p.x += N_x(generate);
     p.y += N_y(generate);
     p.theta += N_theta(generate);
 
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) 
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for (auto& obs: observations) {

     double minDist = std::numeric_limits<float>::max();
     for (const auto& pred: predicted) {
        double distance = dist(obs.x, obs.y, pred.x, pred.y);
        if (minDist > distance) {
           minDist = distance;
           obs.id = pred.id;
        }
     }
  }


}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html


  //collecting valid landmarks 
  for (auto& p: particles) {
     p.weight = 1.0;
     vector<LandmarkObs> predictions;
     for (const auto& lmark: map_landmarks.landmark_list) {
        double distance = dist(p.x, p.y, lmark.x_f, lmark.y_f);
        if (distance < sensor_range) {
           predictions.push_back (LandmarkObs{lmark.id_i, lmark.x_f, lmark.y_f});
        }
     }

     vector <LandmarkObs> observations_map;

     for (const auto& obs: observations) {
        LandmarkObs temp;
        temp.x = (obs.x * cos(p.theta)) - (obs.y * sin(p.theta)) + p.x;
        temp.y = (obs.x * sin(p.theta)) + (obs.y * cos(p.theta)) + p.y;
        observations_map.push_back(temp);
     }

     // Landmark index for each observation
     dataAssociation(predictions, observations_map);
     double x_term, y_term, w;

     // particle weight computation
     for (const auto& obs_m: observations_map) {
        Map::single_landmark_s landmark = map_landmarks.landmark_list.at(obs_m.id-1);
        x_term = pow(obs_m.x - landmark.x_f, 2) / (2 * pow(std_landmark[0], 2));
        y_term = pow(obs_m.y - landmark.y_f, 2) / (2 * pow(std_landmark[1], 2));
        w = exp(-(x_term + y_term)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
        p.weight = p.weight * w;
     }

     weights.push_back(p.weight);

  }
    
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // Generating distribution to its weight
  std::random_device rand;
  std::mt19937 gen(rand());
  std::discrete_distribution<> dist(weights.begin(), weights.end());

  vector<Particle> resampled_particles;
  resampled_particles.resize(num_particles);

  for (int i=0; i < num_particles; i++) {
     int idx = dist(gen);
     resampled_particles[i] = particles[idx];
  }

  particles = resampled_particles;
  weights.clear();

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
