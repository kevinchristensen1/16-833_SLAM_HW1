import sys
import numpy as np
import math

class MotionModel:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """

    def __init__(self):

        """
        TODO : Initialize Motion Model parameters here
        """
        # ---------------------------------------------------
        # Initialize alpha values 
        # (Coefficients for determining variance when sampling)
        self.a1 = 0.0001
        self.a2 = 0.0001
        self.a3 = 0.0001
        self.a4 = 0.0001
        # ---------------------------------------------------
        
    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]   
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """
        # ---------------------------------------------------
        # Obtain odometry difference
        yd = u_t1[1] - u_t0[1]
        xd = u_t1[0] - u_t0[0]
        rot1 = math.atan2(yd,xd) - u_t0[2]
        trans = math.sqrt(xd**2 + yd**2)
        rot2 = u_t1[2] - u_t0[2] - rot1
        # ---------------------------------------------------
        # Random sampling given with 0 mean and variance given by lines 5-7 on pg. 136 of [1]
        mu = 0
        sig1 = self.a1*rot1**2 + self.a2*trans**2
        sig2 = self.a3*trans**2 + self.a4*rot1**2 + self.a4*rot2**2
        sig3 = self.a1*rot2**2 + self.a2*trans**2
        # ---------------------------------------------------
        # Adding Random noise to odometry measurements
        s1 = np.random.normal(mu, sig1)
        s2 = np.random.normal(mu, sig2)
        s3 = np.random.normal(mu, sig3)
        # ---------------------------------------------------
        # Create rot1_prime, etc
        rot1_p  = rot1 - s1
        trans_p = trans - s2
        rot2_p  = rot2 - s3
        # ---------------------------------------------------
        # Update new position with sampled odometry
        xshape = 3
        x_t1 = np.empty([1,xshape])
        x_t1[0,0] = x_t0[0] + trans_p*math.cos(x_t0[2]+rot1_p)
        x_t1[0,1] = x_t0[1] + trans_p*math.sin(x_t0[2]+rot1_p)
        x_t1[0,2] = x_t0[2] + rot1_p + rot2_p
        # ---------------------------------------------------
        # Theta must be ~ [-pi,pi]
        if x_t1[0,2] < -math.pi:
            x_t1[0,2] = 2*math.pi + x_t1[0,2]
        elif x_t1[0,2] > math.pi:
            x_t1[0,2] = x_t1[0,2] - 2*math.pi 
        
        return x_t1

    def laser_position(self, odometry_laser, u_t1, x_t1):
        xshape = 3
        x_l1 = np.empty([1,xshape])
        x_l1[0,0] = x_t1[0,0] + odometry_laser[0] - u_t1[0]
        x_l1[0,1] = x_t1[0,1] + odometry_laser[1] - u_t1[1]
        x_l1[0,2] = x_t1[0,2]
        
        return x_l1

if __name__=="__main__":
    pass