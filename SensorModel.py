import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm
import pdb

from MapReader import MapReader

class SensorModel:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """

    def __init__(self, occupancy_map):

        """
        TODO : Initialize Sensor Model parameters here
        """
        self.occupancy_map = occupancy_map
        
        self.step_size = 5
        
        self.sig_norm = 150
        self.lambda_short = 0.01
        self.z_hit = 0.7
        self.z_rand = 0.098
        self.z_max = 800
        self.z_max_mult = 0.002
        self.z_short = 0.2
        
        self.theta_inc = round(3.14/36,2)
        self.slope_table = [(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),
            (1,9),(1,9),(1,9),(1,9),(1,8),(1,8),(1,8),(1,8),(1,6),(1,6),
            (1,6),(1,6),(2,9),(2,9),(2,9),(2,9),(2,9),(1,4),(1,4),(1,4),
            (1,4),(1,3),(1,3),(1,3),(1,3),(3,8),(3,8),(3,8),(3,8),(3,8),
            (2,5),(2,5),(2,5),(2,5),(4,9),(4,9),(4,9),(4,9),(4,9),(1,2),
            (1,2),(1,2),(1,2),(4,7),(4,7),(4,7),(4,7),(5,8),(5,8),(5,8),
            (5,8),(5,8),(5,8),(5,7),(5,7),(5,7),(7,9),(7,9),(7,9),(7,9),
            (7,9),(5,6),(5,6),(5,6),(5,6),(8,9),(8,9),(8,9),(1,1),(1,1),
            (1,1),(1,1),(1,1),(1,1),(10,9),(10,9),(10,9),(10,9),(6,5),(6,5),
            (6,5),(6,5),(6,5),(9,7),(9,7),(9,7),(9,7),(10,7),(10,7),(10,7),
            (10,7),(8,5),(8,5),(8,5),(8,5),(7,4),(7,4),(7,4),(7,4),(7,4),
            (2,1),(2,1),(2,1),(2,1),(9,4),(9,4),(9,4),(9,4),(9,4),(7,3),
            (7,3),(7,3),(7,3),(8,3),(8,3),(8,3),(8,3),(3,1),(3,1),(3,1),
            (3,1),(7,2),(7,2),(7,2),(7,2),(7,2),(9,2),(9,2),(9,2),(9,2),
            (6,1),(6,1),(6,1),(6,1),(6,1),(8,1),(8,1),(8,1),(8,1),(9,1),
            (9,1),(9,1),(9,1),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0),
            (1,0),(1,0),(1,0),(1,0),(9,-1),(9,-1),(9,-1),(9,-1),(8,-1),
            (8,-1),(8,-1),(8,-1),(6,-1),(6,-1),(6,-1),(6,-1),(6,-1),(9,-2),
            (9,-2),(9,-2),(9,-2),(7,-2),(7,-2),(7,-2),(7,-2),(7,-2),(3,-1),
            (3,-1),(3,-1),(3,-1),(8,-3),(8,-3),(8,-3),(8,-3),(7,-3),(7,-3),
            (7,-3),(7,-3),(9,-4),(9,-4),(9,-4),(9,-4),(9,-4),(2,-1),(2,-1),
            (2,-1),(2,-1),(7,-4),(7,-4),(7,-4),(7,-4),(7,-4),(8,-5),(8,-5),
            (8,-5),(8,-5),(10,-7),(10,-7),(10,-7),(10,-7),(9,-7),(9,-7),(9,-7),
            (9,-7),(6,-5),(6,-5),(6,-5),(6,-5),(6,-5),(10,-9),(10,-9),(10,-9),
            (10,-9),(1,-1),(1,-1),(1,-1),(1,-1),(1,-1),(1,-1),(8,-9),(8,-9),
            (8,-9),(5,-6),(5,-6),(5,-6),(5,-6),(5,-6),(7,-9),(7,-9),(7,-9),
            (7,-9),(7,-9),(5,-7),(5,-7),(5,-7),(5,-8),(5,-8),(5,-8),(5,-8),
            (5,-8),(5,-8),(4,-7),(4,-7),(4,-7),(4,-7),(1,-2),(1,-2),(1,-2),
            (1,-2),(4,-9),(4,-9),(4,-9),(4,-9),(4,-9),(2,-5),(2,-5),(2,-5),
            (2,-5),(3,-8),(3,-8),(3,-8),(3,-8),(3,-8),(1,-3),(1,-3),(1,-3),
            (1,-3),(1,-4),(1,-4),(1,-4),(1,-4),(2,-9),(2,-9),(2,-9),(2,-9),
            (2,-9),(1,-6),(1,-6),(1,-6),(1,-6),(1,-8),(1,-8),(1,-8),(1,-8),
            (1,-9),(1,-9),(1,-9),(1,-9),(0,-1),(0,-1),(0,-1),(0,-1),(0,-1),
            (0,-1),(0,-1),(0,-1),(0,-1)]

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        # print "x_t1: ", x_t1
        z_t1_prior = self.trace_rays(x_t1)
        # print "z_t1_arr: ", z_t1_arr
        # print "z_t1_prior: ", z_t1_prior
        normal_tot = 0
        random_tot = 0
        failure_tot = 0
        short_tot = 0
        for i in xrange(0,180,self.step_size):
            if (z_t1_arr[i] < self.z_max):
            # normal dist
                # norm_exp_num = 
                norm_exp = -0.5 * pow((z_t1_arr[i] - z_t1_prior[i/5]),2) / pow(self.sig_norm,2)
                normal = (1/np.sqrt(2*np.pi*pow(self.sig_norm,2)))*np.exp(norm_exp)
                # normal = normal * self.z_hit
            #random dist
                random = 1/self.z_max # * self.z_rand
            else:
                random = 0
                normal = 0
            # short dist
            if (z_t1_arr[i] <= z_t1_prior[i/5]):
                short = (1/(1 - np.exp(-self.lambda_short*z_t1_prior[i/5])))*self.lambda_short*np.exp(-self.lambda_short*z_t1_arr[i])
                # short = short * self.z_short
            else:
                short = 0
            # failure dist
            if (z_t1_arr[i] >= self.z_max):
                failure = 1
            else:
                failure = 0
            normal_tot += normal
            short_tot += short
            random_tot += random
            failure_tot += failure

        q = normal_tot*self.z_hit + random_tot*self.z_rand + short_tot*self.z_short + failure_tot*self.z_max_mult
        # print "z_t1_arr = ", z_t1_arr
        # print "z_t1_prior = ", z_t1_prior
        # print "q = ", q
        return q

    def trace_rays(self, x_t1):
        dist_priors = list()
        theta_curr = round(x_t1[0,2] - 1.57,2)
        while (theta_curr > 3.14):
            theta_curr -= 6.28
        while (theta_curr < -3.14):
            theta_curr += 6.28
        x_curr = int(x_t1[0,0]/10)
        y_curr = int(x_t1[0,1]/10)
        z_t1_prior = list()
        for i in xrange(0,180,5):
            theta_curr = theta_curr + self.theta_inc
            if theta_curr > 3.14:
                theta_curr = theta_curr - 6.28
            elif theta_curr < -3.14:
                theta_curr = theta_curr + 6.28

            if theta_curr >= 0 and theta_curr <= 3.14:
                x_step, y_step = self.slope_table[int(theta_curr*100)]
                
            else:# theta_curr >= -3.14 and theta_curr < 0:
                # print "theta_curr: ", theta_curr
                x_step, y_step = self.slope_table[int((theta_curr+3.14)*100)]
                x_step *= -1
                y_step *= -1

            y_wall, x_wall = self.trace_ray(x_step, y_step, x_curr, y_curr)

            z_t1_prior.append(np.sqrt(pow((y_wall - y_curr)*10,2) + pow((x_wall-x_curr)*10,2)))
        return z_t1_prior

    def trace_ray(self, x_step, y_step, x_curr, y_curr):
        sign_x = sign_y = 0
        if x_step != 0:
            sign_x = x_step/abs(x_step)
        if y_step != 0:
            sign_y = y_step/abs(y_step)
        x_step = x_step * sign_x
        y_step = y_step * sign_y
        while y_curr < 800 and x_curr < 800 and self.occupancy_map[y_curr][x_curr] < 0.05:
            for i in range(x_step):
                x_curr = x_curr + sign_x
                if x_curr < 0 or x_curr >= 800 or self.occupancy_map[y_curr][x_curr] > 0.05:
                    return y_curr, x_curr
            for i in range(y_step):
                y_curr = y_curr + sign_y
                if y_curr < 0 or y_curr >= 800 or self.occupancy_map[y_curr][x_curr] > 0.05:
                    return y_curr, x_curr
        return y_curr, x_curr
 
if __name__=='__main__':
    pass