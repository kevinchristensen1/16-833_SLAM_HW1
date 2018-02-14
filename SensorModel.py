import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm
import pdb
import tarfile
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
        # Unzips tarfile ~ 8 MB, then loads it into occ as a [y][x][theta] array
        tar = tarfile.open('occ.txt.npy.tar.gz')
        tar.extractall()
        self.occ = np.load('occ.txt.npy')
        # for i in range(399,400):
        #     for j in range(399,400):
        #         for t in range (0,360):
        #             print self.occ[i][j][t]
        self.occupancy_map = occupancy_map
        self.step_size = 5
        self.z_max = 8100.0
        
        self.sig_norm = 50.0
        self.lambda_short = 0.009
        self.z_hit = 0.65
        self.z_rand = 230.0
        
        self.z_max_mult = 0.0007
        self.z_short = 0.09
        
        self.xy_step = 3;
        
        self.theta_inc = round(math.pi/36,2)
        

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        # print "x_t1: ", x_t1
        z_t1_prior = self.trace_rays(x_t1)
        #print "z_t1_arr: ", z_t1_arr
        #print "z_t1_prior: ", z_t1_prior
        # normal_tot = 0
        # random_tot = 0
        # failure_tot = 0
        # short_tot = 0
        # if z_t1_prior < 0:
        #     return 0.0
        x_curr = int(round(x_t1[0,0]/10,0))
        y_curr = int(round(x_t1[0,1]/10,0))
        occ_value = self.occupancy_map[y_curr][x_curr]
        if occ_value < 0 or occ_value > 0.05:
            return 0.0
        debug = 0
        q = 0 #math.log(1.0-occ_value/5.0)
        for i in xrange(0,180,self.step_size):
            if abs(z_t1_arr[i] - z_t1_prior[i/self.step_size]) < 10.0  and debug:
                print "z_t1_arr:   ", z_t1_arr[i]
                print "z_t1_prior: ", z_t1_prior[i/self.step_size]
                
            #   Case 1: Correct range with local measurement noise = Gaussian Distribution
            # & Case 4: Random Measurements = Uniform Distribution
            # 
            if (z_t1_arr[i] < self.z_max):
            # Gaussian Distribution
                #print z_t1_arr[i] - z_t1_prior[i/self.step_size]
                norm_exp = -0.5 * pow((z_t1_arr[i] - z_t1_prior[i/self.step_size]),2) / pow(self.sig_norm,2)
                normal = (1.0/np.sqrt(2*np.pi*pow(self.sig_norm,2)))*np.exp(norm_exp)
                # normal = normal * self.z_hit
            # Uniform Distribution
                random = 1.0/self.z_max # * self.z_rand
            # Measurement is zmax
            else:
                random = 0.0
                normal = 0.0
            # Case 2: Unexpected objects = Exponential Distribution
            if (z_t1_arr[i] <= z_t1_prior[i/self.step_size]):
                short = (1/(1 - np.exp(-self.lambda_short*z_t1_prior[i/self.step_size])))*self.lambda_short*np.exp(-self.lambda_short*z_t1_arr[i])
                # short = short * self.z_short
            else:
                short = 0.0
            # Case 3: Failures = z_max Uniform Distribution
            if (z_t1_arr[i] >= self.z_max):
                failure = 1.0
            else:
                failure = 0.0
            p_hit = normal*self.z_hit
            p_rand = random*self.z_rand
            p_zshort = short*self.z_short
            p_fail = failure*self.z_max_mult
            p_tot = p_hit + p_rand + p_zshort + p_fail
            #if abs(z_t1_arr[i] - z_t1_prior[i/self.step_size]) <10.0 and debug:
            if p_tot > 0.1:
                print "p_hit", p_hit
                print "p_rand", p_rand
                print "p_zshort", p_zshort
                print "p_fail", p_fail
                print "p_tot" , p_tot
            
            # multiply probabilties together = add log of probabilities for numerical stability
            if p_tot > 0:
                q = q + math.log(p_tot)
            else:
                return 0.0
            #print normal*self.z_hit
        
        q = math.exp(q)
        
        # print "z_t1_arr = ", z_t1_arr
        # print "z_t1_prior = ", z_t1_prior
        #print q
        return q

    def trace_rays(self, x_t1):
        #dist_priors = list()
        theta_curr = x_t1[0,2]
        # if theta_curr < 0
        #     theta_curr
        x_curr = int(round(x_t1[0,0]/10,0))
        y_curr = int(round(x_t1[0,1]/10,0))
        x_og = x_curr
        y_og = y_curr
        # print x_og
        # print y_og
        # print theta_curr
        z_t1_prior = list()
        theta_start = int(round(math.degrees(theta_curr),0))-90
        theta_end = theta_start + 180
        for i in xrange(theta_start,theta_end,self.step_size):
            # theta_curr = theta_curr + self.theta_inc
            # if theta_curr > math.pi:
            #     theta_curr = theta_curr - 2*math.pi
            # elif theta_curr < -math.pi:
            #     theta_curr = theta_curr + 2*math.pi

            #y_step = self.xy_step*math.sin(theta_curr)
            #x_step = self.xy_step*math.cos(theta_curr)

            #y_wall, x_wall = self.trace_ray(x_step, y_step, x_curr, y_curr)
            
            
            th = i
            if th < 0:
                th += 360
            elif th >= 360:
                th -= 360
                
            z_curr = self.occ[y_og][x_og][th]
            if z_curr < 0:
                z_curr = 0.0
            z_t1_prior.append(z_curr)
            #print np.sqrt(pow((y_wall - y_curr)*10,2) + pow((x_wall-x_curr)*10,2))
            
            # VISUALIZE RAY TRACING
            # ------------------------------------------------------------------------------
            # Visually confirmed accuracy with plotting function below
            #
            # x_curr = z_curr*math.cos(math.radians(float(i)))/10 + x_og
            # y_curr = z_curr*math.sin(math.radians(float(i)))/10 + y_og
            # plt.plot([x_og,x_curr],[y_og,y_curr],'b')
            # plt.pause(0.000001)
            # ------------------------------------------------------------------------------
        #print z_curr
        
        # Distance returned in cm
        return z_t1_prior

    def trace_ray(self, x_step, y_step, x_curr, y_curr):

        x_og = x_curr
        y_og = y_curr
        xp = int(round(x_curr,0))
        yp = int(round(y_curr,0))
        while y_curr < 800 and y_curr > 0 and x_curr < 700 and x_curr > 300 and self.occupancy_map[yp][xp] < 0.05 :
            x_curr = x_curr + x_step
            y_curr = y_curr + y_step
            xp = int(round(x_curr,0))
            yp = int(round(y_curr,0))
            #print xp
            #print yp
            if xp < 300 or xp > 699 or self.occupancy_map[yp][xp] >= 0.1 or self.occupancy_map[yp][xp] < 0:
                #plt.plot([x_og,x_curr],[y_og,y_curr],'b')
                #plt.pause(0.000001)
                    
                return yp, xp
            if yp < 0 or yp >= 799 or self.occupancy_map[yp][xp] >= 0.1 or self.occupancy_map[yp][xp] < 0:
                #plt.plot([x_og,x_curr],[y_og,y_curr],'b')
                #plt.pause(0.000001)
                    
                return yp, xp
            if (y_curr - y_og)**2 + (x_curr - x_og)**2 > self.z_max**2:
            
                return yp, xp
                
        return yp, xp
 
if __name__=='__main__':
    pass