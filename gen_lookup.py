import numpy as np
import sys
import pdb
import math
from multiprocessing import Process
from MapReader import MapReader
from MotionModel import MotionModel
from SensorModel import SensorModel
from Resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time

def beam_range_finder_model(z_t1_arr, x_t1, occupancy_map):
    """
    param[in] z_t1_arr : laser range readings [array of 180 values] at time t
    param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    param[out] prob_zt1 : likelihood of a range scan zt1 at time t
    """
    occ = occupancy_map[int(round(x_t1[0,1]/10,0))][int(round(x_t1[0,0]/10,0))]
    if occ < 0 or occ > 0.5:
        return 0.0
    
    # print "x_t1: ", x_t1
    z_t1_prior = self.trace_rays(x_t1)
    #print "z_t1_arr: ", z_t1_arr
    #print "z_t1_prior: ", z_t1_prior
    normal_tot = 0
    random_tot = 0
    failure_tot = 0
    short_tot = 0
    
    q = 0
    for i in xrange(0,180,self.step_size):
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
        # print "p_hit", p_hit
        # print "p_rand", p_rand
        # print "p_zshort", p_zshort
        # print "p_fail", p_fail
        # print "p_tot" , p_tot
        
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

def trace_rays(x_t1, occupancy_map):
    #dist_priors = list()
    theta_curr = math.radians(float(x_t1[2]))

    x_curr = float(x_t1[0])
    y_curr = float(x_t1[1])
    z_t1_prior = list()
    

    y_step = float(math.sin(theta_curr))
    x_step = float(math.cos(theta_curr))

    y_wall, x_wall = trace_ray(x_step, y_step, x_curr, y_curr, occupancy_map)
    z_curr = np.sqrt(pow((y_wall - y_curr)*10,2) + pow((x_wall-x_curr)*10,2))
    
        #print np.sqrt(pow((y_wall - y_curr)*10,2) + pow((x_wall-x_curr)*10,2))
    return z_curr

def trace_ray(x_step, y_step, x_curr, y_curr, occupancy_map):

    x_og = x_curr
    y_og = y_curr
    xp = int(round(x_curr,0))
    yp = int(round(y_curr,0))
    while y_curr < 800 and y_curr > 0 and x_curr < 700 and x_curr > 300 and occupancy_map[yp][xp] < 0.05 :
        x_curr = x_curr + x_step
        y_curr = y_curr + y_step
        xp = int(round(x_curr,0))
        yp = int(round(y_curr,0))
        #print xp
        #print yp
        if xp < 300 or xp > 699 or occupancy_map[yp][xp] >= 0.05 or occupancy_map[yp][xp] < 0:
            #plt.plot([x_og,x_curr],[y_og,y_curr],'b')
            #plt.pause(0.000001)
                
            return yp, xp
        if yp < 0 or yp >= 799 or occupancy_map[yp][xp] >= 0.05 or occupancy_map[yp][xp] < 0:
            #plt.plot([x_og,x_curr],[y_og,y_curr],'b')
            #plt.pause(0.000001)
                
            return yp, xp
        if (y_curr - y_og)**2 + (x_curr - x_og)**2 > 8100**2:
        
            return yp, xp
            
    return yp, xp
 
def genLookup():
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]   
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """

    """
    Initialize Parameters
    """
    src_path_map = '../data/map/wean.dat'
    src_path_log = '../data/log/robotdata1.log'

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map() 
    logfile = open(src_path_log, 'r')
    count2 = 0.0
    count = 0.0
    total = 800.0*800.0*360.0
    occ = np.empty((800,800,360), dtype=np.float64)
    ti = time.time()
    for y in range(0,800):
        for x in range(0,800):
            if occupancy_map[y][x] >= 0 and occupancy_map[y][x] < 1.0:
                for theta in range(0,360):
                    count += 1.0
                    occ[y][x][theta] = trace_rays(np.array((x,y,theta)), occupancy_map)
            else:
                for theta in range(0,360):
                    occ[y][x][theta] = -1
                    count2 += 1.0
                    
        print 'total:' , (count+count2)/total
        print 'Non -1 values:' , count/total
    elapsed = time.time() - ti
    print elapsed
    np.save('occ.txt',occ)
if __name__=='__main__':
    genLookup()
