import numpy as np
import sys
import pdb
import math

from MapReader import MapReader
from MotionModel import MotionModel
from SensorModel import SensorModel
from Resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time

def visualize_map(occupancy_map):
    fig = plt.figure()
    plt.switch_backend('TkAgg')
    mng = plt.get_current_fig_manager();  # mng.resize(*mng.window.maxsize())
    plt.ion(); plt.imshow(occupancy_map, cmap='Greys'); plt.axis([0, 800, 0, 800]);


def visualize_timestep(X_bar, tstep, iter_num=-1):
    x_locs = X_bar[:,0]/10.0
    y_locs = X_bar[:,1]/10.0
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o', s=1)
    num_p = X_bar.shape[0]
    txt = plt.text(5,5, 'Number of particles: ' + str(num_p))
    if iter_num > -1:
        if iter_num < 10:

            name = "../images/frame000" + str(iter_num)
        elif iter_num < 100:
            name = "../images/frame00" + str(iter_num)
        elif iter_num < 1000:
            name = "../images/frame0" + str(iter_num)
        else:
            name = "../images/frame" + str(iter_num)
        plt.savefig(name)
    
    plt.pause(0.00001)
    scat.remove()
    txt.remove()

def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    # (randomly across the map) 
    y0_vals = np.random.uniform( 0, 7000, (num_particles, 1) )
    x0_vals = np.random.uniform( 3000, 7000, (num_particles, 1) )
    theta0_vals = np.random.uniform( -3.14, 3.14, (num_particles, 1) )

    # initialize weights for all particles
    w0_vals = np.ones( (num_particles,1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals,y0_vals,theta0_vals,w0_vals))
    
    return X_bar_init

def init_particles_freespace(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    # (in free space areas of the map)

    """
    TODO : Add your code here
    """ 
    # Initializes empty np arrays
    y0_vals = np.empty([num_particles,1])
    x0_vals = np.empty([num_particles,1])
    # Intializes all angles theta
    theta0_vals = np.random.uniform( -3.14, 3.14, (num_particles, 1) )

    for i in range(0,num_particles):
        # Creates x0 and y0 doubles
        y0_vals[i] = np.random.uniform( 0, 7000)
        x0_vals[i] = np.random.uniform( 3000, 7000)
        # If the particles are not in the hallways, generate new values until they are.
        while occupancy_map[int(y0_vals[i]/10)][int(x0_vals[i]/10)] < 0 or occupancy_map[int(y0_vals[i]/10)][int(x0_vals[i]/10)]> 0:
            
            y0_vals[i] = np.random.uniform( 0, 7000)
            x0_vals[i] = np.random.uniform( 3000, 7000)
            
    w0_vals = np.ones( (num_particles,1), dtype=np.float64)
    w0_vals = w0_vals / num_particles
    X_bar_init = np.hstack((x0_vals,y0_vals,theta0_vals,w0_vals))
    
    return X_bar_init

def init_debug(num_particles):
    # Initializes empty np arrays
    y0_vals = np.empty([num_particles,1])
    x0_vals = np.empty([num_particles,1])
    # Intializes all angles theta
    theta0_vals = np.random.uniform( 3.05, 3.14, (num_particles, 1) )

    for i in range(0,num_particles):
        # Creates x0 and y0 doubles
        y0_vals[i] = np.random.uniform( 3900, 4050)
        x0_vals[i] = np.random.uniform( 4000, 4300)
            
    w0_vals = np.ones( (num_particles,1), dtype=np.float64)
    w0_vals = w0_vals / num_particles
    X_bar_init = np.hstack((x0_vals,y0_vals,theta0_vals,w0_vals))
    return X_bar_init

def main():
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]   
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, qy, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """

    """
    Initialize Parameters
    """
    src_path_map = '../data/map/wean.dat'
    src_path_log = '../data/log/robotdata1_lost.log'

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map() 
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    num_particles = 3000
    og_num_particles = num_particles
    sumd = 0
    # ---------------------------------------------------
    # Create intial set of particles
    X_bar = init_particles_freespace(num_particles, occupancy_map)
    
    # Useful for debugging, places particles near correct starting area for log1
    #X_bar = init_debug(num_particles)
    # ---------------------------------------------------
    
    vis_flag = 1
    
    # ---------------------------------------------------
    # Weights are dummy weights for testing motion model
    w0_vals = np.ones( (1,num_particles), dtype=np.float64)
    w_t = w0_vals / num_particles
    #----------------------------------------------------
    
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if vis_flag:
        visualize_map(occupancy_map)

    iter_num = 0
    first_time_idx = True
    for time_idx, line in enumerate(logfile):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        meas_type = line[0] # L : laser scan measurement, O : odometry measurement
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ') # convert measurement values from string to double

        odometry_robot = meas_vals[0:3] # odometry reading [x, y, theta] in odometry frame
        # print "odometry_robot = ", odometry_robot
        time_stamp = meas_vals[-1]

        #if ((time_stamp <= 0.0) | (meas_type == "O")): # ignore pure odometry measurements for now (faster debugging) 
            #continue

        if (meas_type == "L"):
             odometry_laser = meas_vals[3:6] # [x, y, theta] coordinates of laser in odometry frame
             ranges = meas_vals[6:-1] # 180 range measurement values from single laser scan
        
        print "Processing time step " + str(time_idx) + " at time " + str(time_stamp) + "s"

        if (first_time_idx):
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros( (num_particles,4), dtype=np.float64)
        u_t1 = odometry_robot

        yd = u_t1[1]-u_t0[1]
        xd =  u_t1[0]-u_t0[0]
        d = math.sqrt(pow(xd,2) + pow(yd,2))
        if d < 1.0:
            visualize_timestep(X_bar, time_idx, time_idx)
            continue
        if d > 20: # lost robot
            print('\nROBOT IS LOST!!!\nResetting particles...\n')
            X_bar = init_particles_freespace(og_num_particles, occupancy_map)
            num_particles = og_num_particles
            u_t0 = u_t1
            visualize_timestep(X_bar, time_idx, time_idx)
            sumd = 0
        else:
            sumd = sumd + d

        for m in range(0, num_particles):
            """
            MOTION MODEL
            """
            
            x_t0 = X_bar[m, 0:3]
            x_t1 = motion_model.update(u_t0, u_t1, x_t0)
            #motion_last = math.sqrt((x_t1[0,1]-x_t0[0,1])**2 +  (x_t1[0,0]-x_t0[0,0])**2)  

            # ---------------------------------------------------
            # For testing Motion Model 
            # X_bar_new[m,:] = np.hstack((x_t1, w_t))
            # ---------------------------------------------------
            
            """
            SENSOR MODEL
            """
            if (meas_type == "L"):
                z_t = ranges
                x_l1 = motion_model.laser_position(odometry_laser, u_t1, x_t1)
                #print w_t.shape
                w_t = sensor_model.beam_range_finder_model(z_t, x_l1)
                # #print w_t.shape
                # if w_t > 0.0 and X_bar[m,3] > 0.0:
                #     w_new = math.log(X_bar[m,3]) + math.log(w_t)
                #     w_new = math.exp(w_new)
                # else:
                #      w_new = 0.0
                X_bar_new[m,:] = np.hstack((x_t1, [[w_t]]))
                #time.sleep(10)
            else:
                X_bar_new[m,:] = np.hstack((x_t1, [[X_bar[m,3]]]))
       

        # sorted_particles = X_bar[X_bar[:,3].argsort()]
        # print(sorted_particles[499,3])

        X_bar = X_bar_new

        u_t0 = u_t1
        X_bar[:,3] = X_bar[:,3]/sum(X_bar[:,3])

        """
        RESAMPLING
        """
       
        if sumd > 10.0:
            # X_bar = resampler.low_variance_sampler_rand(X_bar, occupancy_map)
            sumd = 0
            if X_bar[:,3].var() < 9.0e-9 and num_particles > 500:
                num_particles = num_particles - 300
                print 'Adapting particles\nCurrent particle size = ', num_particles
            elif X_bar[:,3].var() < 3.0e-8 and num_particles > 300:
                num_particles = num_particles - 100
                print 'Adapting particles\nCurrent particle size = ', num_particles
            
            # if num_particles < og_num_particles and X_bar[:,3].var() > 5.0e-7:
            #     num_particles = num_particles + 100
            #     print 'Adapting particles\nCurrent particle size = ', num_particles
            
            X_bar = resampler.low_variance_sampler(X_bar, num_particles)
            print X_bar[:,3].var()
            
        if vis_flag:
            visualize_timestep(X_bar, time_idx, time_idx)

if __name__=="__main__":
    main()
