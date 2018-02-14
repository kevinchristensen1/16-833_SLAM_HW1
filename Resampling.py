import numpy as np
import pdb

class Resampling:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """

    def __init__(self):
        pass

    def multinomial_sampler(self, X_bar):

        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        pass

    def low_variance_sampler(self, X_bar,num_particles):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        # pdb.set_trace()
        #M = X_bar.shape[0]
        M = num_particles
        # print "\n\n\n\n\n\nX_bar = ", X_bar
        r = np.random.uniform(0, 1.0/M)
        # print "r = ", r
        c = X_bar[0,3]
        i = 0
        X_bar_resampled = np.zeros((M, 4))

        for m in range(0,M):
            u = r + m * 1.0/M

            while u > c:
                i = i + 1
                c = c + X_bar[i, 3]

            X_bar_resampled[m, :] = X_bar[i, :]

        return X_bar_resampled

    def low_variance_sampler_rand(self, X_bar,  occupancy_map):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        rand_num = 200
        # pdb.set_trace()
        M = X_bar.shape[0]
        M = M - rand_num
        # print "\n\n\n\n\n\nX_bar = ", X_bar
        r = np.random.uniform(0, 1.0/M)
        # print "r = ", r
        c = X_bar[0,3]
        i = 0
        X_bar_resampled = np.zeros(((M+rand_num), 4))

        for m in range(0,M):
            # print "m = ", m
            u = r + m * 1.0/(M+rand_num)

            while u > c:
                i = i + 1
                c = c + X_bar[i, 3]

            X_bar_resampled[m, :] = X_bar[i, :]
            X_bar_resampled[m,3] = 1.0/(M+rand_num)
        for i in range(M,M+rand_num):
            # print "i = ", i
            y0_val = np.random.uniform( 0, 7000)
            x0_val = np.random.uniform( 3000, 7000)
            # If the particles are not in the hallways, generate new values until they are.
            while occupancy_map[int(y0_val/10)][int(x0_val/10)] < 0 or occupancy_map[int(y0_val/10)][int(x0_val/10)]> 0:
                
                y0_val = np.random.uniform( 0, 7000)
                x0_val = np.random.uniform( 3000, 7000)
            
            w0_val = np.random.uniform(-3.14, 3.14)
            X_bar_resampled[i, :] = np.hstack((x0_val, y0_val, w0_val, 1.0/(M+rand_num)))
        return X_bar_resampled

def test():
    np.set_printoptions(precision=2)

    M = 300
    X_bar = np.random.uniform(0,1.0/M,(M,4))
    X_bar[:,3] = X_bar[:,3] / X_bar[:,3].sum()
    ind = np.expand_dims(np.arange(1,M+1),1)
    X_bar = np.concatenate((X_bar, ind), 1)

    r = Resampling()
    X_bar_resampled = r.low_variance_sampler(X_bar)
    
    comp = np.concatenate((X_bar[:,3:], X_bar_resampled[:,3:]), 1)

    # pdb.set_trace()

    # print(X_bar_resampled)

if __name__ == "__main__":
    test()