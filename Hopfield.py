'''
    Hopfield Neural Network
'''
import numpy as np
import matplotlib.pylab as plt


class HopfieldNN():
    def __init__(self, N_nodes, Q_patterns, test_imgs, T, Time_steps, N_tests, ini_olp, retrieval_pattern):
        self.N_nodes    = N_nodes
        self.Q_patterns = Q_patterns
        self.T          = T
        self.Time_steps = Time_steps
        self.N_tests    = N_tests
        self.ini_olp    = ini_olp
        self.retrieval_pattern = retrieval_pattern
        self.success    = 0 
        self.state      = np.ones(self.N_nodes)
        self.test_img   = test_imgs
        self.patterns   = np.ones((self.Q_patterns, self.N_nodes))
        self.ev_states  = np.ones((self.N_nodes, self.Time_steps))
        self.avg_olp    = np.zeros(self.Time_steps) 
        
    
    # run Ntest    
    def run_Ntest(self):
        for i in range(self.N_tests):
            np.random.seed(i)
            olp,success = self.run()
            if success==1:
                print("Test # {}, overlep : {} | success ".format(i, olp))
            else:
                print("Test # {}, overlep : {} | Fail  ".format(i, olp))
        
        for t in range(self.Time_steps):
            self.avg_olp[t] /= self.N_tests
            #print("t = {:3.0f}, overlap = {:8.5f} ".format(t, self.avg_olp[t]))
            
        return self.ev_states
        
    def plot_res(self):
         t   = np.linspace(0, self.Time_steps-1, self.Time_steps) 
         plt.plot(t, self.avg_olp,'ko-')
         plt.show()
    
    
    
    # run one test     
    def run(self):
        self.init_patterns()
        self.patterns[self.retrieval_pattern] = self.test_img
        self.init_states()
        self.redo_init_states()
        
        self.init_synaptic()
        
        for t in range(self.Time_steps):
            olp = self.overlaps(self.retrieval_pattern)
            self.avg_olp[t] += olp
            self.update_state()
            self.ev_states[:,t] = self.state
            
            #self.plot_state(self.state)
        
        this_success = 0
        if olp >= 0.95:
            this_success = 1
            self.success += 1
        return olp, this_success
        
        
        # initialize patterns
    def init_patterns(self):
        ran_array = np.random.random((self.Q_patterns, self.N_nodes))
        self.patterns   = np.ones((self.Q_patterns, self.N_nodes))
        self.patterns[np.where(ran_array > 0.5)] = -1
    
    # initialize state randomly
    def init_random_states(self):
        ran_array  = np.random.random(self.N_nodes)
        self.state = np.ones(self.N_nodes)
        self.state[np.where(ran_array > 0.5)] = -1

    # initialize states similar to modes with a similar factor p  
    def init_states(self):
        ran_array = np.random.random(self.N_nodes)
        sfactor   = 0.5 * (1. + self.ini_olp)
        mask      = np.ones(self.N_nodes)
        mask[np.where(ran_array > sfactor)] = -1 
        self.state = self.patterns[self.retrieval_pattern] * mask 
        
        
    # redo state initialization unit |overlap - ini_olp| < 0.0001
    def redo_init_states(self):
        while True:
            ol = self.overlaps(self.retrieval_pattern)
            if np.fabs(ol - self.ini_olp) >= 0.0001:
                self.init_states()
            if np.fabs(ol - self.ini_olp) < 0.0001:
                break

    # calcualte synaptic 
    def init_synaptic(self):
        self.synaptic = np.zeros((self.N_nodes, self.N_nodes))
        self.synaptic =  np.dot(np.transpose(self.patterns), self.patterns)
        self.synaptic /= self.N_nodes;
        np.fill_diagonal(self.synaptic, 0)

    # calculate the overlaps 
    def overlaps(self,pattern_id):
       olp = np.mean(self.patterns[pattern_id]*self.state) 
       return olp
   
    def Prob_function(self, hi, this_state):
        boltz_factor     = np.exp(1.0*this_state*hi/self.T)
        inv_boltz_factor = 1.0/boltz_factor;
        return boltz_factor/(boltz_factor + inv_boltz_factor)

    def update_state(self):
        hi = np.dot(self.synaptic, self.state)
        if self.T == 0.0:
            self.state = np.ones(self.N_nodes)
            self.state[np.where(hi < 0)] = -1
        else:
            prob_s1 = self.Prob_function(hi,  1)
            prob_s0 = self.Prob_function(hi, -1)
            prob_diff = prob_s1 - prob_s0
            self.state = np.ones(self.N_nodes)
            self.state[np.where(prob_diff < 0)] = -1
        

    def print_state(self):
        print(self.state)



if __name__ == "__main__":
    N_nodes    = 900
    Q_patterns = 20
    T_steps    = 12
    N_tests    = 1
    ini_olp    = 0.2
    T          = 0.1
    retrieval_pattern = 0
    
    
    test_img = np.ones(N_nodes)
    np.random.seed(0)
    a   = HopfieldNN(N_nodes, Q_patterns, test_img, T, T_steps, N_tests, ini_olp, retrieval_pattern)
    a.run_Ntest()
    a.plot_res() 
    
    print("load rate: {}".format(Q_patterns/N_nodes))
    
    


