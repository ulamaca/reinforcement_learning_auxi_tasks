import tensorflow as tf
import numpy as np
import gym
import timeit

class Actor(object):
    '''
    This is a actor represented by a Neural Network
    What I need for it is:
        1. Neural Network architecture
        2. Learning: 
            2.1 from critic function
            2.2 gradient computation
    '''
    name = 'A2C_Actor_ver1'
    def __init__(self, sess, N_F, N_A, learning_rate = 1e-1, n_h1 = 20):        
        self.sess = sess # session for NN computation in TensorFlow
        self.s = tf.placeholder(tf.float32, [None, N_F])
        self.a = tf.placeholder(tf.int32, None ) # action is discrete in this case
        # ** is not necessary for an actor!
        # ** but if I like it to be used in the self.loss w/o defining further function, I should do this
        self.advantage = tf.placeholder(tf.float32, [None, 1])  
        
        # 1 construction of a '2-layer' NN representation for the agent
        # ** tf.layers.dense is a simple API to construct a NN FC-layer
        nn_l1 = tf.layers.dense(
            inputs = self.s,            
            units = n_h1, # n_h1: # of units in hidden layer 1
            activation = tf.nn.sigmoid, # MV (and I remember Sutton once mentioned)suggests that we need linear to guanrantee convergence            
            use_bias=True,
            kernel_initializer = tf.random_normal_initializer(0., .1),  # weights
            bias_initializer = tf.constant_initializer(0.1), 
            # name = 'nn_l1' # unnecessary
        ) # 1st-layer for nn_rep
        
        self.pi_nn_rep = tf.layers.dense(
            inputs= nn_l1,
            units = N_A,
            activation = tf.nn.softmax,
            kernel_initializer = tf.random_normal_initializer(0., .1),  # weights
            bias_initializer = tf.constant_initializer(0.1), 
            # name = 
        )        
        
        # ** why there is a 0 in [0, self.a] is not clear        
        # ** can not take gradient directly from a tf-object
        self.log_prob_a = tf.log( self.pi_nn_rep[0, self.a] ) # log_prob of the current action a, for grad.
        
        # self.adv. is a value whereas self.picked_a_prob is a tf-object to take gradient
        # the negative sign here is for minimization (instead of maximization)
        # the tf.reduce_mean enable it to learn using mini-batch
    
        self.loss = - tf.reduce_mean( self.advantage * self.log_prob_a )               
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            self.loss) 
        
    def learn(self, a, s, advtange):
        # The exact learninig law, which is dependent on advantage from value-function
        # pseudo-code:
        # grad = advantage * grad(), and update through SGD        
        s = s[np.newaxis, :] # sugge
        loss, _ = self.sess.run([self.loss, self.train_op], 
                             feed_dict ={self.s: s, self.a: a, self.advantage: advantage})
        return loss
                            
    def control(self, s):
        "for an agent to decision-make"
        # pseudo-code:
        # a = argmax( self.pi_nn_rep(s) )
        s = s[np.newaxis, :]
        pi_s = self.sess.run( self.pi_nn_rep, feed_dict = {self.s: s} )
        return np.random.choice(np.arange( pi_s.shape[1] ), p=pi_s.ravel() ) # ** generate a entry from a discrete pmf! 


class Critic(object):
    '''
    This is a Critic represented by a NN
    What I need for it are:
        1. NN structure <- using tf.contrib.layers
        2. Learning:
            Q-learning + Fixed-Q target
        3. 
    '''
    def __init__(self, sess, N_F, learning_rate = 1e-1, n_h1 = 20): 
        # * I think N_F and N_A are not necessary, I can modify it later
        
        self.sess = sess 
        # the role of sess here is not very clear! 
        # I followed this from MV         
        self.s = tf.placeholder(tf.float32, [None, N_F])
        self.advantage = tf.placeholder(tf.float32, None)
        
        # 1 construction of a '2-layer' NN representation for the agent
        # ** tf.layers.dense is a simple API to construct a NN FC-layer
        nn_l1 = tf.layers.dense(
            inputs = self.s,            
            units = n_h1,
            # activation = tf.nn.relu, # MV (and I remember Sutton once mentioned)suggests that we need linear to guanrantee convergence
            activation = tf.nn.sigmoid, # MV (and I remember Sutton once mentioned)suggests that we need linear to guanrantee convergence
            use_bias=True,
            kernel_initializer = tf.random_normal_initializer(0., .1),  # weights
            bias_initializer = tf.constant_initializer(0.1), 
            # name = 'nn_l1' # unnecessary
        ) # 1st-layer for nn_rep
        
        self.v_nn_rep = tf.layers.dense(
            inputs= nn_l1,
            units = 1,
            activation = None,
            kernel_initializer = tf.random_normal_initializer(0., .1),  # weights
            bias_initializer = tf.constant_initializer(0.1), 
            # name = 
        )
        
        self.value_estimate = tf.squeeze(self.v_nn_rep) # ** this is seemingly redudant but if not, problem for the next line appears!
        self.loss = tf.square( self.advantage - self.value_estimate ) 
        # fitted-Q target, with Discounted Reward R
        # I can't do directly tf.square( self.advantage - self.value_prediction(self.s) )  since it is impossible to input a tf-object into tensor
        
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            self.loss) 
            # I can add: global_step=tf.contrib.framework.get_global_step()
            # I can also use optimizer other than adam
        
     
    def value_prediction(self, s):
        s = s[np.newaxis, :]
        return self.sess.run( self.v_nn_rep, feed_dict = {self.s: s} ) 

    
    def learn(self, s, advantage):        
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.advantage: advantage}
        loss, _ = sess.run([self.loss, self.train_op], feed_dict) # I switch the order to dennybritz's code
        # to see if the above line may go run?
        return loss # actually this loss is not very useful except tracking performance of learning




# working environment setting up
env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

n_episodes= 400 # #episodes the agent plays to learn
T_max = 5000 # max. steps in an episode
gamma = 1 # discount rate for the future total reward
N_F = env.observation_space.shape[0] # ** this should be dimensionality of feature space but not |S|, be careful
N_A = env.action_space.n
n_h1 = 10 # # of hidden layers for both actor and critic
render = False # render or not
lr_A = 1e-2 # learning rate for actor 
lr_C = 1e-3 # learning rate for critic

sess = tf.Session()
bb_critic = Critic(sess, N_F = N_F, learning_rate = lr_C) 
bb_actor = Actor(sess, N_F = N_F, N_A = N_A, learning_rate = lr_A) 
sess.run(tf.global_variables_initializer()) # initialization
saver = tf.train.Saver() #Create a saver object which will save all the variables

r_total_collection = [] # a simple measure for learning
start = timeit.timeit()
for i in range(n_episodes): # from 0 to n_episodes
    s = env.reset()  # the initial state
    t = 0 # steps for the agent to walk    
    experience_buffer = []
    r_total = 0 # total reward (performance measurement)
    for t in  range(T_max):
        if render:
            env.render()
            
        a = bb_actor.control(s)
        s_, r, done, info = env.step(a)
        experience_buffer.append( (s,a,r) )
        s = s_
        r_total = r_total + r
        if done:
            print('episode #: ' + str(i)
                  + ', with total reward: '  + str(r_total)
                  ) # for seeing how training works
            break
        elif t == T_max:
            print('episode #: ' + str(i)
                  + ', with total reward: '  + str(r_total) + 'maximal scores'
                  ) # for seeing how training works
        
    R = bb_critic.value_prediction(s) # bootstrap for the final states for the last episode
    for s,a,r in reversed(experience_buffer):
        R = R + gamma * r
        v_s = bb_critic.value_prediction(s)
        advantage = R - v_s
        bb_critic.learn(s, advantage)
        bb_actor.learn(a, s, advantage)

    r_total_collection.append(r_total)
    
end = timeit.timeit()
    
print(r_total_collection)
print('elapsed time: ' + str(end-start) )

# saver.save(sess, 'a2c-cartpole-' + str(n_episodes), global_step=2)

# ** I like to ask the agent to play the game again to see how                 
   # 1 a problem for re-render the environment after training
   # 2 We need a peformance measure
