import gym
import numpy
import random
env = gym.make('MountainCar-v0')

obsv_space =2
act_space = 3

use_replay_memory=True
use_replace_weight=False

weight = [[0 for i in xrange(obsv_space)] for j in xrange(act_space)]
weight_= weight
c_step = 1000

replay_memory=[]

gamma=0.9
epsilon = 0.6
test_interval = 1000
total_eps=100000
max_timestep=200
#env.monitor.start('/tmp/cartpole_experiment1',video_callable=lambda i_episode: i_episode % test_interval == 0, force=True)

class sudo_gym:
    def __init__(self):
        self.x = 0.25 + 0.5 * random.random()
    def step(self,action):
        if(action==0):
            self.x-=0.01
        else:
            self.x+=0.01

        observation=[self.x]
        if(self.x<=0):
            reward=-1
            done=True
        elif(self.x>=1):
            reward=1
            done=True
        else:
            reward=0
            done=False

        info=0

        return (observation, reward, done, info)

def func(observation):

    f_list=[observation[x] for x in xrange(obsv_space)]
    #f_list.append(0.001)

    return f_list

def Q_val(observation): # q = W * F

    f=func(observation)
    q=[numpy.inner(weight[x],f) for x in xrange(act_space)]

    return q

def Q_val_(observation): # q = W * F

    f = func(observation)
    q = [numpy.inner(weight[x], f) for x in xrange(act_space)]

    return q

def weight_update(sample, prediction, prev_observation, action):
    alpha = 0.1

    diff = sample - prediction
    f = func(prev_observation)

    for x in range(obsv_space):
        weight[action][x] += alpha * diff * f[x]

for i_episode in xrange(total_eps+1):
    observation = env.reset()

    for t in xrange(max_timestep):
        #print(observation)
        #env.render()

        ran_val=random.random()         # range of [0,1]
        if(ran_val<epsilon):            # epsilon greedy
            action = random.randrange(0,act_space)      # exploration
        else:
            q_value = Q_val_(observation)
            action = numpy.argmax(q_value)  # index of max Q-value

        prev_observation = observation

        observation, reward, done, info = env.step(action)  # s->a->s'
        if done:
            reward+=201
        if(use_replay_memory):
            new_history = [prev_observation,action,reward,observation]
            replay_memory.append(new_history)

        else:
            next_q_val=Q_val_(observation)
            sample = reward + gamma * max(next_q_val)

            Q_prediction = Q_val(prev_observation)
            Q_prediction = Q_prediction[action]

            weight_update(sample, Q_prediction, prev_observation, action)

        ##if(t % c_step==0):
        ##    weight_ = weight

        if done:
            break

    if(use_replay_memory):
        for i in xrange(1000):
            randomly_selected_history = random.choice(replay_memory)

            selected_prev_observation = randomly_selected_history[0]
            selected_action = randomly_selected_history[1]
            selected_reward = randomly_selected_history[2]
            selected_observation = randomly_selected_history[3]

            next_q_val = Q_val_(selected_observation)  # observation of state s'
            sample = selected_reward + gamma * max(next_q_val)  # observation of state s' is used

            Q_prediction = Q_val_(selected_prev_observation)  # observation of state s is used

            Q_prediction = Q_prediction[selected_action]

            weight_update(sample, Q_prediction, selected_prev_observation, selected_action)

        del(replay_memory)          # reset replay_memory
        replay_memory=[]

    if (i_episode % test_interval == 0):  # only for test

        timestep_sum=0

        for i in xrange(10):
            observation = env.reset()
            for t in xrange(max_timestep):
                q_value = Q_val_(observation)
                action = numpy.argmax(q_value)  # index of max Q-value
                observation, reward, done, info = env.step(action)
                if(done or t==max_timestep-1):
                    timestep_sum+=(t+1)
                    print(t)
                    break

        print "Test finished.",
        print "Episode {} finished".format(i_episode)
        print "Average timestep = {}".format(float(timestep_sum)/10)

#env.monitor.close()