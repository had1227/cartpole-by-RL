import gym
import numpy
import random
env = gym.make('CartPole-v0')

weight = [[-0.5 + random.random() for x in range(5)],[-0.5 + random.random() for x in range(5)]]
weight_= weight
c_step = 4

replay_memory=[]

gamma=0.9
epsilon = 0.7
test_interval = 10000
total_eps=100000
aa=0
b=0
wow=False
#env.monitor.start('/tmp/cartpole_experiment1',video_callable=lambda i_episode: i_episode % test_interval == 0, force=True)

def func(observation):

    f_list=[observation[x] for x in range(4)]
    f_list.append(0.001)

    return f_list

def Q_val(observation): # q = W * F

    f=func(observation)
    q=[numpy.inner(weight[x],f) for x in range(2)]

    return q

def Q_val_(observation): # q = W * F

    f = func(observation)
    q = [numpy.inner(weight[x],f) for x in range(2)]

    return q

def weight_update(diff, prev_observation, action):
    alpha = 0.1
    f = func(prev_observation)

    for x in range(5):
        weight[action][x] += alpha * diff * f[x]

for i_episode in range(total_eps+1):
    observation = env.reset()

    for t in range(1000):
        #print(observation)
        #env.render()

        if(i_episode % test_interval==0):     # only for test
            q_value=Q_val_(observation)
            if (q_value[0] > q_value[1]):
                action = 0
            else:
                action = 1
        else:
            ran_val=random.random()
            if(ran_val<epsilon):
                action = random.randrange(0,2)      # exploration
            else:
                q_value = Q_val_(observation)
                if (q_value[0] > q_value[1]):
                    action = 0
                else:
                    action = 1

        prev_observation = observation

        observation, reward, done, info = env.step(action)  # s->a->s'

        new_history = [prev_observation,action,reward,observation]
        replay_memory.append(new_history)

        randomly_selected_history=random.choice(replay_memory)

        selected_prev_observation = randomly_selected_history[0]
        selected_action = randomly_selected_history[1]
        selected_reward = randomly_selected_history[2]
        selected_observation=randomly_selected_history[3]

        next_q_val=Q_val_(selected_observation)     # observation of state s'
        sample = reward + gamma*max(next_q_val[0],next_q_val[1])    # observation of state s' is used

        Q_prediction = Q_val_(selected_prev_observation)  # observation of state s is used

        Q_prediction = Q_prediction[selected_action]

        diff = sample - Q_prediction

        weight_update(diff, selected_prev_observation, selected_action)

        if(t % c_step==0):
            weight_ = weight

        if done:
            if(i_episode % test_interval == 0 ):
                print "Test finished.",
                print "Episode {} finished".format(i_episode),
                print "after {} timesteps".format(t+1)

                print(weight)
                print(Q_prediction)
                if(wow):
                    aa+=t+1
                    b+=1
                #print weight
            if (i_episode % test_interval == 0 and t > 300):
                epsilon=0
                print "Well Done!"
                wow=True
            break

#env.monitor.close()
print(float(aa)/float(b))