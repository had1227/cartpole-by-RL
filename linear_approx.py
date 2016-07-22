import gym
import numpy
import random
env = gym.make('CartPole-v0')

weight = [-0.5 + random.random() for x in range(4)]
epsilon = 0.7
test_interval = 1000
total_eps=100000
aa=0
b=0
wow=False
#env.monitor.start('/tmp/cartpole_experiment1',video_callable=lambda i_episode: i_episode % test_interval == 0, force=True)

def func(observation, action):
    if(action==0):
        f_list=[observation[x] for x in range(4)]       # f=[observation] if action==0
    else:
        f_list=[-observation[x] for x in range(4)]       # f=[-observation] if action==1
    #f_list.append(0.001)

    return f_list

def Q_val(observation, action): # q = W * F

    f=func(observation, action)
    q=numpy.inner(weight,f)

    return q

def weight_update(diff, prev_observation, action):
    alpha = 0.1
    f = func(prev_observation, action)

    for x in range(4):
        weight[x] += alpha * diff * f[x]

for i_episode in range(total_eps+1):
    observation = env.reset()

    for t in range(1000):
        #print(observation)
        #env.render()

        if(i_episode % test_interval==0):     # only for test
            if (Q_val(observation,0) > Q_val(observation,1)):
                action = 0
            else:
                action = 1
        else:
            ran_val=random.random()
            if(ran_val<epsilon):
                action = random.randrange(0,2)      # exploration
            else:
                if (Q_val(observation,0) > Q_val(observation,1)):
                    action = 0
                else:
                    action = 1

        Q_prediction = Q_val(observation,action)      # Q(s,a)
        prev_observation=observation            # observation of state s

        observation, reward, done, info = env.step(action)  # s->a->s'

        sample = reward + max(Q_val(observation,0),Q_val(observation,1))    # observation of state s' is used

        diff = sample - Q_prediction

        weight_update(diff, prev_observation, action)

        if done:
            if(i_episode % test_interval == 0 ):
                print "Test finished.",
                print "Episode {} finished".format(i_episode),
                print "after {} timesteps".format(t+1)

                if(wow):
                    aa+=t+1
                    b+=1
                #print weight
            if (i_episode % test_interval == 0 and t > 400):
                epsilon=0.1
                print "Well Done!"
                wow=True
            break

#env.monitor.close()
print(float(aa)/float(b))