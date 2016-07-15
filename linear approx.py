import gym
import numpy
import random
env = gym.make('CartPole-v0')

weight = [random.random() for x in range(5)]
epsilon = 0.7
test_interval = 10000
total_eps=100000

#env.monitor.start('/tmp/cartpole_experiment1',video_callable=lambda i_episode: i_episode % test_interval == 0, force=True)

def Q_val(observation, action): # q = W * F
    if(action==0):
        q = numpy.inner(weight[0:4], observation)
        q += weight[4]
    else:
        q = -numpy.inner(weight[0:4], observation)
        q += weight[4]

    return q

def weight_update(diff, prev_observation, action):      # action 0 -> f=+O, action 1 -> f=-O
    alpha = 0.001

    if(action==0):
        for x in range(4):
            weight[x] += alpha * diff * prev_observation[x]
        weight[4]+=alpha*diff
    else:
        for x in range(4):
            weight[x] += alpha * diff * -prev_observation[x]
        weight[4] += alpha * diff

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
        prev_observation=observation

        observation, reward, done, info = env.step(action)  # s->a->s'

        sample = reward + max(Q_val(observation,0),Q_val(observation,1))

        diff = sample - Q_prediction

        weight_update(diff, prev_observation, action)

        if done:
            if(i_episode % test_interval == 0 ):
                print "Test finished.",
            if(i_episode % 1000 == 0):
                print "Episode {} finished".format(i_episode),
                print "after {} timesteps".format(t+1)
                print sample,diff,Q_prediction
            if (i_episode % test_interval == 0 and t > 200):
                print "Well Done!"
            break

#env.monitor.close()