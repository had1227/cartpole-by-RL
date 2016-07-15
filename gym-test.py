import gym
import numpy
import random
env = gym.make('CartPole-v0')

act_space=numpy.zeros((40,60,40,60,2))
epsilon = 0.7
a=numpy.zeros(8)        # for debugging
test_interval = 10000
total_eps=1000000

def quantized_state(observation):

    state = numpy.zeros(4,int)
    state[0] = int(20 + round(observation[0] * 40))
    state[1] = int(20 + round(observation[1] * 10))
    state[2] = int(20 + round(observation[2] * 40))
    state[3] = int(20 + round(observation[3] * 10))

    if (state[0] > 39):
        state[0] = 39
        a[0] += 1
    if (state[0] < 0):
        state[0] = 0
        a[1] += 1
    if (state[1] > 59):
        state[1] = 59
        a[2] += 1
    if (state[1] < 0):
        state[1] = 0
        a[3] += 1
    if (state[2] > 39):
        state[2] = 39
        a[4] += 1
    if (state[2] < 0):
        state[2] = 0
        a[5] += 1
    if (state[3] > 59):
        state[3] = 59
        a[6] += 1
    if (state[3] < 0):
        state[3] = 0
        a[7] += 1

    return state

#env.monitor.start('/tmp/cartpole_experiment1',video_callable=lambda i_episode: i_episode % test_interval == 0, force=True)

for i_episode in range(total_eps+1):
    observation = env.reset()

    for t in range(1000):
        #print(observation)
        #env.render()

        state1 = quantized_state(observation)

        #print(state1)

        if(i_episode % test_interval==0):     # only for test
            if (act_space[state1[0]][state1[1]][state1[2]][state1[3]][0] >
                    act_space[state1[0]][state1[1]][state1[2]][state1[3]][1]):
                action = 0
            else:
                action = 1
        else:
            ran_val=random.random()
            if(ran_val<epsilon):
                action = random.randrange(0,2)      # exploration
            else:
                if (act_space[state1[0]][state1[1]][state1[2]][state1[3]][0] >
                        act_space[state1[0]][state1[1]][state1[2]][state1[3]][1]):
                    action = 0
                else:
                    action = 1

        observation, reward, done, info = env.step(action)

        state2 = quantized_state(observation)

        sample = reward + max(act_space[state2[0]][state2[1]][state2[2]][state2[3]])

        #print("sample, action:",sample, action)
        if(t!=0):
            alpha=float(1)/float(t)
            q=act_space[state1[0]][state1[1]][state1[2]][state1[3]][action]
            act_space[state1[0]][state1[1]][state1[2]][state1[3]][action]=(1-alpha)*q+alpha*sample

        else:
            act_space[state1[0]][state1[1]][state1[2]][state1[3]][action]=sample

        #print("Q value",act_space[state1[0]][state1[1]][state1[2]][state1[3]][action])
        if done:
            if(i_episode % test_interval == 0 ):
                print "Test finished.",
            if(i_episode % 10000 == 0):
                print "Episode {} finished".format(i_episode),
                print "after {} timesteps".format(t+1)
            if (i_episode % test_interval == 0 and t > 200):
                print "Well Done!"
            break

print(a)

#env.monitor.close()