import random
import math
import numpy as np
import gym
import sys

def weight_update(weight, target, current, grad, learning_rate):
    weight_new = weight + learning_rate*(target - current)*grad
    return weight_new

def Q(weight, observation):
    return np.dot(weight,observation)

def feature(observation):
    # return observation
    return np.hstack((observation,np.square(observation)))

#def policy()


env = gym.make('CartPole-v0')

naction = 2
feature_dim = 8
N = 50000 # the number of episode
T = 500 # length of episode
batch_size = 500
alpha = 0.99
gamma = 0.99
epsilon = 1

w = np.random.randn(naction,feature_dim)

for n in range(N/batch_size):
    print 'iteration: ',n

    batch_total_time = 0;
    batch_episode = [];
    for batch_idx in range(batch_size):
        observation = env.reset()
        observation = feature(observation)
        if np.random.rand < epsilon:
            action = np.randomrange(0,2)
        else:
            action = np.argmax(Q(w,observation))

        episode_observation = [observation]
        episode_action = [action]
        episode_reward = []

        for t in range(T):
            observation, reward, done, info = env.step(action)
            observation = feature(observation)
            episode_reward.append(reward)
            if done or t == T:
                break
            else:
                action = np.argmax(Q(w,observation))
                episode_observation.append(observation)
                episode_action.append(action)
        batch_episode.append((episode_action,episode_observation,episode_reward))
        batch_total_time += t
    print batch_total_time / batch_size

    w_old = w
    w_new = w
    for batch_idx in range(batch_size):
        episode_action, episode_observation, episode_reward = batch_episode[batch_idx]
        for i in range(len(episode_reward)-1):
            observation = episode_observation[i]
            next_observation = episode_observation[i+1]
            action = episode_action[i]
            reward = episode_reward[i+1]
            w_new[action,:] = weight_update(w_new[action,:], reward + gamma*np.amax(Q(w_old,next_observation)), Q(w_old,observation)[action], observation, alpha/batch_size/T)

    w = w_new
    print w
    # w = w_new/sum(sum(w_new))
    epsilon = epsilon * 0.9
    print epsilon