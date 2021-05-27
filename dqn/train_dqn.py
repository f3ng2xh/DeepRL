import gym
from dqn import DQNAgent
import matplotlib.pyplot as plt


def collect_relay_buffer(env, agent, num_episodes):
    s0 = env.reset()
    relay_buffer_list = []

    for episode in range(num_episodes):
        s0 = env.reset()
        while True:
            env.render()
            a0 = agent.act_e_greedy(s0)
            s1, r1, done, _ = env.step(a0)
            if done:
                r1 = -10
            relay_buffer_list.append([list(s0), a0, r1, list(s1)])
            if done:
                break
            s0 = s1

    return relay_buffer_list


def compute_avg_reward(env, agent, num_episodes=10):
    total_reward = 0.0
    for _ in range(num_episodes):
        s0 = env.reset()
        episode_reward = 0.0

        while True:
            env.render()
            a0 = agent.act(s0)
            s1, r1, done, _ = env.step(a0)
            episode_reward += r1
            if done:
                break

            s0 = s1

        total_reward += episode_reward
    return total_reward / num_episodes


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    agent = DQNAgent()
    relay_buffer_list = collect_relay_buffer(env, agent, 100)
    print("samples : %d" % len(relay_buffer_list))
    agent.fit(relay_buffer_list)

    for i in range(25):
        agent.fit(relay_buffer_list, batch_size=10)
        avg_reward = compute_avg_reward(env, agent, 100)
        print("avg_reward: %d" % avg_reward)
