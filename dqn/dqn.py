import tensorflow as tf
import random
import math
import numpy as np


class QNet(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(QNet, self).__init__()
        self.input_layer = tf.keras.layers.Dense(
            100, input_shape=(input_dim,), activation='relu',
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal'))
        self.hidden_layer = tf.keras.layers.Dense(
            50, activation='relu',
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal'))
        self.output_layer = tf.keras.layers.Dense(
            output_dim, kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03))

    def call(self, inputs, training=None, mask=None):
        outputs = self.input_layer(inputs)
        outputs = self.hidden_layer(outputs)
        return self.output_layer(outputs)


def group_fn(buffer_list, group_size=4):
    np_array = np.array(buffer_list, dtype=object)
    m = len(buffer_list)
    n = len(buffer_list[0])
    for i in range(0, m, group_size):
        yield [np_array[i:i+group_size, j] for j in range(n)]


def parse_fn(buffer_list):
    for (s0, a0, r1, s1) in buffer_list:
        s0_ = np.array(list(s0))
        a0_ = np.stack((np.arange(0, len(r1), dtype=int), np.array(a0, dtype=int)), axis=1)
        s1_ = np.array(list(s1))
        yield s0_, a0_, r1, s1_


class DQNAgent(object):
    def __init__(self, gamma=0.8, status_dim=4, action_space_dim=2, epsi_low=0.02, epsi_high=0.9, epsi_decay = 200):
        # 创建一个网络来拟合 Q_* (s_t, a_t)
        self.q_net = QNet(input_dim=status_dim, output_dim=action_space_dim)
        self.gamma = gamma
        self.action_space_dim = action_space_dim
        self.optimizer = tf.keras.optimizers.SGD()

        self.epsi_low = epsi_low
        self.epsi_high = epsi_high
        self.epsi_decay = epsi_decay

        self.steps = 0

    def train_loop(self, s0, a0, r1, s1):
        q_net = self.q_net
        with tf.GradientTape() as tape:
            all_q0 = q_net(s0)
            q0 = tf.gather_nd(all_q0, a0)
            all_q1 = q_net(s1)
            max_q1 = tf.math.reduce_max(all_q1, axis=1)
            y_true = r1 + self.gamma * tf.stop_gradient(max_q1)
            loss = tf.reduce_mean(tf.math.squared_difference(y_true, q0))

        trainable_variables = q_net.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))
        return loss

    # relay_buffer $(s_t, a_t, r_t, s_{t+1})$
    # $Q_*(s_t,a_t) = E_{s_{t+1}}[r_t + \gamma * {max}_A Q_*(s_{t+1}, A)]$
    def fit(self, relay_buffer_array, batch_size=10):
        print("start training ... %d" % self.steps)
        for (s0, a0, r1, s1) in parse_fn(group_fn(relay_buffer_array, group_size=batch_size)):
            _ = self.train_loop(s0, a0, r1, s1)
            self.steps += 1

    # e-greedy
    def act_e_greedy(self, s):
        epsi = self.epsi_low + (self.epsi_high-self.epsi_low) * \
               math.exp(-1 * self.steps/self.epsi_decay)

        if random.random() < epsi:
            return random.randrange(0, self.action_space_dim)
        else:
            return self.act(s)

    def act(self, s):
        s = np.reshape(s, (1, -1))
        a = tf.argmax(self.q_net(s), 1)
        return tf.squeeze(a).numpy()


