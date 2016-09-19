from __future__ import division, print_function
import tensorflow as tf
import gym
import numpy as np
import math


class GenericModel(object):

    # hyperparameters
    discount_factor = 0.99
    learning_rate = 0.01
    stochastic_factor = 0.05
    regularization = 0.1

    # implementation
    max_batch = 10000

    def create_network(self):
        self.inputs, output_preactivation, weight_norm = self.create_innards()

        pre_output = tf.nn.softmax(output_preactivation)
        stochastic_var = tf.placeholder(tf.float32)
        scaled_output = pre_output * (1-stochastic_var)
        output = scaled_output + stochastic_var / self.ACTION_COUNT

        rewards = tf.placeholder(tf.float32, [None])
        actions_taken = tf.placeholder(tf.int32, [None])
        actions_one_hot = tf.one_hot(
                actions_taken, self.ACTION_COUNT, dtype=tf.float32)
        probs_taken = output * actions_one_hot
        probs = tf.reduce_sum(probs_taken, reduction_indices=[1])
        logprobs = tf.log(probs)

        regularization_var = tf.placeholder(tf.float32)
        regular_loss = regularization_var * weight_norm
        loss = -tf.reduce_sum(logprobs * rewards) + regular_loss
        rate_var = tf.placeholder(tf.float32)
        opt = tf.train.RMSPropOptimizer(rate_var).minimize(loss)

        self.output = output
        self.stochastic_var = stochastic_var
        self.rewards = rewards
        self.actions_taken = actions_taken
        self.regularization_var = regularization_var
        self.rate_var = rate_var
        self.loss = loss
        self.opt = opt


    def update(self, states, actions, expecteds):
        
        for i in range(math.ceil(len(states) / self.max_batch)):
            s = slice(i * self.max_batch,(i+1) * self.max_batch)
            self.session.run([self.loss, self.opt],
                    {self.inputs: states[s],
                     self.actions_taken: actions[s],
                     self.rewards: expecteds[s],
                     self.stochastic_var: self.stochastic_factor,
                     self.regularization_var: self.regularization,
                     self.rate_var: self.learning_rate})

    def __init__(self, **kwargs):
        # you can set arbitrary hyperparameters
        for k, v in kwargs.items():
            if getattr(self, k, None) is None:
                raise ValueError('undefined param %s' % k)
            setattr(self, k, v)
        self.session = tf.Session()
        with self.session:
            self.create_network()
        self.session.run(tf.initialize_all_variables())

    def act(self, state):
        result = self.session.run(self.output,
                {self.inputs: [state],
                 self.stochastic_var: self.stochastic_factor})
        ret = np.argmax(np.random.multinomial(1, result[0]))
        return ret

    def preproc_state(self, state):
        return state

    def collate_states(self, a, b, c):
        return c

    def is_done(self, state, reward, done, info):
        return done

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
   
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class PoleModel(GenericModel):

    GAME_NAME = 'CartPole-v0'
    ACTION_COUNT = 2

    hidden_size = 20

    def create_innards(self):
        inputs = tf.placeholder(tf.float32, [None, 4])

        W1 = weight_variable([4, self.hidden_size])
        bias1 = bias_variable([self.hidden_size])
        hidden_preactivation = tf.matmul(inputs, W1) + bias1
        hidden_activation = tf.nn.relu6(hidden_preactivation)

        W2 = weight_variable([self.hidden_size, 2])
        bias2 = bias_variable([2])
        output_preactivation = tf.matmul(hidden_activation, W2) + bias2

        return inputs, output_preactivation, 0


class BasePongModel(GenericModel):

    GAME_NAME = 'Pong-v0'
    ACTION_COUNT = 6

    discount_factor = 0.9
    learning_rate = .001
    reset_per_round = True

    def preproc_state(self, I):
        # Stolen from Karpathy's Pong work
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        return I.astype(np.float)

    def collate_states(self, a, b, c):
        prev = np.zeros_like(c) if b is None else b
        return c - prev

    def is_done(self, state, reward, done, info):
        return done or (reward != 0 and self.reset_per_round)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

class PongModel(BasePongModel):
    max_batch = 125
    def create_innards(self):
        FIELD_WIDTH = 80
        FIELD_AREA = FIELD_WIDTH * FIELD_WIDTH

        pre_inputs = tf.placeholder(tf.float32,
                [None, FIELD_WIDTH, FIELD_WIDTH])
        inputs = tf.reshape(pre_inputs, [-1, FIELD_WIDTH, FIELD_WIDTH, 1])

        W_conv1 = weight_variable([3, 3, 1, 16])
        b_conv1 = bias_variable([16]) 
        h_conv1 = tf.nn.relu6(conv2d(inputs, W_conv1) + b_conv1)

        W_conv2 = weight_variable([3, 3, 16, 16])
        b_conv2 = bias_variable([16])
        h_conv2 = tf.nn.relu6(conv2d(h_conv1, W_conv2) + b_conv2)

        W_fc1 = weight_variable([FIELD_AREA * 16, 200])
        b_fc1 = bias_variable([200])
        h_conv2_flat = tf.reshape(h_conv2, [-1, FIELD_AREA*16])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

        W_fc2 = tf.Variable(
                tf.zeros_initializer([200, self.ACTION_COUNT]))
        b_fc2 = bias_variable([self.ACTION_COUNT])
        y = tf.matmul(h_fc1, W_fc2) + b_fc2
        return pre_inputs, y, 0

class SimplePongModel(BasePongModel):
    hidden_size = 200
    def create_innards(self):
        pre_inputs = tf.placeholder(tf.float32, [None, 80, 80])
        inputs = tf.reshape(pre_inputs, [-1, 6400])

        W1 = weight_variable([6400, self.hidden_size])
        b1 = bias_variable([self.hidden_size])
        h1 = tf.nn.relu6(tf.matmul(inputs, W1) + b1)
        W1_norm = tf.reduce_sum(W1 * W1) 
        b1_norm = tf.reduce_sum(b1 * b1)

        W2 = weight_variable([self.hidden_size, self.ACTION_COUNT])
        b2 = bias_variable([self.ACTION_COUNT])
        output = tf.matmul(h1, W2) + b2
        W2_norm = tf.reduce_sum(W2 * W2)
        b2_norm = tf.reduce_sum(b2 * b2)

        norm = W1_norm + W2_norm + b1_norm + b2_norm
        return pre_inputs, output, 0



class GymTrainer(object):
    def __init__(self, model):
        self.env = gym.make(model.GAME_NAME)
        self.recent_states = (None, None, None)
        self.model = model
        self.affix_new_state(self.env.reset())
        self.states = []
        self.rewards = []
        self.actions = []

    def reset_game(self):
        self.recent_states = None, None, None
        self.affix_new_state(self.env.reset())

    def reset(self):
        self.actions[:] = []
        self.states[:] = []
        self.rewards[:] = []

    def update_model(self):
        reversed_expecteds = []
        running_reward = 0
        for r in reversed(self.rewards):
            running_reward += r
            reversed_expecteds.append(running_reward)
            running_reward *= self.model.discount_factor
        expecteds = np.array(reversed_expecteds[::-1])
        expecteds -= np.mean(expecteds)
        expecteds /= np.std(expecteds)

        self.model.update(self.states, self.actions, expecteds)

    def affix_new_state(self, newstate):
        procstate = self.model.preproc_state(newstate)
        a, b, c = self.recent_states
        self.recent_states = b, c, procstate

    def step(self, render=False):
        if render:
            self.env.render()

        collated_state = self.model.collate_states(*self.recent_states)

        action = self.model.act(collated_state)
        
        self.actions.append(action)
        self.states.append(collated_state)
        
        newstate, reward, done, info = self.env.step(action)
        mid_break = self.model.is_done(newstate, reward, done, info)
        self.affix_new_state(newstate)
        self.rewards.append(reward)
        if mid_break:
            self.update_model()
            self.reset()
        if done:
            self.reset_game()

    def run(self, steps, run_every=10000, frames=1):
        for i in range(steps):
            render = i % run_every < frames
            self.step(render=render)

def main():
    model = PongModel()
    trainer = GymTrainer(model)
    for _ in range(10000):
        trainer.step(render=True)
        for _ in range(5):
            trainer.step()

if __name__ == "__main__":
    main()
