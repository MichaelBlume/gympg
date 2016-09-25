from __future__ import division, print_function
import tensorflow as tf
import gym
import numpy as np
import math
import os.path

def reduce_stdev(t):
    m = tf.reduce_mean(t)
    return tf.sqrt(tf.reduce_mean(tf.square(t - m)))

def explained_variance(t, p):
    return 1 - reduce_stdev(t -p) / reduce_stdev(t)

class GenericModel(object):

    # hyperparameters
    discount_factor = 0.99
    learning_rate = 0.01
    stochastic_factor = 0.05
    regularization = 0.1
    bias_slowdown = 0.1
    value_rate = 0

    # implementation
    max_batch = 10000

    # output
    saves_dir = 'saves'

    # state

    # default nodes
    predicted_reward = tf.constant(0.0)

    def create_network(self):
        self.slow_bias_var = tf.placeholder(tf.float32)
        tf.scalar_summary('bias_slowdown', self.slow_bias_var)

        self.inputs, output_preactivation, weight_norm = \
                self.create_innards()

        pre_output = tf.nn.softmax(output_preactivation)
        stochastic_var = tf.placeholder(tf.float32)
        scaled_output = pre_output * (1-stochastic_var)
        output = scaled_output + stochastic_var / self.ACTION_COUNT
        tf.scalar_summary('stochastic', stochastic_var)

        self.episode_length = tf.placeholder(tf.float32)
        tf.scalar_summary('episode_length', self.episode_length)

        logprobs = tf.log(output)
        expected_logs = logprobs * output
        entropy = tf.reduce_sum(-expected_logs, reduction_indices=[1])
#        tf.scalar_summary('entropy', tf.reduce_mean(entropy))

        rewards = tf.placeholder(tf.float32, [None])
        actions_taken = tf.placeholder(tf.int32, [None])
        actions_one_hot = tf.one_hot(
                actions_taken, self.ACTION_COUNT, dtype=tf.float32)
        logs_taken = logprobs * actions_one_hot
        logprobs_taken = tf.reduce_sum(logs_taken, reduction_indices=[1])

        regularization_var = tf.placeholder(tf.float32)
        tf.scalar_summary('regularization', regularization_var)
        regular_loss = regularization_var * weight_norm
        tf.scalar_summary('weight_norm', weight_norm)
        surprise_rewards = rewards - tf.stop_gradient(self.predicted_reward)
        value_rate_var = tf.placeholder(tf.float32)
        value_term = value_rate_var * tf.reduce_sum(
                tf.square(rewards - self.predicted_reward))
        tf.scalar_summary('explained_variance',
                tf.maximum(explained_variance(rewards, self.predicted_reward), -0.5))
        reward_term = -tf.reduce_sum(logprobs_taken * surprise_rewards)
        loss = reward_term + value_term + regular_loss

        rate_var = tf.placeholder(tf.float32)
        tf.scalar_summary('rate', rate_var)
        self.global_step = tf.Variable(
                0, name='global_step', trainable=False)
        opt = tf.train.AdamOptimizer(rate_var).minimize(loss,
                global_step=self.global_step)


        self.output = output
        self.stochastic_var = stochastic_var
        self.rewards = rewards
        self.actions_taken = actions_taken
        self.regularization_var = regularization_var
        self.rate_var = rate_var
        self.value_rate_var = value_rate_var
        self.loss = loss
        self.opt = opt
        self.summaries = tf.merge_all_summaries()


    def update(self, states, actions, expecteds):
        
        epl = len(states)
        for i in range(math.ceil(epl / self.max_batch)):
            s = slice(i * self.max_batch,(i+1) * self.max_batch)
            _, summaries, step = self.session.run(
                    [self.opt, self.summaries, self.global_step],
                    {self.inputs: states[s],
                     self.actions_taken: actions[s],
                     self.rewards: expecteds[s],
                     self.slow_bias_var: self.bias_slowdown,
                     self.stochastic_var: self.stochastic_factor,
                     self.episode_length: epl,
                     self.regularization_var: self.regularization,
                     self.value_rate_var: self.value_rate,
                     self.rate_var: self.learning_rate})
            self.summary_writer.add_summary(summaries, step)
        self.saver.save(self.session, self.save_path)

    def __init__(self, save_name, **kwargs):
        self.save_name = save_name
        # you can set arbitrary hyperparameters
        for k, v in kwargs.items():
            if getattr(self, k, None) is None:
                raise ValueError('undefined param %s' % k)
            setattr(self, k, v)
        self.session = tf.Session()
        with self.session:
            self.create_network()
        self.summary_writer = tf.train.SummaryWriter(
                '%s/%s_summaries' % (self.saves_dir, self.save_name))
        self.saver = tf.train.Saver()
        self.save_path = '%s/%s.ckpt' % (self.saves_dir, self.save_name)
        if os.path.isfile(self.save_path):
            self.saver.restore(self.session, self.save_path)
        else:
            self.session.run(tf.initialize_all_variables())

    def act(self, state):
        result = self.session.run(self.output,
                {self.inputs: [state],
                 self.slow_bias_var: self.bias_slowdown,
                 self.stochastic_var: self.stochastic_factor})
        ret = np.argmax(np.random.multinomial(1, result[0]))
        return ret

    def preproc_state(self, state):
        return state

    def collate_states(self, a, b, c):
        return c

    def is_done(self, state, reward, done, info):
        return done

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial) * self.slow_bias_var

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
   
class PoleModel(GenericModel):

    GAME_NAME = 'CartPole-v0'
    ACTION_COUNT = 2

    hidden_size = 20

    def create_innards(self):
        inputs = tf.placeholder(tf.float32, [None, 4])

        W1 = weight_variable([4, self.hidden_size])
        bias1 = self.bias_variable([self.hidden_size])
        hidden_preactivation = tf.matmul(inputs, W1) + bias1
        hidden_activation = tf.nn.relu6(hidden_preactivation)

        W2 = weight_variable([self.hidden_size, 2])
        bias2 = self.bias_variable([2])
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

def norm(*args):
    return sum(tf.reduce_sum(a * a) for a in args)

class ConvPongModel(BasePongModel):
    value_rate = 1
    max_batch = 125
    def create_innards(self):
        FIELD_WIDTH = 80
        FIELD_AREA = FIELD_WIDTH * FIELD_WIDTH

        pre_inputs = tf.placeholder(tf.float32,
                [None, FIELD_WIDTH, FIELD_WIDTH])
        inputs = tf.reshape(pre_inputs, [-1, FIELD_WIDTH, FIELD_WIDTH, 1])

        W_conv1 = weight_variable([3, 3, 1, 16])
        b_conv1 = self.bias_variable([16])
        h_conv1 = tf.nn.relu6(conv2d(inputs, W_conv1) + b_conv1)

        W_fc1 = weight_variable([FIELD_AREA * 16, 200])
        b_fc1 = self.bias_variable([200])
        h_conv1_flat = tf.reshape(h_conv1, [-1, FIELD_AREA*16])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv1_flat, W_fc1) + b_fc1)

        W_fcr = weight_variable([200, 1])
        b_fcr = self.bias_variable([1])
        self.predicted_reward = tf.matmul(h_fc1, W_fcr) + b_fcr

        W_fc2 = tf.Variable(
                tf.zeros_initializer([200, self.ACTION_COUNT]))
        b_fc2 = self.bias_variable([self.ACTION_COUNT])
        y = tf.matmul(h_fc1, W_fc2) + b_fc2
        return pre_inputs, y, norm(
                W_conv1, b_conv1, W_fc1, b_fc1, W_fc2, b_fc2)

class SimplePongModel(BasePongModel):
    hidden_size = 200
    def create_innards(self):
        pre_inputs = tf.placeholder(tf.float32, [None, 80, 80])
        inputs = tf.reshape(pre_inputs, [-1, 6400])

        W1 = weight_variable([6400, self.hidden_size])
        b1 = self.bias_variable([self.hidden_size])
        h1 = tf.nn.relu6(tf.matmul(inputs, W1) + b1)

        W2 = weight_variable([self.hidden_size, self.ACTION_COUNT])
        b2 = self.bias_variable([self.ACTION_COUNT])
        output = tf.matmul(h1, W2) + b2

        return pre_inputs, output, norm(W1, b1, W2, b2)



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

    def run(self, steps, run_every=None, frames=None):
        for i in range(steps):
            render = run_every and frames and i % run_every < frames
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
