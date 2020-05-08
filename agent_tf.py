import numpy as np
import environment as env
import tensorflow as tf
from tensorflow import keras
import copy


def softmax(x):
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x)


class maze_agent:
    layer_sizes = []
    model = 0

    previous_action_values = 0
    previous_action = 0
    previous_state = 0

    environment = 0
    terminal = 0
    reward = 0

    def __init__(self, hidden_layer_neurons, num_episodes, discount, learning_rate):
        self.environment = env.maze_environment()
        self.model = keras.models.Sequential([
            keras.layers.Dense(hidden_layer_neurons, activation=tf.nn.relu, input_shape=(25,)),
            keras.layers.Dense(4)
        ])
        self.model.compile(loss=tf.losses.mse, optimizer=tf.optimizers.Adam())
        self.model.summary()
        self.learn_to_play(num_episodes, discount, learning_rate)

    def action_value_policy(self, state):
        action_values = self.model.predict(state.T).T
        return action_values

    def update_weights(self, actual):
        self.model.fit(self.previous_state.T, actual.T, verbose=0, epochs=1)

    def choose_action(self, action_values):
        av_distribution = softmax(action_values).squeeze()
        action = np.random.choice(np.arange(4), p=av_distribution)
        return action

    def agent_start(self):
        self.previous_state = self.environment.state.copy()
        self.previous_action_values = self.action_value_policy(self.environment.state)
        self.previous_action = self.choose_action(self.previous_action_values)
        self.terminal, self.reward = self.environment.make_move(self.previous_action)

    def agent_step(self, discount, learning_rate):

        action_values = self.action_value_policy(self.environment.state)
        max_q = np.max(action_values)

        actual = np.zeros((4, 1))
        if self.terminal == 0:
            actual[self.previous_action] = self.reward + discount * max_q
        else:
            actual[self.previous_action] = self.reward

        self.update_weights(actual)

        self.previous_state = self.environment.state.copy()
        self.previous_action = self.choose_action(action_values)
        self.previous_action_values = action_values

        self.terminal, self.reward = self.environment.make_move(self.previous_action)

    def learn_to_play(self, num_episodes, discount, learning_rate):
        for game in range(num_episodes):
            print(game)
            self.environment.reset()
            self.agent_start()
            while True:
                if self.terminal:
                    self.agent_step(discount, learning_rate)
                    break
                else:
                    self.agent_step(discount, learning_rate)
