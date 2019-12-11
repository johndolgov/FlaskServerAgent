import random
import numpy as np
import json
import time
from collections import deque
from keras import initializers
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM, Embedding
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam


class LSTMDeque(object):
    def __init__(self, seq_size, size):
        self.seq_size = seq_size
        self.size = size
        self.deque = np.zeros((seq_size, size))

    def push(self, x):
        slc = self.deque[1:]
        self.deque = np.zeros((self.seq_size, self.size))
        self.deque[:-1] = slc
        self.deque[-1] = x

    def show(self):
        return self.deque


class DQNLSTMActor:
    def __init__(self, state_size, action_size, seq_size, first_lstm_layer, second_lstm_layer, name='actor'):
        self.STATE = state_size
        self.ACTIONS = action_size  # number of  actions
        self.FINAL_GAMMA = 0.9  # decay rate of past observations
        self.INITIAL_GAMMA = 0.9  # decay rate of past observations
        self.GAMMA = self.INITIAL_GAMMA  # decay rate of past observations
        self.OBSERVATION = 50.  # timesteps to observe before training
        # self.EXPLORE = 80000.  # frames over which to anneal epsilon
        self.EXPLORE = 50000.  # frames over which to anneal epsilon
        self.GAMMA_EXPLORE = 50000.  # frames over which to anneal epsilon
        # как правило мы используем e-greedy policy и хотим, чтобы в начале действия были
        # более рандомны, чем в начале, поэтому мы снижаем e от INITIAL до FINAL за EXPLORE шагов
        self.FINAL_EPSILON = 0.01  # final value of epsilon
        self.INITIAL_EPSILON = 0.4  # starting value of epsilon
        self.REPLAY_MEMORY = 30  # number of previous transitions to remember
        self.LEARNING_RATE = 1e-4
        self.seq_size = seq_size
        self.lstm_deque = LSTMDeque(seq_size=seq_size, size=self.STATE)
        self.D = deque(maxlen=self.REPLAY_MEMORY)
        self.model = self.buildmodel(first_lstm_layer, second_lstm_layer)
        self.epsilon = self.INITIAL_EPSILON
        self.replay_counter = 0
        self.can_replay = False

    def buildmodel(self, first_lstm_layer = 32, second_lstm_layer = None):
        model = Sequential()
        if second_lstm_layer != 0:
            model.add(LSTM(first_lstm_layer, input_dim=self.STATE, activation='tanh', return_sequences=True))
            model.add(LSTM(second_lstm_layer, activation='tanh'))
        else:
            model.add(LSTM(first_lstm_layer, input_dim=self.STATE, activation='tanh'))
        model.add(Dense(self.ACTIONS, activation='relu'))
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model

    def remember(self, SARSA):
        self.D.append(SARSA)
        return True

    def act(self, states, mask, sched=False):
        if max(mask) == 0:
            raise Exception("no valid actions")
        # choose an action epsilon greedy
        if random.random() <= self.epsilon and not sched:
            q = np.random.random(self.ACTIONS)
            if mask is not 'none':
                q *= mask
            action_index = np.argmax(q)
        else:
            q = self.model.predict(states)
            if mask is not 'none':
                q *= mask
            if q.max() == 0:
                print("random action: max(Q)==0")
                q = np.random.random(self.ACTIONS)
                if mask is not 'none':
                    q *= mask
            action_index = np.argmax(q)
        return action_index

    def replay(self, batch_size):
        model_loss = 0
        if self.replay_counter > self.OBSERVATION:
            if not self.can_replay:
                self.can_replay = True
        else:
            if not self.can_replay:
                self.can_replay += 1

        # observation используется для того чтобы заполнить память рандомными действиями

        if self.can_replay and len(self.D) >= batch_size:
            #print('Replay is working')
            # выбираем batch_size действий из памяти
            minibatch = random.sample(self.D, batch_size)
            # мы оптимизируем функцию Q(s,a), соотвественно - вход сети s, выход a
            inputs = np.zeros((len(minibatch), self.seq_size, self.STATE))
            targets = np.zeros((inputs.shape[0], self.ACTIONS))
            # преобразуем каждую строку из памяти, чтобы использовать sarsa
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]  # This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                # if terminated, only equas reward
                inputs[i:i + 1] = state_t  # I saved down s_t
                targets[i] = self.model.predict(state_t)  # Hitting each buttom probability
                Q_sa = self.model.predict(state_t1)
                if terminal:
                    targets[i, action_t] = (1-self.GAMMA) * targets[i, action_t] + self.GAMMA*reward_t
                else:
                    # Q(s,a) это предположительная награда, если мы перейдём из состояния s в s_1
                    # c помощью дейтсивя a, поэтому мы меняем Q по алгоритму SARSA для действий из
                    # памяти
                    # targets[i, action_t] = reward_t + self.GAMMA * np.max(Q_sa)
                    targets[i, action_t] = (1-self.GAMMA) * targets[i, action_t] + self.GAMMA*reward_t + 0.1*np.max(Q_sa)

            # обучаем сеть
            model_loss = self.model.train_on_batch(inputs, targets)
            # print("Loss = {}".format(model_loss))
            if self.epsilon > self.FINAL_EPSILON:
                # print('Decrease epsilon')
                self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.EXPLORE
            # if self.GAMMA < self.FINAL_GAMMA:
            if self.GAMMA > self.FINAL_GAMMA:
                # print('Decrease epsilon')
                # self.GAMMA += (self.FINAL_GAMMA - self.INITIAL_GAMMA) / self.GAMMA_EXPLORE
                self.GAMMA -= (self.INITIAL_GAMMA - self.FINAL_GAMMA) / self.GAMMA_EXPLORE
        return {'loss': float(model_loss)}

    # функции для загрузки/сохранения весов
    def save(self, name):
        print("Now we save model")
        self.model.save_weights(name+"_lstm.h5", overwrite=True)
        json_model = self.model.to_json()
        with open(name+"_lstm.json", "w") as outfile:
            json.dump(self.model.to_json(), outfile)
        return json_model

    def load(self, name):
        print("Now we load weight")
        self.model.load_weights(name)
        adam = Adam(lr=self.LEARNING_RATE)
        self.model.compile(loss='mse', optimizer=adam)
        print("Weight load successfully")

    def get_model(self):
        return self.model


