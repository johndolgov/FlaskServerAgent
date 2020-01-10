import random
import numpy as np
import json
import pathlib
import time
from collections import deque
from keras import initializers
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM, Embedding
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam


class DQNActor:
    def __init__(self, first, second, third, state_size, action_size, name='actor', host='localhost', port='8000'):
        self.FIRST_LAYER = first
        self.SECOND_LAYER = second
        self.THIRD_LAYER = third
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
        self.REPLAY_MEMORY = 3000  # number of previous transitions to remember
        self.LEARNING_RATE = 1e-4
        self.D = deque(maxlen=self.REPLAY_MEMORY)
        self.model = self.buildmodel(self.FIRST_LAYER, self.SECOND_LAYER, self.THIRD_LAYER)
        self.epsilon = self.INITIAL_EPSILON
        self.replay_counter = 0
        self.can_replay = False

    def buildmodel(self, first=1024, second=512, third=256):
        model = Sequential()
        model.add(Dense(first, input_dim=self.STATE, activation='relu'))
        model.add(Dense(second, activation='relu'))
        model.add(Dense(third, activation='relu'))
        model.add(Dense(self.ACTIONS, activation='relu'))
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model

    def remember(self, SARSA):
        self.D.append(SARSA)
        return True

    def act(self, state, mask, sched=False):
        if max(mask) == 0:
            raise Exception("no valid actions")
        # choose an action epsilon greedy
        if random.random() <= self.epsilon and not sched:
            q = np.random.random(self.ACTIONS)
            if mask is not 'none':
                q *= mask
            action_index = np.argmax(q)
        else:
            q = self.model.predict(state)
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
            inputs = np.zeros((len(minibatch), self.STATE))
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
        self.model.save_weights(name+"_fc.h5", overwrite=True)
        json_model = self.model.to_json()
        with open(name+"_fc.json", "w") as outfile:
            json.dump(self.model.to_json(), outfile)

        config = {'first-fc-layer': self.FIRST_LAYER,
                  'second-fc-layer': self.SECOND_LAYER,
                  'third-fc-layer': self.THIRD_LAYER,
                  'name': name+"_fc.h5"}

        with open('config.json', 'w') as fp:
            json.dump(config, fp)

        return json_model

    def load(self, name, path=None):
        print("Now we load weight")
        if path is not None:
            json_path = pathlib.Path(path) / 'config.json'
            model_path = pathlib.Path(path) / name
        else:
            json_path = pathlib.Path('config.json')
            model_path = pathlib.Path(name)
        with open(json_path, 'r') as fp:
            config = json.load(fp)

        self.model = self.buildmodel(first=config['first-fc-layer'],
                                     second=config['second-fc-layer'],
                                     third=config['third-fc-layer'])

        self.model.load_weights(model_path)
        adam = Adam(lr=self.LEARNING_RATE)
        self.model.compile(loss='mse', optimizer=adam)
        print("Weight load successfully")

    def get_model(self):
        return self.model


