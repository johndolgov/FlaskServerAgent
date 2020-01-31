import numpy as np
import env.context as ctx
import requests
from wf_gen_funcs import tree_data_wf, read_wf
from argparse import ArgumentParser
from draw_figures import write_schedule
from actorlstm import LSTMDeque
from heft_deps.heft_settings import run_heft
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pathlib
import os
import time
import csv
import glob
import pandas as pd
from datetime import datetime

parser = ArgumentParser()

parser.add_argument('--alg', type=str, default='nns')

parser.add_argument('--host', type=str, default='localhost')
parser.add_argument('--port', type=int, default=9900)
parser.add_argument('--task-par', type=int, default=None)
parser.add_argument('--agent-task', type=int, default=None)
parser.add_argument('--task-par-min', type=int, default=None)
parser.add_argument('--nodes', type=np.ndarray, default=None)
parser.add_argument('--state-size', type=int, default=None)
parser.add_argument('--seq-size', type=int, default=5)
parser.add_argument('--batch-size', type=int, default=None)
parser.add_argument('--wfs-name', type=str, default=None)
parser.add_argument('--is-test', type=bool, default=False)
parser.add_argument('--num-episodes', type=int, default=1)
parser.add_argument('--is-lstm-agent', type=bool, default=False)
parser.add_argument('--run-name', type=str, default='NoName')
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--plot-csvs', type=bool, default=False)
parser.add_argument('--result-folder', type=str, default='')

DEFAULT_CONFIG = {'task_par': 30, 'agent_task': 5, 'task_par_min': 20,
                  'nodes': np.array([4, 8, 8, 16]), 'state_size': 64,
                  'batch_size': 64, 'wfs_name': ['Montage_25'], 'seq_size': 5}


def parameter_setup(args, config):
    dict_args = vars(args)
    for key, value in dict_args.items():
        if value is not None:
            if key is 'wfs_name':
                config[key] = [value]
            else:
                config[key] = value
    return config


def wf_setup(wfs_names):
    wfs_real = [read_wf(name) for name in wfs_names]
    test_wfs = []
    test_times = dict()
    test_scores = dict()
    test_size = len(wfs_real)
    for i in range(test_size):
        wf_components = tree_data_wf(wfs_real[i])
        # tasks_n = np.random.randint(task_par_min, task_par+1)
        # wf_components = tree_data_gen(tasks_n)
        test_wfs.append(wf_components)
        test_times[i] = list()
        test_scores[i] = list()
    return test_wfs, test_times, test_scores, test_size


def episode(ei, config, test_wfs, test_size):
    global URL
    ttree, tdata, trun_times = test_wfs[ei % test_size]
    wfl = ctx.Context(config['agent_task'], config['nodes'], trun_times, ttree, tdata)
    wfl.name = config['wfs_name'][ei % test_size]
    done = wfl.completed
    state = list(map(float, list(wfl.state)))
    sars_list = list()
    for act_time in range(100):
        if act_time > 100:
            raise Exception("attempt to provide action after wf is scheduled")
        mask = list(map(int, list(wfl.get_mask())))
        action = requests.post(f"{URL}act", json={'state': state, 'mask': mask, 'sched': False}).json()['action']
        act_t, act_n = wfl.actions[action]
        reward, wf_time = wfl.make_action(act_t, act_n)
        next_state = list(map(float, list(wfl.state)))
        done = wfl.completed
        sars_list.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            return reward, sars_list


def episode_lstm(ei, config, test_wfs, test_size):
    global URL
    ttree, tdata, trun_times = test_wfs[ei % test_size]
    wfl = ctx.Context(config['agent_task'], config['nodes'], trun_times, ttree, tdata)
    wfl.name = config['wfs_name'][ei % test_size]
    deq = LSTMDeque(seq_size=config['seq_size'], size=config['state_size'])
    done = wfl.completed
    state = list(map(float, list(wfl.state)))
    deq.push(state)
    state = deq.show()
    sars_list = list()
    for act_time in range(100):
        if act_time > 100:
            raise Exception("attempt to provide action after wf is scheduled")
        mask = list(map(int, list(wfl.get_mask())))
        action = requests.post(f'{URL}act', json={'state': state.tolist(), 'mask': mask, 'sched': False}).json()['action']
        act_t, act_n = wfl.actions[action]
        reward, wf_time = wfl.make_action(act_t, act_n)
        next_state = list(map(float, list(wfl.state)))
        deq.push(next_state)
        next_state = deq.show()
        done = wfl.completed
        sars_list.append((state.tolist(), action, reward, next_state.tolist(), done))
        state = next_state
        if done:
            return reward, sars_list


def remember(sars_list):
    global URL
    for sars in sars_list:
        _ = requests.post(f'{URL}remember', json={'SARSA': sars})


def replay(batch_size):
    global URL
    loss = requests.post(f'{URL}replay', json={'batch_size': batch_size}).json()['loss']
    return loss


def run_episode_not_parallel(ei, args):
    config = parameter_setup(args, DEFAULT_CONFIG)
    test_wfs, test_times, test_scores, test_size = wf_setup(config['wfs_name'])
    reward, sars_list = episode(ei, config, test_wfs, test_size)
    remember(sars_list)
    replay(config['batch_size'])
    if ei % 100 == 0:
        print("episode {} completed".format(ei))
    return reward


def run_episode_lstm(ei, args):
    config = parameter_setup(args, DEFAULT_CONFIG)
    test_wfs, test_times, test_scores, test_size = wf_setup(config['wfs_name'])
    reward, sars_list = episode_lstm(ei, config, test_wfs, test_size)
    remember(sars_list)
    replay(config['batch_size'])
    if ei % 100 == 0:
        print("episode lstm {} completed".format(ei))
    return reward


def test_agent(args):
    global URL
    config = parameter_setup(args, DEFAULT_CONFIG)
    test_wfs, test_times, test_scores, test_size = wf_setup(config['wfs_name'])
    for i in range(test_size):
        ttree, tdata, trun_times = test_wfs[i]
        wfl = ctx.Context(config['agent_task'], config['nodes'], trun_times, ttree, tdata)
        wfl.name = config['wfs_name'][i]
        state = list(map(float, list(wfl.state)))
        for time in range(wfl.n):
            mask = list(map(int, list(wfl.get_mask())))
            action = requests.post(f'{URL}test', json={'state': state, 'mask': mask, 'sched': False}).json()['action']
            act_t, act_n = wfl.actions[action]
            reward, wf_time = wfl.make_action(act_t, act_n)
            next_state = list(map(float, list(wfl.state)))
            done = wfl.completed
            state = next_state
            if done:
                test_scores[i].append(reward)
                test_times[i].append(wf_time)
                break
        write_schedule(args.run_name, i, wfl)


def test_agent_lstm(args):
    global URL
    config = parameter_setup(args, DEFAULT_CONFIG)
    test_wfs, test_times, test_scores, test_size = wf_setup(config['wfs_name'])
    for i in range(test_size):
        ttree, tdata, trun_times = test_wfs[i]
        wfl = ctx.Context(config['agent_task'], config['nodes'], trun_times, ttree, tdata)
        wfl.name = config['wfs_name'][i]
        deq = LSTMDeque(seq_size=config['seq_size'], size=config['state_size'])
        done = wfl.completed
        state = list(map(float, list(wfl.state)))
        deq.push(state)
        state = deq.show()
        for time in range(wfl.n):
            mask = list(map(int, list(wfl.get_mask())))
            action = requests.post(f'{URL}test', json={'state': state.tolist(), 'mask': mask, 'sched': False}).json()['action']
            act_t, act_n = wfl.actions[action]
            reward, wf_time = wfl.make_action(act_t, act_n)
            next_state = list(map(float, list(wfl.state)))
            deq.push(next_state)
            next_state = deq.show()
            done = wfl.completed
            state = next_state
            if done:
                test_scores[i].append(reward)
                test_times[i].append(wf_time)
        write_schedule(args.run_name, i, wfl)


def test_heft(args):
    global URL
    config = parameter_setup(args, DEFAULT_CONFIG)
    test_wfs, test_times, test_scores, test_size = wf_setup(config['wfs_name'])
    response = requests.post(f'{URL}heft', json={'wf_name': config['wfs_name'][0],
                                                 'nodes': config['nodes'].tolist()}).json()
    actions = response['actions']
    dif_actions = list(map(lambda x: (x[0] % config['agent_task'], x[1]), actions))
    for i in range(test_size):
        ttree, tdata, trun_times = test_wfs[i]
        wfl = ctx.Context(config['agent_task'], config['nodes'], trun_times, ttree, tdata)
        wfl.name = config['wfs_name'][i]
        reward = 0
        wf_time = 0
        for idx in range(len(dif_actions)):
            print(wfl.candidates)
            print(wfl.actions)
            act_t, act_n = dif_actions[idx]
            reward, wf_time = wfl.make_action(act_t, act_n)
            next_state = list(map(float, list(wfl.state)))
            done = wfl.completed
            state = next_state
            if done:
                test_scores[i].append(reward)
                test_times[i].append(wf_time)
                break
        write_schedule(args.run_name, 0, wfl)


def test_heft_simple(args):
    global URL
    config = parameter_setup(args, DEFAULT_CONFIG)
    test_wfs, test_times, test_scores, test_size = wf_setup(config['wfs_name'])
    ttree, tdata, trun_times = test_wfs[0]
    wfl = ctx.Context(config['agent_task'], config['nodes'], trun_times, ttree, tdata)
    worst_time = wfl.worst_time
    response = requests.post(f'{URL}heft', json={'wf_name': config['wfs_name'][0],
                                                 'nodes': config['nodes'].tolist(), 'worst_time': worst_time}).json()
    actions = response['actions']
    worst_time = wfl.worst_time
    reward = worst_time / response['makespan']
    return reward


def save():
    model = requests.post(f'{URL}save')


def do_heft(args):
    global URL
    config = parameter_setup(args, DEFAULT_CONFIG)
    test_wfs, test_times, test_scores, test_size = wf_setup(config['wfs_name'])
    ttree, tdata, trun_times = test_wfs[0]
    wfl = ctx.Context(config['agent_task'], config['nodes'], trun_times, ttree, tdata)
    worst_time = wfl.worst_time
    print(config['wfs_name'][0], config['nodes'].tolist())
    response = requests.post(f'{URL}heft', json={'wf_name': config['wfs_name'][0],
                                                 'nodes': config['nodes'].tolist(), 'worst_time': worst_time}).json()
    return response


def plot_csvs(args):
    cur_dir = os.getcwd()
    reward_path = pathlib.Path(cur_dir) / 'results'
    if args.result_folder != '':
        reward_path = pathlib.Path(cur_dir) / 'results' / args.result_folder
    os.chdir(reward_path)
    files = glob.glob('*.csv')
    plt.style.use("seaborn-muted")
    plt.figure(figsize=(10, 5))
    length = 0
    for file in reversed(files):
        rewards = pd.read_csv(file)['reward']

        if 'heft' in file:
            rewards = [rewards[0] for _ in range(length)]
            plt.plot(rewards, label="rewards")
            plt.ylabel('reward')
            plt.xlabel('episodes')
            plt.legend()
        else:
            length = len(rewards)
            plt.plot(rewards, '--', label="rewards")
            plt.ylabel('reward')
            plt.xlabel('episodes')
            plt.legend()
    plt_path = pathlib.Path(cur_dir) / 'results' / f'{args.run_name}_all_plt.png'
    plt.savefig(plt_path)


def merge_results(args):
    cur_dir = os.getcwd()
    reward_path = pathlib.Path(cur_dir) / 'results'
    os.chdir(reward_path)
    files = glob.glob('*rewards.csv')
    dfs_mean = []
    dfs = []

    for file in files:
        df = pd.read_csv(file)
        if 'mean' in file:
            dfs_mean.append(df)
        else:
            dfs.append(df)

    len_dfs_mean = len(dfs_mean)
    len_dfs = len(dfs)

    for idx, df in enumerate(dfs):
        l = len(df[0])
        indexes = [(idx+i)*len_dfs for i in range(l)]
        df['index'] = indexes
    df_reward_final = pd.concat(dfs)
    df_reward_final.sort_values(by=['index'])
    for idx, df in enumerate(dfs_mean):
        l = len(df[0])
        indexes = [(idx+i)*len_dfs_mean for i in range(l)]
        df['index'] = indexes
    df_mean_reward_final = pd.concat(dfs_mean)
    df_mean_reward_final.sort_values(by=['index'])
    df_reward_final.to_csv('reward_final.csv', index=False, encoding='utf-8')
    df_mean_reward_final.to_csv('reward_mean_final.csv', index=False, encoding='utf-8')






if __name__ == '__main__':
    start = time.time()
    args = parser.parse_args()
    URL = f'http://{args.host}:{args.port}/'
    cur_dir = os.getcwd()
    if args.plot_csvs:
        plot_csvs(args)
    if args.alg == '':
        print('Done')
    elif args.alg == 'nns':
        if not args.is_test:
            if not args.is_lstm_agent:
                rewards = [run_episode_not_parallel(ei, args) for ei in range(args.num_episodes)]
                means = np.convolve(rewards, np.ones((500,)))[499:-499] / 500
                means = means.tolist()
            else:
                rewards = [run_episode_lstm(ei, args) for ei in range(args.num_episodes)]
                means = np.convolve(rewards, np.ones((500,)))[499:-499] / 500
                means = means.tolist()

            a = time.time() - start
            plt.style.use("seaborn-muted")
            plt.figure(figsize=(10, 5))
            plt.plot(rewards, '--', label="rewards")
            plt.plot(means, '-', label="avg")
            plt.ylabel('reward')
            plt.xlabel('episodes')
            plt.legend()
            plt_path = pathlib.Path(cur_dir) / 'results' / f'{args.run_name}_{datetime.now()}_plt.png'
            plt.savefig(plt_path)

            reward_path = pathlib.Path(cur_dir) / 'results' / f'{args.run_name}_{datetime.now()}_rewards.csv'
            rewards = np.array(rewards)
            result = pd.DataFrame()
            result['reward'] = rewards
            result.to_csv(reward_path, sep=',', index=None, columns=['reward'])

            mean_reward_path = pathlib.Path(cur_dir) / 'results' / f'{args.run_name}_{datetime.now()}_mean_rewards.csv'
            means = np.array(means)
            result = pd.DataFrame()
            result['reward'] = means
            result.to_csv(mean_reward_path, sep=',', index=None, columns=['reward'])

        else:
            if not args.is_lstm_agent:
                test_agent(args)
            else:
                test_agent_lstm(args)
        if args.save:
            save()
    elif args.alg == 'heft':
        ideal_flops = 8.0
        reward = test_heft_simple(args)
        reward_path = pathlib.Path(cur_dir) / 'results' / f'{args.run_name}_heft_rewards.csv'
        rewards = np.array([reward])
        result = pd.DataFrame()
        result['reward'] = rewards
        result.to_csv(reward_path, sep=',', index=None, columns=['reward'])
