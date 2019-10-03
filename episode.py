import numpy as np
import env.context as ctx
import requests
from wf_gen_funcs import tree_data_wf, read_wf
from argparse import ArgumentParser
from typing import List
from draw_figures import write_schedule

parser = ArgumentParser()
parser.add_argument('--host', type=str, default='localhost')
parser.add_argument('--port', type=int, default=9900)
parser.add_argument('--task-par', type=int, default=None)
parser.add_argument('--agent-task', type=int, default=None)
parser.add_argument('--task-par-min', type=int, default=None)
parser.add_argument('--nodes', type=np.ndarray, default=None)
parser.add_argument('--state-size', type=int, default=None)
parser.add_argument('--batch-size', type=int, default=None)
parser.add_argument('--wfs-name', type=List, default=None)
parser.add_argument('--is-test', type=bool, default=False)
parser.add_argument('--num-episodes', type=int, default=1)
parser.add_argument('--is-parallel', type=bool, default=False)
parser.add_argument('--save', type=bool, default=False)

DEFAULT_CONFIG = {'task_par': 30, 'agent_task': 5, 'task_par_min': 20,
                  'nodes': np.array([4, 8, 8, 16]), 'state_size': 64, 'batch_size': 16, 'wfs_name': ['Montage_25']}


def parameter_setup(args, config):
    dict_args = vars(args)
    for key, value in dict_args.items():
        if value is not None:
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
        action = requests.post(f'{URL}act', json={'state': state, 'mask': mask, 'sched': False}).json()['action']
        act_t, act_n = wfl.actions[action]
        reward, wf_time = wfl.make_action(act_t, act_n)
        next_state = list(map(float, list(wfl.state)))
        done = wfl.completed
        sars_list.append((state, action, reward, next_state, done))
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
    return reward


def test_agent(args):
    global URL
    config = parameter_setup(args, DEFAULT_CONFIG)
    test_wfs, test_times, test_scores, test_size = wf_setup(config['wfs_name'])
    for i in range(test_size):
        best_saves = []
        for k in range(5):
            ttree, tdata, trun_times = test_wfs[i]
            wfl = ctx.Context(config['agent_task'], config['nodes'], trun_times, ttree, tdata)
            wfl.name = config['wfs_name'][i]
            done = wfl.completed
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
                    best_saves.append((wfl, wf_time))
                    break
        best_saves = sorted(best_saves, key=lambda x: x[1])
        write_schedule(i, best_saves[0][0])


def save():
    model = requests.post(f'{URL}save')


if __name__ == '__main__':
    args = parser.parse_args()
    URL = f'http://{args.host}:{args.port}/'
    if not args.is_test:
        rewards = [run_episode_not_parallel(ei, args) for ei in range(args.num_episodes)]
    else:
        test_agent(args)

    if args.save:
        save()
