import flask
from flask import request, send_from_directory
from actor import DQNActor
from flask import jsonify
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
import os

parser = ArgumentParser()
parser.add_argument('--state-size', type=int, default=64)
parser.add_argument('--agent-tasks', type=int, default=5)
parser.add_argument('--nodes', type=int, default=4)
parser.add_argument('--host', type=str, default='localhost')
parser.add_argument('--port', type=int, default=9900)

args = parser.parse_args()
app = flask.Flask(__name__)
app.config['state_size'] = args.state_size
app.config['action_size'] = args.agent_tasks*args.nodes


@app.route('/')
def get_model():
    state_size = app.config.get('state_size')
    action_size = app.config.get('action_size')
    name = 'actor'
    return DQNActor(state_size, action_size, name)


@app.route('/act', methods=['POST'])
def act():
    global graph
    data = request.get_json(force=True)
    state = np.array(data['state']).reshape(1, model.STATE)
    mask = np.array(data['mask'])
    sched = data['sched']
    with graph.as_default():
        action = model.act(state, mask, sched)
    return jsonify(action=int(action))


@app.route('/test', methods=['POST'])
def test():
    global graph
    data = request.get_json(force=True)
    state = np.array(data['state']).reshape(1, model.STATE)
    mask = np.array(data['mask'])
    sched = data['sched']
    with graph.as_default():
        eps = model.epsilon
        model.epsilon = 0.0
        action = model.act(state, mask, sched)
        model.epsilon = eps
    return jsonify(action=int(action))


@app.route('/replay', methods=['POST'])
def replay():
    global graph
    data = request.get_json()
    batch_size = data['batch_size']
    with graph.as_default():
        response = model.replay(batch_size)
    return response


@app.route('/remember', methods=['POST'])
def remember():
    data = request.get_json()
    SARSA = data['SARSA']
    state = np.array(SARSA[0]).reshape(1, model.STATE)
    action = int(SARSA[1])
    reward = float(SARSA[2])
    next_state = np.array(SARSA[3]).reshape(1, model.STATE)
    done = bool(SARSA[4])
    response = {'is_remembered': model.remember((state, action, reward, next_state, done))}
    return response


@app.route('/save', methods=['POST'])
def save():
    json_model = model.save('model')
    return send_from_directory(os.getcwd(), 'model.h5')


if __name__ == '__main__':
    graph = tf.get_default_graph()
    model = get_model()
    URL = f'http://{args.host}:{args.port}/'
    app.run(host=args.host, port=args.port, debug=True)





