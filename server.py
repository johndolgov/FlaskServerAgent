import flask
from flask import request, send_from_directory
from actor import DQNActor
from actorlstm import DQNLSTMActor
from flask import jsonify
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
import os

parser = ArgumentParser()
parser.add_argument('--state-size', type=int, default=64)
parser.add_argument('--agent-tasks', type=int, default=5)
parser.add_argument('--is-lstm-agent', type=bool, default=False)
parser.add_argument('--first-lstm-layer', type=int, default=32)
parser.add_argument('--second-lstm-layer', type=int,default=0)
parser.add_argument('--first-fc-layer', type=int, default=1024)
parser.add_argument('--second-fc-layer', type=int, default=512)
parser.add_argument('--third-fc-layer', type=int, default=256)
parser.add_argument('--seq-size', type=int, default=5)
parser.add_argument('--nodes', type=int, default=4)
parser.add_argument('--load', type=bool, default=False)
parser.add_argument('--host', type=str, default='localhost')
parser.add_argument('--port', type=int, default=9900)

args = parser.parse_args()
app = flask.Flask(__name__)
app.config['state_size'] = args.state_size
app.config['action_size'] = args.agent_tasks*args.nodes
app.config['is_lstm_agent'] = args.is_lstm_agent
app.config['first_lstm_layer'] = args.first_lstm_layer
app.config['second_lstm_layer'] = args.second_lstm_layer
app.config['first_fc_layer'] = args.first_fc_layer
app.config['second_fc_layer'] = args.second_fc_layer
app.config['third_fc_layer'] = args.third_fc_layer
app.config['seq_size'] = args.seq_size
app.config['load'] = args.load


@app.route('/')
def get_model():
    state_size = app.config.get('state_size')
    action_size = app.config.get('action_size')
    is_lstm = app.config.get('is_lstm_agent')
    seq_size = app.config.get('seq_size')
    load = app.config.get('load')
    first_lstm_layer = app.config.get('first_lstm_layer')
    second_lstm_layer = app.config.get('second_lstm_layer')
    first_fc_layer = app.config.get('first_fc_layer')
    second_fc_layer = app.config.get('second_fc_layer')
    third_fc_layer = app.config.get('third_fc_layer')
    name = 'actor'
    if not load:
        if not is_lstm:
            return DQNActor(first_fc_layer, second_fc_layer, third_fc_layer, state_size, action_size, name)
        else:
            return DQNLSTMActor(state_size, action_size, seq_size, first_lstm_layer, second_lstm_layer, name)
    else:
        if not is_lstm:
            fc_model = DQNActor(first_fc_layer, second_fc_layer, third_fc_layer,state_size, action_size, name)
            fc_model.load('model_fc.h5')
            return fc_model
        else:
            lstm_model = DQNLSTMActor(state_size, action_size, seq_size, first_lstm_layer, second_lstm_layer, name)
            lstm_model.load('model_lstm.h5')
            return lstm_model


@app.route('/act', methods=['POST'])
def act():
    global graph
    data = request.get_json(force=True)
    if not args.is_lstm_agent:
        state = np.array(data['state']).reshape(1, model.STATE)
        mask = np.array(data['mask'])
        sched = data['sched']
    else:
        state = np.asarray(data['state']).reshape((1, model.seq_size, model.STATE))
        mask = np.array(data['mask'])
        sched = data['sched']
    with graph.as_default():
        action = model.act(state, mask, sched)
    return jsonify(action=int(action))


@app.route('/test', methods=['POST'])
def test():
    global graph
    data = request.get_json(force=True)
    if not args.is_lstm_agent:
        state = np.asarray(data['state']).reshape(1, model.STATE)
    else:
        state = np.asarray(data['state']).reshape((1, model.seq_size, model.STATE))
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
    if not args.is_lstm_agent:
        state = np.asarray(SARSA[0]).reshape(1, model.STATE)
        next_state = np.asarray(SARSA[3]).reshape(1, model.STATE)
    else:
        state = np.asarray(SARSA[0]).reshape((1, model.seq_size, model.STATE))
        next_state = np.asarray(SARSA[3]).reshape((1, model.seq_size, model.STATE))
    action = int(SARSA[1])
    reward = float(SARSA[2])
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





