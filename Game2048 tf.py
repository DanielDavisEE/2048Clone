from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import reverb

from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec

from tf_agents.networks import network
from tf_agents.networks import actor_distribution_network

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym

from tf_agents.policies import py_policy
from tf_agents.policies import random_py_policy
from tf_agents.policies import scripted_py_policy

from tf_agents.policies import tf_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import actor_policy
from tf_agents.policies import q_policy
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy

from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step as ts

from tf_agents.agents.reinforce import reinforce_agent

from tf_agents.drivers import py_driver

from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils

from tf_agents.utils import common

class GameEnv2048(py_environment.PyEnvironment):

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(4,), dtype=np.float32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(16,), dtype=np.int32, minimum=0, name='observation')

        self._step_cap = 10
        self._softmax = tf.keras.layers.Softmax()

        self._state = [0] * 16
        self._merged = [False] * 16
        self._spawn_tile(2)
        self._score = 0
        self._steps_since_score_increase = 0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = [0] * 16
        self._merged = [False] * 16
        self._spawn_tile(2)
        self._score = 0
        self._steps_since_score_increase = 0
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):
        """
        up:    0
        right: 1
        down:  2
        left:  3
        """
        if self._steps_since_score_increase >= self._step_cap:
            return ts.termination(np.array(self._state, dtype=np.int32), self._score) 
        self._steps_since_score_increase += 1

        directions = [0, 1, 2, 3]
        direction = np.random.choice(
            directions, 
            p=self._softmax(action).numpy()
        )

        if not self._check_viable_move(direction):
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=0.0, discount=1.0)

        row_iter = (0, 4)
        if direction == 2:
            row_iter = (3, -1, -1)
        col_iter = (0, 4)
        if direction == 1:
            col_iter = (3, -1, -1)

        for m in range(*row_iter):
            for n in range(*col_iter):
                index = m * 4 + n
                if self._state[index] == 0:
                    continue

                self._move_tile(index, direction)

        self._merged = [False] * 16
        gameLost = self._spawn_tile()
        if gameLost:
            self._episode_ended = True

        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype=np.int32), self._score)
        else:
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=0.0, discount=0.95)

    def _merge(self, index, new_index):
        """
        Merge two tiles together i.e. the new_index value will double and the index value will be deleted.
        It is assumed that the merge is valid, if not the changes will occur anyway.
        """
        self._state[new_index] *= 2
        self._state[index] = 0
        self._score += self._state[new_index]
        self._steps_since_score_increase = 0
        self._merged[new_index] = True

    def _move_tile(self, index, action):
        """ Move a tile as far as possible in the indicated direction, merge 
        with other tile if they have the same value.
        Return a tuble of initial coordinates, new coordinates, and whether it 
        has been merged and whether it was moved.
        index -> (row, col)
        current_index -> (row, col)
        merged -> bool
        inPlace -> bool
        """

        isVert = action % 2 == 0
        isPos = (action + 1) // 2 == 1

        shift_amount = 4 if isVert else 1
        increment = lambda i: i + shift_amount if isPos else i - shift_amount

        row = index // 4
        inBounds = lambda i: 0 <= i < 16 if isVert else row == i // 4

        safe_index = index
        next_index = increment(safe_index)

        # If there is a valid, empty tile at the location of safe_index, check the next location
        while (inBounds(next_index) and
               self._state[next_index] == 0):

            safe_index, next_index = next_index, increment(next_index)

        if index != safe_index:
            self._state[safe_index] = self._state[index]
            self._state[index] = 0

        # If the location of next_index is still a valid tile, and has a value equal to the
        #    current location, and the next_index tile hasn't already been merged, merge them on the next_index
        if (inBounds(next_index) and 
            self._state[safe_index] == self._state[next_index] and
            self._merged[next_index] == False):

            self._merge(safe_index, next_index)

    def _check_viable_move(self, action):
        """
        Iterates over the game board checking if a player's move is viable.
        i.e. There is a possibility for tiles to move.
        It checks each row/column in the direction of the move one after the other, in the same direction as the move.
        For upwards and leftwards moves this means it must iterate the row/column in reverse.
        For upwards and downwards moves this means the coordinates must be switched to iterate by column then row.

        up:    1
        right: 2
        down:  3
        left:  4
        """

        isPos = (action + 1) // 2 == 1
        if isPos:
            inner_iter = (0, 4)
        else:
            inner_iter = (3, -1, -1)

        isVert = action % 2 == 0
        to_index = lambda x, y: y * 4 + x if isVert else x * 4 + y

        # Iterate orthogonally to the move
        for m in range(0, 4):

            numFound, last_num = False, 0

            # Iterate in the same direction as the move
            for n in range(*inner_iter):
                value = self._state[to_index(m, n)] # Swap coordinates if vertical move

                if value > 0: # Where 0 is the empty value
                    if value == last_num: # Merge is possible
                        return True
                    numFound, last_num = True, value

                elif numFound == True: # There is a gap for a previously found number to move into
                    return True

        return False

    def _spawn_tile(self, repeat=1):
        """
        Spawns either a 2 or a 4 in an empty square of the gameboard. By default, this happens once at a time.
        After each tile spawn, if there are no empty spots, check if a valid move exists.
        """
        empty_tiles = self._state.count(0)
        assert empty_tiles > 0

        for _ in range(repeat):
            index = rd.randrange(0, empty_tiles)
            index_tmp = index

            # Find the missing value which corresponds to index by iterating through
            #    the gameboard and decrementing the index to 0.
            for i, value in enumerate(self._state):
                if value == 0:
                    if index_tmp == 0:
                        self._state[i] = 2 if rd.random() < 0.9 else 4
                        empty_tiles -= 1
                        break
                    index_tmp -= 1

            # If the board has no empty slots, check for viable moves in every direction.
            if empty_tiles == 0:
                for direction in range(1, 5):
                    if self._check_viable_move(direction):
                        break # A viable move exists
                else:
                    return True # The game is lost

        return False


def print_board(game_inst):
    """
    Prints the gameboard row by row with _ as empty tiles
    """
    print(game_inst._score)
    board_string = ''
    col = 0
    for tile in game_inst._state:
        value = '_' if tile == 0 else tile
        board_string += f"{value:^4}"
        col += 1
        if col % 4 == 0:
            board_string += '\n'
    print(board_string)

def main():

    move_dict = {'quit': 'q',
                 'q': 'q',
                 'w': 0,
                 'd': 1,
                 's': 2,
                 'a': 3,
                 'shhh': 'shhh'
                 }

    verbose = True

    def get_move():
        nonlocal verbose
        move = None
        while move is None:
            try:
                move = move_dict[input().lower()]
            except KeyError:
                if verbose:
                    print("Invalid move.")
            else:
                if move == 'shhh':
                    verbose = False
                    move = None
        return move

    print("""To play, use the 'wasd' keys to input moves.
To quit, type 'quit' or 'q'.
To stop the 'Invlaid move.' dialogue, type 'shhh'.
Follow all inputs with a newline press.
    """)

    play = True

    while play == True:
        gameLost = False
        game_instance = GameEnv2048()
        print_board(game_instance)
        while not gameLost:
            move = get_move()
            if move == 'q':
                break
            game_instance._step(move)
            print_board(game_instance)

        print(f"    You Lost\n\nFinal Score: {game_instance.score}\n\nPlay Again? (y/n)")
        if input().lower() == 'n':
            break
    print('Thanks for playing.')


if __name__ == '__main__':
    
    # Hyperparameters
    num_iterations = 200 # @param {type:"integer"}
    collect_episodes_per_iteration = 2 # @param {type:"integer"}
    replay_buffer_capacity = 2000 # @param {type:"integer"}

    fc_layer_params = (256,256)

    learning_rate = 1e-3 # @param {type:"number"}
    log_interval = 25 # @param {type:"integer"}
    num_eval_episodes = 10 # @param {type:"integer"}
    eval_interval = 50 # @param {type:"integer"}

    # Environment
    train_py_env = GameEnv2048()
    eval_py_env = GameEnv2048()

    #env_name = "CartPole-v0" # @param {type:"string"}
    #train_py_env = suite_gym.load(env_name)
    #eval_py_env = suite_gym.load(env_name)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # Agent
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

    # Optimiser
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    tf_agent = reinforce_agent.ReinforceAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter)
    tf_agent.initialize()

    # Metrics and Evaluation
    def compute_avg_return(environment, policy, num_episodes=10):

        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    # Replay Buffer
    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(
        tf_agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature)
    table = reverb.Table(
        table_name,
        max_size=replay_buffer_capacity,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        tf_agent.collect_data_spec,
        table_name=table_name,
        sequence_length=None,
        local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddEpisodeObserver(
        replay_buffer.py_client,
        table_name,
        replay_buffer_capacity
    )

    # Data Collection
    def collect_episode(environment, policy, num_episodes):

        driver = py_driver.PyDriver(
            environment,
            py_tf_eager_policy.PyTFEagerPolicy(
                policy, use_tf_function=True),
            [rb_observer],
            max_episodes=num_episodes)
        initial_time_step = environment.reset()
        driver.run(initial_time_step)

    # Training the Agent

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    tf_agent.train = common.function(tf_agent.train)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    returns = [avg_return]

    print("Starting training...")
    for _ in range(num_iterations):

        # Collect a few episodes using collect_policy and save to the replay buffer.
        collect_episode(
            train_py_env, tf_agent.collect_policy, collect_episodes_per_iteration)

        # Use data from the buffer and update the agent's network.
        iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
        trajectories, _ = next(iterator)
        train_loss = tf_agent.train(experience=trajectories)  

        replay_buffer.clear()

        step = tf_agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)

    # Visualisation    
    steps = range(0, num_iterations + 1, eval_interval)
    plt.plot(steps, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Step')
    #plt.ylim(top=250)
    plt.show()
