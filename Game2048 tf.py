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

from tf_agents.utils import common
from tf_agents.drivers import py_driver
from tf_agents.metrics import py_metrics

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

from tf_agents.policies import actor_policy
from tf_agents.policies import q_policy
from tf_agents.policies import greedy_policy

from tf_agents.policies import py_policy
from tf_agents.policies import random_py_policy
from tf_agents.policies import scripted_py_policy

from tf_agents.policies import tf_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import py_tf_eager_policy

from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step as ts

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network

from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils

from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers

from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils

ACTION_UP = 0
ACTION_RIGHT = 1
ACTION_DOWN = 2
ACTION_LEFT = 3

class GameEnv2048(py_environment.PyEnvironment):

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(4,), dtype=np.float32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4,4,1), dtype=np.float32, minimum=0, name='observation')

        self._step_cap = 10
        self._softmax = tf.keras.layers.Softmax()

        self._state = np.zeros([4, 4], dtype=np.float32)
        self._merged = np.zeros([4, 4], dtype=np.bool8)
        self._spawn_tile(2)
        self._score = np.array(0, dtype=np.float32)
        self._steps = 0
        self._steps_since_score_increase = 0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = np.zeros([4, 4], dtype=np.float32)
        self._merged = np.zeros([4, 4], dtype=np.bool8)
        self._spawn_tile(2)
        self._score = np.array(0, dtype=np.float32)
        self._steps = 0
        self._steps_since_score_increase = 0
        self._episode_ended = False
        self.condition_state()
        return ts.restart(np.array(self.condition_state(), dtype=np.float32))
    
    def condition_state(self):
        tmp = self._state.copy().reshape(4,4,1)
        return np.log2(tmp, out=tmp, where=tmp!=0)
    
    def _step(self, action):
        """
        up:    0
        right: 1
        down:  2
        left:  3
        """
        if self._steps_since_score_increase >= self._step_cap:
            return ts.termination(self.condition_state(), self._score) 
        self._steps += 1
        self._steps_since_score_increase += 1
        
        if type(action) is int:
            direction = action
        else:
            directions = [ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT]
            direction = np.random.choice(
                directions, 
                p=self._softmax(action).numpy()
            )

        if not self._check_viable_move(direction):
            return ts.transition(self.condition_state(), reward=-0.1, discount=0.995)

        row_iter = (0, 4)
        if direction == ACTION_DOWN: # If direction is down, iterate rows upwards
            row_iter = (3, -1, -1)
        col_iter = (0, 4)
        if direction == ACTION_RIGHT: # If direction is right, iterate columns leftwards
            col_iter = (3, -1, -1)

        for m in range(*row_iter):
            for n in range(*col_iter):
                if self._state[m,n] == 0:
                    continue

                self._move_tile((m, n), direction)
        
        merges = self._merged.sum()
        self._merged = np.zeros([4, 4], dtype=np.bool8)
        gameLost = self._spawn_tile()
        
        if gameLost:
            self._episode_ended = True

        if self._episode_ended:
            exponent_based = self.condition_state()
            return ts.termination(exponent_based, reward=exponent_based.max()+self._steps/100)
        else:
            return ts.transition(self.condition_state(), reward=(merges-1)/8, discount=0.995)

    def _merge(self, coords, new_coords):
        """
        Merge two tiles together i.e. the new_coords value will double and the coords value will be deleted.
        It is assumed that the merge is valid, if not the changes will occur anyway.
        """
        self._state[new_coords] *= 2
        self._state[coords] = 0
        self._score += self._state[new_coords]
        self._steps_since_score_increase = 0
        self._merged[new_coords] = True

    def _move_tile(self, coords, action):
        """ Move a tile as far as possible in the indicated direction, merge 
        with other tile if they have the same value.
        Return a tuble of initial coordinates, new coordinates, and whether it 
        has been merged and whether it was moved.
        coords -> (row, col)
        current_coords -> (row, col)
        merged -> bool
        inPlace -> bool
        
        up:    0
        right: 1
        down:  2
        left:  3
        """

        def increment(c):
            if action == ACTION_UP:
                return c[0] - 1, c[1]
            elif action == ACTION_RIGHT:
                return c[0], c[1] + 1
            elif action == ACTION_DOWN:
                return c[0] + 1, c[1]
            elif action == ACTION_LEFT:
                return c[0], c[1] - 1
            else:
                raise ValueError
            
        inBounds = lambda c: 0 <= c[0] < 4 and 0 <= c[1] < 4

        safe_coords = coords
        next_coords = increment(safe_coords)

        # If there is a valid, empty tile at the location of safe_coords, check the next location
        while (inBounds(next_coords) and
               self._state[next_coords] == 0):

            safe_coords, next_coords = next_coords, increment(next_coords)

        if coords != safe_coords:
            self._state[safe_coords] = self._state[coords]
            self._state[coords] = 0

        # If the location of next_coords is still a valid tile, and has a value equal to the
        #    current location, and the next_coords tile hasn't already been merged, merge them on the next_coords
        if (inBounds(next_coords) and 
            self._state[safe_coords] == self._state[next_coords] and
            self._merged[next_coords] == False):

            self._merge(safe_coords, next_coords)

    def _check_viable_move(self, action):
        """
        Iterates over the game board checking if a player's move is viable.
        i.e. There is a possibility for tiles to move.
        It checks each row/column in the direction of the move one after the other, in the same direction as the move.
        For upwards and leftwards moves this means it must iterate the row/column in reverse.
        For upwards and downwards moves this means the coordinates must be switched to iterate by column then row.

        up:    0
        right: 1
        down:  2
        left:  3
        """

        if action in [ACTION_UP, ACTION_LEFT]:
            inner_iter = (0, 4)
        else:
            inner_iter = (3, -1, -1)

        arrange_coords = lambda x, y: (y, x) if action in [ACTION_UP, ACTION_DOWN] else (x, y)

        # Iterate orthogonally to the move
        for m in range(0, 4):

            spaceAvailable, last_num = False, 0

            # Iterate in the same direction as the move
            for n in range(*inner_iter):
                value = self._state[arrange_coords(m, n)] # Swap coordinates if vertical move

                if value > 0: # Where 0 is the empty value
                    if value == last_num or spaceAvailable: # Merge is possible or There is a gap for the number to move into
                        return True
                    last_num = value
                else:
                    spaceAvailable = True

        return False

    def _spawn_tile(self, repeat=1):
        """
        Spawns either a 2 or a 4 in an empty square of the gameboard. By default, this happens once at a time.
        After each tile spawn, if there are no empty spots, check if a valid move exists.
        """
        empty_tiles = (self._state == 0).sum()
        assert empty_tiles > 0
            
        def isGameOver():
            """
            If the game board is full, check for potential tile merges which would allow further moves.
            """
            for i in range(3):
                for j in range(3):
                    if(self._state[i, j] == self._state[i + 1, j] or
                       self._state[i, j] == self._state[i, j + 1]):
                        return False
         
            for j in range(3):
                if(self._state[3, j] == self._state[3, j + 1]):
                    return False
         
            for i in range(3):
                if(self._state[i, 3] == self._state[i + 1, 3]):
                    return False
                
            return True
        
        gameOver = False
        for _ in range(repeat):
            index = rd.randrange(0, empty_tiles)

            # Find the missing value which corresponds to index by iterating through
            #    the gameboard and decrementing the index to 0.
            for i, row in enumerate(self._state):
                for j, value in enumerate(row):
                    if value == 0:
                        index -= 1
                        if index < 0:
                            self._state[i,j] = 2 if rd.random() < 0.9 else 4
                            empty_tiles -= 1
                            break
                        
                if index < 0:
                    break

            # If the board has no empty slots, check for viable moves in every direction.
            if empty_tiles == 0:
                if isGameOver():
                    return True # The game is lost             

        return False

    def print_board(game_inst):
        """
        Prints the gameboard row by row with _ as empty tiles
        """
        print(game_inst._score)
        board_string = ''
        col = 0
        for row in game_inst._state:
            for tile in row:
                value = '_' if tile == 0 else tile
                board_string += f"{value:^4}"
            board_string += '\n'
        print(board_string)


def train_agent():
    # Hyperparameters
    num_iterations = 10000 # @param {type:"integer"}
    collect_episodes_per_iteration = 5 # @param {type:"integer"}
    replay_buffer_capacity = 2000 # @param {type:"integer"}
    
    # (filters, kernel_size, stride)
    conv_layer_params = None#[(32, 3, 1),
                         #(64, 2, 1)]
    fc_layer_params = [300, 300, 200, 200, 100]
    dropout_layer_params = [0.75, 0.75, 0.75, 0.75, 0.75]

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
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
        dropout_layer_params=dropout_layer_params)

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
        max_tile = 0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
                max_tile = max(max_tile, 2 ** time_step[3].numpy().max())
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0], max_tile

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
            print(f'step = {step}: loss = {train_loss.loss}')

        if step % eval_interval == 0:
            avg_return, max_tile = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
            print(f'step = {step}: Average Return = {avg_return}, Max Tile = {max_tile}')
            returns.append(avg_return)

    # Visualisation    
    steps = range(0, num_iterations + 1, eval_interval)
    plt.plot(steps, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Step')
    #plt.ylim(top=250)
    plt.show()


def test_env():
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
            game_instance.print_board()

        print(f"    You Lost\n\nFinal Score: {game_instance.score}\n\nPlay Again? (y/n)")
        if input().lower() == 'n':
            break
    print('Thanks for playing.')


if __name__ == '__main__':
    train_agent()