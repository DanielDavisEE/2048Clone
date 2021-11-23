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
import os
import tempfile
import time
import gin

from typing import Any, Iterable, Optional, Text

from tf_agents.drivers import py_driver
from tf_agents.metrics import py_metrics
from tf_agents.typing import types

from tf_agents.utils import common
from tf_agents.utils import numpy_storage

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

        self._step_cap = 500
        self._repeated_step_cap = 10
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
        if self._steps_since_score_increase >= self._step_cap or self._steps >= self._step_cap:
            exponent_based = self.condition_state()
            return ts.termination(exponent_based, reward=exponent_based.max()) 
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


class CustomNumpyDeque(py_metrics.NumpyDeque):
    def max(self, dtype: Optional[np.dtype] = None):
        if self._len == self._buffer.shape[0]:
            return np.max(self._buffer).astype(dtype)
    
        assert self._start_index == 0
        return np.max(self._buffer[:self._len]).astype(dtype)

@gin.configurable
class MaxTileMetric(py_metrics.StreamingMetric):
    """Computes the average episode length."""

    def __init__(self,
                 name: Text = 'MaxTile',
                 buffer_size: types.Int = 10,
                 batch_size: Optional[types.Int] = None):
        """Creates an MaxTileMetric."""
        self._np_state = numpy_storage.NumpyState()
        # Set a dummy value on self._np_state.episode_return so it gets included in
        # the first checkpoint (before metric is first called).
        self._np_state.max_tile = np.int64(0)
        super(MaxTileMetric, self).__init__(
            name, buffer_size=buffer_size, batch_size=batch_size)
        self._buffer = CustomNumpyDeque(maxlen=buffer_size, dtype=np.float64)

    def _reset(self, batch_size):
        """Resets stat gathering variables."""
        self._np_state.max_tile = np.zeros(
            shape=(batch_size,), dtype=np.int64)

    def _batched_call(self, trajectory):
        """Processes the trajectory to update the metric.
        Args:
          trajectory: a tf_agents.trajectory.Trajectory.
        """
        max_tile = self._np_state.max_tile

        max_tile = np.maximum(max_tile, 2 ** trajectory[1].max())

        is_last = np.where(trajectory.is_last())
        self.add_to_buffer(max_tile[is_last])

    def result(self) -> np.float32:
        """Returns the value of this metric."""
        if self._buffer:
            return self._buffer.max(dtype=np.int32)
        return np.array(0, dtype=np.int32)


def train_agent():
    # Hyperparameters
    tempdir = tempfile.gettempdir()

    # Use "num_iterations = 1e6" for better results (2 hrs)
    # 1e5 is just so this doesn't take too long (1 hr)
    num_iterations = 100000 # @param {type:"integer"}

    initial_collect_steps = 100 # @param {type:"integer"}
    initial_collect_episodes = 100 # @param {type:"integer"}
    collect_episodes_per_iteration = 1 # @param {type:"integer"}
    replay_buffer_capacity = initial_collect_steps * initial_collect_episodes # @param {type:"integer"}

    batch_size = 256 # @param {type:"integer"}

    critic_learning_rate = 3e-4 # @param {type:"number"}
    actor_learning_rate = 3e-4 # @param {type:"number"}
    alpha_learning_rate = 3e-4 # @param {type:"number"}
    target_update_tau = 0.005 # @param {type:"number"}
    target_update_period = 1 # @param {type:"number"}
    gamma = 0.99 # @param {type:"number"}
    reward_scale_factor = 1.0 # @param {type:"number"}

    actor_fc_layer_params = (256, 256)
    critic_joint_fc_layer_params = (256, 256)

    log_interval = 1000 # @param {type:"integer"}

    num_eval_episodes = 20 # @param {type:"integer"}
    eval_interval = 10000 # @param {type:"integer"}

    policy_save_interval = 5000 # @param {type:"integer"}

    # Environment
    collect_env = GameEnv2048()
    eval_env = GameEnv2048()    

    # Agent
    observation_spec, action_spec, time_step_spec = (
        spec_utils.get_tensor_specs(collect_env))

    critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=critic_joint_fc_layer_params,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=actor_fc_layer_params,
        continuous_projection_net=(
            tanh_normal_projection_network.TanhNormalProjectionNetwork))

    train_step = train_utils.create_train_step()

    tf_agent = sac_agent.SacAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=alpha_learning_rate),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        train_step_counter=train_step)

    tf_agent.initialize()

    # Replay Buffer
    #rate_limiter=reverb.rate_limiters.SampleToInsertRatio(samples_per_insert=3.0, min_size_to_sample=3, error_buffer=3.0)
    table_name = 'uniform_table'
    table = reverb.Table(
        table_name,
        max_size=replay_buffer_capacity,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1))

    reverb_server = reverb.Server([table])

    reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
        tf_agent.collect_data_spec,
        sequence_length=2,
        table_name=table_name,
        local_server=reverb_server)

    dataset = reverb_replay.as_dataset(
        sample_batch_size=batch_size, num_steps=2).prefetch(50)
    experience_dataset_fn = lambda: dataset

    # Policies
    tf_eval_policy = tf_agent.policy
    eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_eval_policy, use_tf_function=True)

    tf_collect_policy = tf_agent.collect_policy
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_collect_policy, use_tf_function=True)

    random_policy = random_py_policy.RandomPyPolicy(
        collect_env.time_step_spec(), collect_env.action_spec())

    # Actors
    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        reverb_replay.py_client,
        table_name,
        sequence_length=2,
        stride_length=1)

    initial_collect_actor = actor.Actor(
        collect_env,
        random_policy,
        train_step,
        steps_per_run=initial_collect_steps,
        episodes_per_run=initial_collect_episodes,
        observers=[rb_observer])
    initial_collect_actor.run()

    env_step_metric = py_metrics.EnvironmentSteps()
    collect_actor = actor.Actor(
        collect_env,
        collect_policy,
        train_step,
        episodes_per_run=collect_episodes_per_iteration,
        metrics=actor.collect_metrics(10),
        summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
        observers=[rb_observer, env_step_metric])

    eval_metrics = actor.eval_metrics(num_eval_episodes)
    eval_metrics.append(MaxTileMetric(buffer_size=num_eval_episodes))
    eval_actor = actor.Actor(
        eval_env,
        eval_policy,
        train_step,
        episodes_per_run=num_eval_episodes,
        metrics=eval_metrics,
        summary_dir=os.path.join(tempdir, 'eval'),
    )

    # Learners
    saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

    # Triggers to save the agent's policy checkpoints.
    learning_triggers = [
        triggers.PolicySavedModelTrigger(
            saved_model_dir,
            tf_agent,
            train_step,
            interval=policy_save_interval),
        triggers.StepPerSecondLogTrigger(train_step, interval=1000),
    ]

    agent_learner = learner.Learner(
        tempdir,
        train_step,
        tf_agent,
        experience_dataset_fn,
        triggers=learning_triggers)

    # Metrics and Evaluation
    def get_eval_metrics():
        eval_actor.run()
        results = {}
        for metric in eval_actor.metrics:
            results[metric.name] = metric.result()
        return results

    metrics = get_eval_metrics()

    def log_eval_metrics(step, metrics):
        eval_results = (', ').join(
            f'{name} = {result:.2f}' for name, result in metrics.items())
        print(f'step = {step}: {eval_results}')

    log_eval_metrics(0, metrics)

    # Training the Agent
    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = get_eval_metrics()["AverageReturn"]
    returns = [avg_return]

    start_time = time.time()
    for _ in range(num_iterations):
        # Training.
        collect_actor.run()
        loss_info = agent_learner.run(iterations=1)

        # Evaluating.
        step = agent_learner.train_step_numpy

        #if int(np.log2(step)) == np.log2(step):
            #print(f"Training Step: {step:>6} ({time.time() - start_time:.2f} s)")

        if eval_interval and step % eval_interval == 0:
            metrics = get_eval_metrics()
            log_eval_metrics(step, metrics)
            returns.append(metrics["AverageReturn"])

        if log_interval and step % log_interval == 0:
            print(f'step = {step}: loss = {loss_info.loss.numpy()}')

    rb_observer.close()
    reverb_server.stop()

    # Visualization
    steps = range(0, num_iterations + 1, eval_interval)
    plt.plot(steps, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Step')
    plt.ylim()
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