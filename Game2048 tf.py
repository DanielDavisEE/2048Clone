from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import random as rd

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

class GameEnv2048(py_environment.PyEnvironment):

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(16,), dtype=np.int32, minimum=0, name='observation')
        
        self._state = [0] * 16
        self._merged = [False] * 16
        self._spawn_tile(2)
        self._score = 0
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
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):
        """
        up:    0
        right: 1
        down:  2
        left:  3
        """
        if not self._check_viable_move(action):
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=0.0, discount=1.0)
        
        row_iter = (0, 4)
        if action == 2:
            row_iter = (3, -1, -1)
        col_iter = (0, 4)
        if action == 1:
            col_iter = (3, -1, -1)
        
        for m in range(*row_iter):
            for n in range(*col_iter):
                index = m * 4 + n
                if self._state[index] == 0:
                    continue
                
                self._move_tile(index, action)
        
        self._merged = [False] * 16
        gameLost = self._spawn_tile()
        if gameLost:
            self._episode_ended = True

        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype=np.int32), self._score)
        else:
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=0.0, discount=1.0)
                            
    def _merge(self, index, new_index):
        """
        Merge two tiles together i.e. the new_index value will double and the index value will be deleted.
        It is assumed that the merge is valid, if not the changes will occur anyway.
        """
        self._state[new_index] *= 2
        self._state[index] = 0
        self._score += self._state[new_index]
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
    
        
        
class CardGameEnv(py_environment.PyEnvironment):

    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
        shape=(1,), dtype=np.int32, minimum=0, name='observation')
        self._state = 0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Make sure episodes don't go on forever.
        if action == 1:
            self._episode_ended = True
        elif action == 0:
            new_card = np.random.randint(1, 11)
            self._state += new_card
        else:
            raise ValueError('`action` should be 0 or 1.')

        if self._episode_ended or self._state >= 21:
            reward = self._state - 21 if self._state <= 21 else -21
            return ts.termination(np.array([self._state], dtype=np.int32), reward)
        else:
            return ts.transition(
          np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)


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
    
    def print_board(game_inst):
        """
        Prints the gameboard row by row with _ as empty tiles
        """
        print(game_inst._score)
        board_string = ''
        col = 0
        for tile in game_inst._state:
            value = '_' if tile is 0 else tile
            board_string += f"{value:^4}"
            col += 1
            if col % 4 == 0:
                board_string += '\n'
        print(board_string)
    
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
    environment = CardGameEnv()
    utils.validate_py_environment(environment, episodes=5)
    
    
    #main()