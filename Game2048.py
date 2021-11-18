"""

"""

import random as rd


class GameState():
    
    def __init__(self, game_board=None):
        """ Initialises values for a new instance of 2048, defaults to 4 by 4 board.
        Creates game board, 
        """
        if game_board is None:
            self.game_board = [None] * 16
        else:
            self.game_board = game_board
            
        self.score = 0
        self.merged = [False] * 16
        
        if not any(self.game_board):
            self._spawn_tile(2)
        
    def __str__(self):
        """
        Prints the gameboard row by row with _ as empty tiles
        """
        board_string = ''
        col = 0
        for tile in self.game_board:
            value = '_' if tile is None else tile
            board_string += f"{value:^4}"
            col += 1
            if col % 4 == 0:
                board_string += '\n'
        return board_string
        
    def _to_index(self, coords):
        """
        Converts (row, col) style coordinates to a single list index
        """
        return coords[0] * 4 + coords[1]
    
    def __getitem__(self, coords):
        """
        Gets a value on the gameboard using (row, col) style coordinates
        """
        return self.game_board[self._to_index(coords)]
    
    def __setitem__(self, coords, value):
        """
        Gets a value on the gameboard using (row, col) style coordinates
        """
        self.game_board[self._to_index(coords)] = value
                            
    def _merge(self, coords, new_coords):
        """
        Merge two tiles together i.e. the new_coords value will double and the coords value will be deleted.
        It is assumed that the merge is valid, if not the changes will occur anyway.
        """
        self[new_coords] *= 2
        self[coords] = None
        self.score += self[new_coords]
        self.merged[self._to_index(new_coords)] = True
                
    def _move_tile(self, coords, isVert=True, isPos=True):
        """ Move a tile as far as possible in the indicated direction, merge 
        with other tile if they have the same value.
        Return a tuble of initial coordinates, new coordinates, and whether it 
        has been merged and whether it was moved.
        coords -> (row, col)
        current_coords -> (row, col)
        merged -> bool
        inPlace -> bool
        """
        if isVert:
            increment = lambda x, y: (x + (1 if isPos else -1), y)
        else:
            increment = lambda x, y: (x, y + (1 if isPos else -1))
        
        merged = False
        current_coords = coords
        new_coords = increment(*current_coords)
        
        # If there is a valid, empty tile at the location of new_coords, move the value to it
        while (0 <= new_coords[0] < 4 and
               0 <= new_coords[1] < 4 and
               self[new_coords] is None):
            
            self[new_coords] = self[current_coords]
            self[current_coords] = None
            
            current_coords, new_coords = new_coords, increment(*new_coords)
        
        # If the location of new_coords is still a valid tile, and has a value equal to the
        #    current location, merge them on the new_coords
        if (0 <= new_coords[0] < 4 and
            0 <= new_coords[1] < 4):
            
            if (self[current_coords] == self[new_coords] and
                self.merged[self._to_index(new_coords)] == False):
                
                self._merge(current_coords, new_coords)
                current_coords = new_coords
                merged = True
                
        return [coords, current_coords, merged, coords == current_coords]
    
    def check_viable_move(self, direction):
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
        
        horz = lambda x, y: (x, y)
        vert = lambda x, y: (y, x)
        check_dict = {
            #  [inner_iter, dir_type]
            1: [(3, -1, -1), vert],
            2: [(0, 4), horz],
            3: [(0, 4), vert],
            4: [(3, -1, -1), horz],
        }
        
        # Iterate orthogonally to the move
        for m in range(0, 4):
            
            numFound, last_num = False, None
            
            # Iterate in the same direction as the move
            for n in range(*check_dict[direction][0]):
                value = self[check_dict[direction][1](m, n)] # Swap coordinates if vertical move
                
                if value is not None: # Where None is the empty value
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
        empty_tiles = self.game_board.count(None)
        assert empty_tiles > 0
            
        for _ in range(repeat):
            index = rd.randrange(0, empty_tiles)
            index_tmp = index
            
            # Find the missing value which corresponds to index by iterating through
            #    the gameboard and decrementing the index to 0.
            for i, value in enumerate(self.game_board):
                if value == None:
                    if index_tmp == 0:
                        self.game_board[i] = 2 if rd.random() < 0.9 else 4
                        empty_tiles -= 1
                        break
                    index_tmp -= 1
            
            # If the board has no empty slots, check for viable moves in every direction.
            if empty_tiles == 0:
                for direction in range(1, 5):
                    if self.check_viable_move(direction):
                        break # A viable move exists
                else:
                    return True, (i, self.game_board[i]) # The game is lost
                
        return False, (i, self.game_board[i])
                
    def make_move(self, direction):
        """
        up:    1
        right: 2
        down:  3
        left:  4
        
        Returns tuble of viableMove, gameLost, moves_made, new_tile
        """
        if not self.check_viable_move(direction):
            return False, False, [], -1

        check_dict = {
            #  [row_iter, col_iter, isVert, isPos]
            1: [(0, 4), (0, 4), True, False],
            2: [(0, 4), (3, -1, -1), False, True],
            3: [(3, -1, -1), (0, 4), True, True],
            4: [(0, 4), (0, 4), False, False],
        }
        
        move_info = check_dict[direction]
        moves_made = []
        
        for m in range(*move_info[0]):
            for n in range(*move_info[1]):
                coords = m, n
                if self[coords] is None:
                    continue
                
                tile_move = self._move_tile(coords,
                                            isVert=move_info[2],
                                            isPos=move_info[3])
                if tile_move[3] == False:
                    moves_made.append(tile_move)
        
        self.merged = [False] * 16
        gameLost, new_tile = self._spawn_tile()
        return True, gameLost, moves_made, new_tile


def main():
    
    move_dict = {'quit': 0,
                 'q': 0,
                 'w': 1,
                 'd': 2,
                 's': 3,
                 'a': 4,
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
        game_instance = GameState()
        print(game_instance)
        while not gameLost:
            move = get_move()
            if not move:
                break
            viableMove, gameLost, _, _ = game_instance.make_move(move)
            if viableMove:
                print(game_instance)
    
        print(f"    You Lost\n\nFinal Score: {game_instance.score}\n\nPlay Again? (y/n)")
        if input().lower() == 'n':
            break
    print('Thanks for playing.')
        
            
if __name__ == '__main__':
    main()