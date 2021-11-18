"""

"""

import random as rd


class GameState():
    
    def __init__(self, game_board=[None] * 16):
        self.game_board = game_board
        self.score = 0
        self.merged = [False] * 16
        
        if not any(game_board):
            self._spawn_tile(2)
        
    def __str__(self):
        board_string = ''
        col = 0
        for tile in self.game_board:
            value = '_' if tile is None else tile
            board_string += f"{value:^4}"
            col += 1
            if col % 4 == 0:
                board_string += '\n'
        return board_string
    
    def get_value(self, coords):
        return self.game_board[self._to_index(coords)]
    
    def set_value(self, coords, value):
        self.game_board[self._to_index(coords)] = value
        
    def _to_index(self, coords):
        return coords[0] * 4 + coords[1]
                            
    def _merge(self, coords, new_coords):
        self.set_value(new_coords, 2 * self.get_value(new_coords))
        self.set_value(coords, None), 
        self.score += self.get_value(new_coords)
        self.merged[self._to_index(new_coords)] = True
                
    def _move_tile(self, *coords, isVert=True, isPos=True):
        """ Move a tile as far as possible in the indicated direction, merge 
        with other tile if they have the same value.
        Return a tuble of initial coordinates, new coordinates, and whether it 
        has been merged.
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
        new_coords = increment(*coords)
        
        while (0 <= new_coords[0] < 4 and
               0 <= new_coords[1] < 4 and
               self.get_value(new_coords) is None):
            
            self.set_value(new_coords, self.get_value(current_coords))
            self.set_value(current_coords, None)
            
            current_coords, new_coords = new_coords, increment(*new_coords)
            self.__str__()

        if (0 <= new_coords[0] < 4 
            and 0 <= new_coords[1] < 4):
            
            if (self.get_value(current_coords) == self.get_value(new_coords)
                and self.merged[self._to_index(new_coords)] == False):
                self._merge(current_coords, new_coords)
                merged = True
                
        return [coords, current_coords, merged, coords == current_coords]
    
    def check_viable_move(self, direction):
        """
        up:    1
        right: 2
        down:  3
        left:  4
        """

        horz = lambda x, y: (x, y)
        vert = lambda x, y: (y, x)
        check_dict = {1: [(0, 4), (3, -1, -1), vert],
                      2: [(0, 4), (0, 4), horz],
                      3: [(0, 4), (0, 4), vert],
                      4: [(0, 4), (3, -1, -1), horz],
                      }
        
        for m in range(*check_dict[direction][0]):
            numFound, last_num = False, None
            
            for n in range(*check_dict[direction][1]):
                value = self.get_value(check_dict[direction][2](m, n))
                
                if value is not None:
                    if value == last_num:
                        return True
                    numFound, last_num = True, value
                    
                elif numFound == True:
                    return True
                
        return False
    
    def _spawn_tile(self, repeat=1):
        for _ in range(repeat):
            spawned = False
            while not spawned:
                index = rd.randint(0, 15)
                if self.game_board[index] == None:
                    self.game_board[index] = 2 if rd.random() < 0.9 else 4
                    spawned = True
                    
            if not None in self.game_board:
                for direction in range(1, 5):
                    if self.check_viable_move(direction):
                        break
                else:
                    return True, (0, 0)
                
        return False, index
                
    def make_move(self, direction):
        """
        up:    1
        right: 2
        down:  3
        left:  4
        
        Returns tuble of viableMove, gameLost, moves_made
        """
        if not self.check_viable_move(direction):
            return False, False, [], -1

        horz = lambda x, y: (y, x)
        vert = lambda x, y: (x, y)
        check_dict = {1: [(0, 4), (0, 4), vert, True, False],
                      2: [(3, -1, -1), (0, 4), horz, False, True],
                      3: [(3, -1, -1), (0, 4), vert, True, True],
                      4: [(0, 4), (0, 4), horz, False, False],
                      }
        
        moves_made = []
        for m in range(*check_dict[direction][0]):
            for n in range(*check_dict[direction][1]):
                if self.get_value(check_dict[direction][2](m, n)) is None:
                    continue
                moves_made.append(self._move_tile(*check_dict[direction][2](m, n),
                                                  isVert=check_dict[direction][3],
                                                  isPos=check_dict[direction][4]))
        
        self.merged = [False] * 16
        gameLost, new_tile = self._spawn_tile()
        return not gameLost, gameLost, moves_made, new_tile


def main():
    
    move_dict = {'quit': 0,
                 'w': 1,
                 'd': 2,
                 's': 3,
                 'a': 4,
                 }
    
    def get_move():
        move = None
        while move is None:
            try:
                move = move_dict[input()]
            except KeyError:
                continue
        return move
    
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