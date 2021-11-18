import pygame
from pygame.locals import *
from Game2048 import GameState
import myStorage
from GUICreator import myGUI, Text

BASE_UNIT = 24
TILE_DIM = 4.875
DEBUG = False

#class Tile(Text):
    
    #def __init__(self, **kwargs):
        #self.position = kwargs.get('position', (0, 0)) # The location of the tile in terms of (row, col)
        #self.source = self.position                    # Where the tile came from. Only different to position
        #self.inPlace = kwargs.get('inPlace', True)     # Whether the coordinates of the tile matches its location
        #self.toMerge = kwargs.get('toMerge', False)    # Whether a tile is going to merge into another tile and be deleted
        
        #super().__init__(**kwargs)
        
    #def get_target_coordinates(self):
        #return self.location_to_coordinates(self.position)
    
    #def location_to_coordinates(self, location):
        #x = (location[0] * (TILE_DIM + 0.5) + 0.5) * BASE_UNIT
        #y = (location[1] * (TILE_DIM + 0.5) + 0.5) * BASE_UNIT
        #return x, y
    
    #def move_to_target(self):
        #self.coordinates = self.get_target_coordinates()
        #self.source = self.position
        #self.inPlace = True
        

class Board(myGUI):
    
    def __init__(self):
        if not pygame.font:
            raise ImportError("Fonts not imported")
    
        window_size = win_width, win_height = (BASE_UNIT * 24,
                                               BASE_UNIT * 32)
        caption = "2048"
        bg_color = 230, 220, 205
        colours = {
            'bg_colour': (230, 220, 205),
            'title_colour': (238, 201, 0),
            'board_colour': (105, 95, 80),
            'tile_colour': (134, 122, 102),
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'offwhite': (240, 240, 240),
        }        
        
        super().__init__(window_size, caption, bg_color, colour_palette=colours)
        
        # Add GUI elements
        block_info = {
            'parent': self.window,
            'dimensions': (24 * BASE_UNIT, 8 * BASE_UNIT),
            'coordinates': (0 * BASE_UNIT, 0 * BASE_UNIT),
            'colour': 'bg_colour'
        }    
        header_block = self.create_block(**block_info)

        block_info = {
            'parent': self.window,
            'dimensions': (24 * BASE_UNIT, 24 * BASE_UNIT),
            'coordinates': (0 * BASE_UNIT, 8 * BASE_UNIT),
            'colour': 'bg_colour'
        }    
        board_block = self.create_block(**block_info)

        block_info = {
            'parent': header_block,
            'dimensions': (6 * BASE_UNIT, 6 * BASE_UNIT),
            'alignx': 'left',
            'aligny': 'centre',
            'marginx': BASE_UNIT,
            'colour': (238, 201, 0),
            'text_value': '2048',
            'text_colour': 'white',
            'text_size': 50
        }    
        title_block = self.create_text(**block_info)

        block_info = {
            'parent': header_block,
            'dimensions': (6 * BASE_UNIT, 6 * BASE_UNIT),
            'alignx': 'centre',
            'aligny': 'centre',
            'colour': (165, 150, 127),
            'text_value': '0',
            'text_size': 40
        }
        score_block = self.create_text(**block_info)

        block_info = {
            'parent': header_block,
            'dimensions': (6 * BASE_UNIT, 6 * BASE_UNIT),
            'alignx': 'right',
            'aligny': 'centre',
            'marginx': BASE_UNIT,
            'colour': (165, 150, 127),
            'text_value': '0',
            'text_size': 40
        }    
        high_score_block = self.create_text(**block_info)

        block_info = {
            'parent': board_block,
            'dimensions': (22 * BASE_UNIT, 22 * BASE_UNIT),
            'alignx': 'centre',
            'aligny': 'centre',
            'colour': (105, 95, 80)
        }    
        tile_board = self.create_block(**block_info)
        
        block_info = {
            'parent': tile_board,
            'dimensions': (TILE_DIM * BASE_UNIT, TILE_DIM * BASE_UNIT),
            'colour': (134, 122, 102)
        }        
        for i in range(16):
            block_info['coordinates'] = ((0.5 + (i // 4) * (TILE_DIM + 0.5)) * BASE_UNIT,
                                         (0.5 + (i % 4) * (TILE_DIM + 0.5)) * BASE_UNIT)
            self.create_block(**block_info)
                                             
        self.header_block = header_block
        self.board_block = board_block
        self.title_block = title_block
        self.score_block = score_block
        self.high_score_block = high_score_block
        self.tile_board = tile_board
        
        # Initiate Game
        self.colour_key = {
            2: (255, 240, 232),
            4: (244, 218, 181),
            8: (255, 181, 117),
            16: (237, 131, 61),
            32: (255, 83, 40),
            64: (255, 55, 0),
            128: (255, 161, 0),
            256: (137, 196, 0),
            512: (0, 155, 49),
            1024: (0, 160, 144),
            2048: (0, 84, 153),
            4096:(0, 23, 201),
            8192: (103, 0, 201),
            16384: (138, 0, 142),
            32768: (112, 0, 0),
            65536: (10, 10, 10),
        }
            
        self.loop_functions.extend([self.update_score,
                                    self.make_moves,
                                    self.update_values])
        self.init_game()
        
    def init_game(self):
        # If this is not the first game then any existing tiles need to be removed
        try:
            for tile in self.tiles.values():
                tile.delete()
        except AttributeError:
            pass
        
        self.game_inst = GameState()
        self.tiles = {}                 # A dictionary used to track the game tiles. 
                                        #    Keys are the (row, col) of the tile and the tile object are the values
        self.tile_queue = []            # A queue of locations (row, col) to spawn a tile at.
        self.move_queue = []            # A queue of list of tile moves, each items corresponds to a tile in the tile_queue
        self.merge_queue = []           # A list of tiles which need their values doubled due to a merge
        self.gameOver = False
        
        for index, tile_value in enumerate(self.game_inst.game_board):
            if tile_value is not None:
                self.spawn_tile((index, tile_value))
                
        if not DEBUG:
            print(self.game_inst)
        
        
    def update_score(self):
        score = self.game_inst.score
        self.score_block.set_text(str(score))
        
    def keyboard_event_handler(self, event):
        if DEBUG:        
            print(self.game_inst)
            
        move_dict = {
            K_UP: 1,
            K_RIGHT: 2,
            K_DOWN: 3,
            K_LEFT: 4,
        }
        
        if self.gameOver and event.key == K_RETURN:
            self.init_game()
            return
        
        try:
            direction = move_dict[event.key]
        except KeyError:
            return
        move_info = self.game_inst.make_move(direction)
            
        viableMove, gameOver, moves_made, new_tile = move_info
        if gameOver:
            self.gameOver = True
        
        if viableMove:
            self.tile_queue.append(new_tile)
            self.move_queue.append(moves_made)
            
            for move in moves_made:
                original_coords, dest_coords, toMerge, inPlace = move
                
                # Ensuring some basic assumptions about moves
                assert inPlace is False
                assert original_coords != dest_coords
                assert (original_coords[0] == dest_coords[0] or
                        original_coords[1] == dest_coords[1])
                
                # Convert inPlace bool to move counter
                move[3] = int(inPlace)
            
        if not DEBUG:        
            print(self.game_inst)
            
    
    def spawn_tile(self, tile_info):     
        
        index, value = tile_info
        row, col = index // 4, index % 4
        
        block_info = {
            'parent': self.tile_board,
            'dimensions': (TILE_DIM * BASE_UNIT, TILE_DIM * BASE_UNIT),
            'coordinates': ((0.5 + col * (TILE_DIM + 0.5)) * BASE_UNIT,
                            (0.5 + row * (TILE_DIM + 0.5)) * BASE_UNIT),      
            'colour': self.colour_key[value],
            'text_value': str(value),
            'text_size': 60,
            'priority': 2
        }          
        new_tile = self.create_text(**block_info)
        self.tiles[(row, col)] = new_tile
            
    def update_values(self):
        for coords in self.merge_queue:
            new_value = 2 * int(self.tiles[coords].get_text())
            new_colour = self.colour_key[new_value]
            
            self.tiles[coords].set_text(new_value)
            self.tiles[coords].set_colour(new_colour)
        self.merge_queue = []
                
    def delete_tile(self, original_coords, dest_coords):
        """
        Begins tile merge by deleting the tile from original_coords.
        """
        if DEBUG:
            print(f"Deleting tile from {original_coords} at {dest_coords}.")
            
        old_tile = self.tiles[original_coords]
        del self.tiles[original_coords]      # Remove tile from the tile dictionary
        old_tile.delete()               # Delete tile from GUI
        
        self.merge_queue.append(dest_coords)
                
    def make_moves(self):
        """
        If there is at least one move in the move_queue, implement one step of it. If that move is completed,
        remove it from the queue and spawn its corresponding tile.
        """
        try:
            moves_made = self.move_queue[0]
        except IndexError:
            return
        
        move_fraction = 4
        moveCompleted = False
        find_x = lambda coords: (coords[1] * (TILE_DIM + 0.5) + 0.5) * BASE_UNIT
        find_y = lambda coords: (coords[0] * (TILE_DIM + 0.5) + 0.5) * BASE_UNIT
        
        if DEBUG:
            for move in moves_made:
                print(move)
            print()
        
        for move in moves_made:
            
            original_coords, dest_coords, toMerge, move_progress = move
            
            if move_progress < move_fraction:
                if move_progress == 0 and toMerge:
                    self.tiles[original_coords].priority = 1
                    
                move_progress += 1
                    
                # The starting x/y coordinates of the tile
                original_x = find_x(original_coords)
                original_y = find_y(original_coords)

                # The target x/y coordinates of the tile                
                dest_x = find_x(dest_coords)
                dest_y = find_y(dest_coords)

                # The amount to change the coordinates of the tile by at each step
                del_x = (dest_x - original_x) / move_fraction
                del_y = (dest_y - original_y) / move_fraction
                
                # If the tile hasn't move enough steps, take one step
                if move_progress < move_fraction:
                    self.tiles[original_coords].move(del_x = int(del_x),
                                                     del_y = int(del_y))
                
                # Otherwise, move to exactly the destination
                else:                    
                    self.tiles[original_coords].move(x = dest_x,
                                                     y = dest_y)
                    
                move[3] = move_progress # Update the step count for the move
            
            # Once a tile has been move to the correct place on the GUI, it either needs
            #    its (row,col) coordinates updated in self.tiles if it is not merging,
            #    or to be deleted from self.tiles if it is merging
            else:     
                moveCompleted = True
                if DEBUG:
                    print(f"Processing tile {original_coords}")
                    
                if toMerge:
                    self.delete_tile(original_coords, dest_coords)
                
                else:
                    if DEBUG:
                        print(f"Reassigning tile {original_coords} as tile {dest_coords}")
                    assert self.tiles.get(dest_coords, None) is None, f"Tile {original_coords} is attempting to overwrite tile {dest_coords}."
                        
                    tile = self.tiles[original_coords]
                    del self.tiles[original_coords]
                    self.tiles[dest_coords] = tile
        
        # Once a move is completed for all tiles, remove the move from the queue,
        #    then remove the new tile from the tile queue and spaen it.
        #    This is the only place items are removed from either queue and
        #    keeps them synchronous.
        if moveCompleted:
            self.move_queue.pop(0)
            new_tile = self.tile_queue.pop(0)
            self.spawn_tile(new_tile)
        
            

def main():
    
    pygame.init()
    
    game_gui = Board()
    game_gui.run_GUI()    
    
if __name__ == '__main__':
    main()