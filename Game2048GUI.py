import pygame
from pygame.locals import *
from Game2048 import GameState as GS
import myStorage

BASE_UNIT = 24
TILE_DIM = 4.875
pygame.init()


class Tile(pygame.sprite.Sprite):
    def __init__(self, value, row, col):
        
        self.colour_key = {2: (255, 240, 232),
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
        
        self.value = value
        self.row, self.col = row, col
        colour = self.colour_key[value]
        
        pygame.sprite.Sprite.__init__(self)
        
        tile_size = TILE_DIM * BASE_UNIT, TILE_DIM * BASE_UNIT
        self.image = pygame.Surface(tile_size)

        self.update_image()
 
        self.rect = self.image.get_rect()
        self.position_rect()
        
    def __repr__(self):
        return f"<Tile; {self.value}, ({self.row}, {self.col})>"
    
    def position_rect(self):
        self.x_pos = (1.5 + (TILE_DIM + 0.5) * self.col) * BASE_UNIT
        self.y_pos = (1.5 + (TILE_DIM + 0.5) * self.row) * BASE_UNIT + 8 * BASE_UNIT
        self.rect.move_ip(self.x_pos, self.y_pos)
        
    def update_pos(self):
        self.row = ((self.y_pos - 8 * BASE_UNIT) / BASE_UNIT - 1.5) / (TILE_DIM + 0.5)
        self.col = (self.x_pos / BASE_UNIT - 1.5) / (TILE_DIM + 0.5)
        
    def move_tile(self, direction):
        if direction <= 2:
            mult = 1
        else:
            mult = -1
        move_distance = mult * (TILE_DIM + 0.5) * BASE_UNIT / 16
        
            
        if direction % 2 == 0:
            self.x_pos += move_distance
            self.rect.move_ip(move_distance, 0)
        else:
            self.y_pos -= move_distance
            self.rect.move_ip(0, -1 * move_distance)
            
        self.update_pos()
    
    def update_image(self):
        self.colour = self.colour_key[self.value]
        self.image.fill(self.colour)
        
        title_font = pygame.font.Font(None, 60)
        text = title_font.render(str(self.value), 1, (0, 0, 0))
        textpos = (text.get_rect(centerx=self.image.get_width()/2,
                                 centery=self.image.get_height()/2))
        self.image.blit(text, textpos)        
        
    def _adjacent_tile(self, direction):
        """
        direction = 1 -> -1, 0
        direction = 2 -> 0, 1
        direction = 3 -> 1, 0
        direction = 4 -> 0, -1
        """
        sign = -1 if direction in [1, 4] else 1
        row_mod = direction % 2
        col_mod = (direction - 1) % 2
        row = self.row + sign * row_mod
        col = self.col + sign * col_mod
        return row, col
        
    def merge(self, tiles_pos, direction):
        merge_into = tiles_pos[self._adjacent_tile(direction)]
        merge_into.value *= 2
        merge_into.update_image()
        self.kill()
        

class Board():
    
    def __init__(self, window):
        if not pygame.font:
            raise ImportError("Fonts not imported")
        
        bg_color = 230, 220, 205
        self.block_list = myStorage.MyList(window)
        self.window = window
                
        header_block = self.create_block(24, 8, bg_color, window, 0, 0)
        self.header_block = header_block
        board_block = self.create_block(24, 24, bg_color, window, 0, 8)
        
        title_block = self.create_block(6, 6, (238, 201, 0), header_block, 1, 1)
        score_block = self.create_block(6, 6, (165, 150, 127), header_block, 9, 1)
        high_score_block = self.create_block(6, 6, (165, 150, 127), header_block, 16, 1)
        board = self.create_block(22, 22, (105, 95, 80), board_block, 1, 1)
        
        tile_dict = {}
        for i in range(16):
            tile_dict[i] = self.create_block(TILE_DIM, TILE_DIM, (134, 122, 102), board,
                                             0.5 + (i // 4) * (TILE_DIM + 0.5),
                                             0.5 + (i % 4) * (TILE_DIM + 0.5))
        
        
        self.score_block = score_block
        self.update_score()
        
        title_font = pygame.font.Font(None, 50)
        text = title_font.render("2048", 1, (250, 250, 250))
        textpos = (text.get_rect(centerx=title_block.get_width()/2,
                                 centery=title_block.get_height()/2))
        self.block_list.append(text, title_block, textpos)
        
        self.draw_background()
        
    def create_block(self, width, height, colour, parent, x_pos, y_pos):
        block_size = (width * BASE_UNIT, height * BASE_UNIT)
        block = pygame.Surface(block_size)
        block = block.convert()
        block.fill(colour)
        self.block_list.append(block, parent, (x_pos * BASE_UNIT, y_pos * BASE_UNIT))
        return block
    
    def update_score(self, score=0):
        """Score not currently displaying. Reason unknown.
        """
        try:
            del self.block_list[self.score_text]
            del self.block_list[self.score_block]
        except (KeyError, AttributeError):
            pass    
        score_block = self.create_block(6, 6, (165, 150, 127), self.header_block, 9, 1)
        self.score_block = score_block
        self.score = score
        score_font = pygame.font.Font(None, 40)
        score_text = score_font.render(f"{self.score}", 1, (10, 10, 10))
        score_pos = (score_text.get_rect(centerx=score_block.get_width()/2,
                                 centery=score_block.get_height()*3/4))
        self.block_list.append(score_text, score_block, score_pos)
        self.score_text = score_text
        print(score)
    
    def draw_background(self, root_block=None):
        if root_block == None:
            root_block = self.window
        #print(self.block_list[self.score_block][2])
        print('.', end=' ')
        for block, parent, position in reversed(self.block_list.section(root_block)):
            parent.blit(block, position)
        
def draw_board(draw_objects):
    window, game_board, tile_group = draw_objects
    game_board.draw_background()
    tile_group.draw(window)
    pygame.display.flip()
    
def spawn_tile(tile_group, game_inst, index, tiles_pos):
    tile_value = game_inst.game_board[index]
    row, col = index // 4, index % 4
    new_tile = Tile(tile_value, row, col)
    tile_group.add(new_tile)
    tiles_pos[(row, col)] = new_tile
    
def shift_tiles(draw_objects, moves_made, tiles_pos, direction, debug=False):
    
    if debug:
        [print(x) for x in moves_made]
        print()
        [print(x, y) for x, y in tiles_pos.items()]
        print()

    moving = True
    while moving:
        moving = False
        for i, move in enumerate(moves_made):
            original_pos, dest, toMerge, inPlace = move
            if not inPlace or toMerge:
                try:
                    tile = tiles_pos[original_pos]
                except KeyError:
                    if debug:
                        print(move)
                        print()
                        [print(x, y) for x, y in tiles_pos.items()]
                        print()
                    raise KeyError(original_pos)
                    
                if not inPlace:
                    inPlace = dest == (tile.row, tile.col)
                    
                    if not inPlace:
                        moving = True
                        tile.move_tile(direction)
                    else: # Update tiles_pos with new position
                        del tiles_pos[original_pos]
                        tiles_pos[dest] = tile
                        moves_made[i][0] = dest
                        moves_made[i][3] = True
                        
                if inPlace and toMerge:
                    del tiles_pos[move[1]]
                    try:
                        tile.merge(tiles_pos, direction)
                    except KeyError:
                        if debug:
                            print(move)
                        raise KeyError(original_pos)
                    moves_made[i][2] = False
            

def main():
    window_size = win_width, win_height = (BASE_UNIT * 24,
                                           BASE_UNIT * 32)    
    window = pygame.display.set_mode(window_size)
    pygame.display.set_caption("2048")
    #[2 ** x for x in range(1, 17)]
    #[2, 2, 4, 8, 128, 64, 32, 16, 256, 512, 1024, 2048, 32768, 16384, 8192, 4096]
    game_inst = GS()
    game_gui = Board(window)
    tile_group = pygame.sprite.Group()
    tiles_pos = {}
    move_dict = {'quit': 0,
                 'up': 1,
                 'right': 2,
                 'down': 3,
                 'left': 4,
                 }
    
    print(game_inst)
    for index, tile_value in enumerate(game_inst.game_board):
        if tile_value is not None:
            spawn_tile(tile_group, game_inst, index, tiles_pos)
    
    running = True
    
    draw_objects = window, game_gui, tile_group
    draw_board(draw_objects)

    pygame.key.set_repeat(1000, 100)
    while running:
        pygame.time.delay(50)
        for event in pygame.event.get():
            if event.type == KEYDOWN and event.key == K_ESCAPE:
                running = False
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        gameOver = False
        
        if not gameOver:
            direction = None
            move_info = False, False, [], -1
            if keys[pygame.K_UP]:
                direction = move_dict['up']
                move_info = game_inst.make_move(direction)
            if keys[pygame.K_RIGHT]:
                direction = move_dict['right']
                move_info = game_inst.make_move(direction)
            if keys[pygame.K_DOWN]:
                direction = move_dict['down']
                move_info = game_inst.make_move(direction)
            if keys[pygame.K_LEFT]:
                direction = move_dict['left']
                move_info = game_inst.make_move(direction)
                
            viableMove, gameOver, moves_made, new_tile = move_info
            
            if viableMove:
                shift_tiles(draw_objects, moves_made, tiles_pos, direction)
                
                draw_board(draw_objects)
            
                pygame.time.delay(50)                
                
                spawn_tile(tile_group, game_inst, new_tile, tiles_pos)
                game_gui.update_score(game_inst.score)
                draw_board(draw_objects)
                
                print(game_inst)
        else:
            print(f"Game Over\n\nScore: {score}")
            pass
        
    pygame.quit()

    
if __name__ == '__main__':
    main()