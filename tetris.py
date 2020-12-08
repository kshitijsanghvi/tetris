import random
import cv2
import numpy as np
from PIL import Image
from time import sleep

# Tetris game class
class Tetris:

    '''Tetris game class'''

    # BOARD
    MAP_EMPTY = 0
    MAP_BLOCK = 1
    MAP_PLAYER = 2
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20

    TETROMINOS = {
        0: { # I
            0: np.array( [(0,0), (1,0), (2,0), (3,0)] ),
            90: np.array( [(1,0), (1,1), (1,2), (1,3)] ),
            180: np.array( [(3,0), (2,0), (1,0), (0,0)] ),
            270: np.array( [(1,3), (1,2), (1,1), (1,0)] ),
        },
        1: { # T
            0: np.array( [(1,0), (0,1), (1,1), (2,1)] ),
            90: np.array( [(0,1), (1,2), (1,1), (1,0)] ),
            180: np.array( [(1,2), (2,1), (1,1), (0,1)] ),
            270: np.array( [(2,1), (1,0), (1,1), (1,2)] ),
        },
        2: { # L
            0: np.array( [(1,0), (1,1), (1,2), (2,2)] ),
            90: np.array( [(0,1), (1,1), (2,1), (2,0)] ),
            180: np.array( [(1,2), (1,1), (1,0), (0,0)] ),
            270: np.array( [(2,1), (1,1), (0,1), (0,2)] ),
        },
        3: { # J
            0: np.array( [(1,0), (1,1), (1,2), (0,2)] ),
            90: np.array( [(0,1), (1,1), (2,1), (2,2)] ),
            180: np.array( [(1,2), (1,1), (1,0), (2,0)] ),
            270: np.array( [(2,1), (1,1), (0,1), (0,0)] ),
        },
        4: { # Z
            0: np.array( [(0,0), (1,0), (1,1), (2,1)] ),
            90: np.array( [(0,2), (0,1), (1,1), (1,0)] ),
            180: np.array( [(2,1), (1,1), (1,0), (0,0)] ),
            270: np.array( [(1,0), (1,1), (0,1), (0,2)] ),
        },
        5: { # S
            0: np.array( [(2,0), (1,0), (1,1), (0,1)] ),
            90: np.array( [(0,0), (0,1), (1,1), (1,2)] ),
            180: np.array( [(0,1), (1,1), (1,0), (2,0)] ),
            270: np.array( [(1,2), (1,1), (0,1), (0,0)] ),
        },
        6: { # O
            0: np.array( [(1,0), (2,0), (1,1), (2,1)] ),
            90: np.array( [(1,0), (2,0), (1,1), (2,1)] ),
            180: np.array( [(1,0), (2,0), (1,1), (2,1)] ),
            270: np.array( [(1,0), (2,0), (1,1), (2,1)] ),
        }
    }
    
    TETROMINOS_MIN_Y = {
        0: { # I
            0: np.array( [1,1,1,1] ),
            90: np.array( [4] ),
            180: np.array( [1,1,1,1] ),
            270: np.array( [4] ),
        },
        1: { # T
            0: np.array( [2,2,2] ),
            90: np.array( [2,3] ),
            180: np.array( [2,3,2] ),
            270: np.array( [3,2] ),
        },
        2: { # L
            0: np.array( [3,3] ),
            90: np.array( [2,2,2] ),
            180: np.array( [1,3] ),
            270: np.array( [3,2,2] ),
        },
        3: { # J
            0: np.array( [3,3] ),
            90: np.array( [2,2,3] ),
            180: np.array( [3,1] ),
            270: np.array( [2,2,2] ),
        },
        4: { # Z
            0: np.array( [1,2,2] ),
            90: np.array( [3,2] ),
            180: np.array( [1,2,2] ),
            270: np.array( [3,2] ),
        },
        5: { # S
            0: np.array( [2,2,1] ),
            90: np.array( [2,3] ),
            180: np.array( [2,2,1] ),
            270: np.array( [2,3] ),
        },
        6: { # O
            0: np.array( [2,2] ),
            90: np.array( [2,2] ),
            180: np.array( [2,2] ),
            270: np.array( [2,2] ),
        }
    }
    
    
    COLORS = {
        0: (255, 255, 255),
        1: (247, 64, 99),
        2: (0, 167, 247),
    }


    def __init__(self):
        self.reset()

    
    def reset(self):
        '''Resets the game, returning the current state'''
        self.board = np.zeros( (Tetris.BOARD_HEIGHT,Tetris.BOARD_WIDTH), dtype=int )
        self.column_heights = np.zeros(Tetris.BOARD_WIDTH, dtype=int)
        self.game_over = False
        self.bag = list(range(len(Tetris.TETROMINOS)))
        random.shuffle(self.bag)
        self.next_piece = self.bag.pop()
        self._new_round()
        self.score = 0
        return self._get_board_props(self.board, self.column_heights)


    def _get_rotated_piece(self):
        '''Returns the current piece, including rotation'''
        return Tetris.TETROMINOS[self.current_piece][self.current_rotation]


    def _get_complete_board(self):
        '''Returns the complete board, including the current piece'''
        piece = self._get_rotated_piece()
        piece = piece + self.current_pos
        board = self.board.copy()
        board[ piece[:,1], piece[:,0] ] = Tetris.MAP_PLAYER
        
        return board


    def get_game_score(self):
        '''Returns the current game score.

        Each block placed counts as one.
        For lines cleared, it is used BOARD_WIDTH * lines_cleared ^ 2.
        '''
        return self.score
    

    def _new_round(self):
        '''Starts a new round (new piece)'''
        # Generate new bag with the pieces
        if len(self.bag) == 0:
            self.bag = list(range(len(Tetris.TETROMINOS)))
            random.shuffle(self.bag)
        
        self.current_piece = self.next_piece
        self.next_piece = self.bag.pop()
        self.current_pos = [3, 0]
        self.current_rotation = 0

        min_x = self._get_rotated_piece()[:,0].min()
        max_x = self._get_rotated_piece()[:,0].max()
        min_y = Tetris.TETROMINOS_MIN_Y[self.current_piece][0]
        if (Tetris.BOARD_HEIGHT - ( self.column_heights[ self.current_pos[0]+min_x : self.current_pos[0]+max_x+1 ] + min_y ).max()) < 0:
            self.game_over = True

    def _check_collision(self, piece, pos):
        '''Check if there is a collision between the current piece and the board'''
        for x, y in piece:
            x += pos[0]
            y += pos[1]
            if x < 0 or x >= Tetris.BOARD_WIDTH \
                    or y < 0 or y >= Tetris.BOARD_HEIGHT \
                    or self.board[y][x] == Tetris.MAP_BLOCK:
                return True
        return False

    def _rotate(self, angle):
        '''Change the current rotation'''
        self.current_rotation = self.current_rotation + angle % 360


    def _add_piece_to_board(self, piece, pos):
        '''Place a piece in the board, returning the resulting board'''
        board = self.board.copy()
        board[piece[:,1] + pos[1], piece[:,0] + pos[0]] = Tetris.MAP_BLOCK
        return board


    def _clear_lines(self, board):
        '''Clears completed lines in a board'''
        # Check if lines can be cleared
        
        lines_to_not_clear = np.where(board.sum(1) < Tetris.BOARD_WIDTH)[0]

        new_board = np.zeros((Tetris.BOARD_HEIGHT,Tetris.BOARD_WIDTH), dtype=int)
        new_board[-len(lines_to_not_clear):] = board[lines_to_not_clear]
        
        return Tetris.BOARD_HEIGHT-len(lines_to_not_clear), new_board


    def _number_of_holes(self, board, column_heights):
        '''Number of holes in the board (empty sqquare with at least one block above it)'''
        holes = 0
        return sum(column_heights) - board.sum()


    def _bumpiness(self, board, column_heights):
        '''Sum of the differences of heights between pair of columns'''
        bumpiness = np.abs(column_heights[:-1] - column_heights[1:])
        total_bumpiness = sum(bumpiness)
        max_bumpiness = max(bumpiness)
        
        return total_bumpiness, max_bumpiness


    def _height(self, board, column_heights):
        '''Sum and maximum height of the board'''
        sum_height = sum(column_heights)
        max_height = max(column_heights)
        min_height = min(column_heights)

        return sum_height, max_height, min_height


    def _column_heights(self,board):
        # assert Tetris.MAP_BLOCK == 1
        
        return np.sum( np.cumsum(board,axis=0) >= 1, axis=0 )
        # return np.sum( np.sign(np.cumsum(board,axis=0)), axis=0 )
    
    
    def _get_board_props(self, board, column_heights):
        '''Get properties of the board'''
        lines, board = self._clear_lines(board)
        column_heights -= lines
        holes = self._number_of_holes(board, column_heights)
        total_bumpiness, max_bumpiness = self._bumpiness(board, column_heights)
        sum_height, max_height, min_height = self._height(board, column_heights)
        return [lines, holes, total_bumpiness, sum_height]
        
    def get_state_size(self):
        '''Size of the state'''
        return 4


    def set_board(self, board):
        self.board = board
        self.column_heights = self._column_heights(board)

    def get_next_states(self):
        '''Get all possible next states'''
        states = {}
        piece_id = self.current_piece
        column_heights = self.column_heights
        
        if piece_id == 6: 
            rotations = [0]
        elif piece_id in [0,4,5]:
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]

        # For all rotations
        for rotation in rotations:
            piece = Tetris.TETROMINOS[piece_id][rotation]
            min_y = Tetris.TETROMINOS_MIN_Y[piece_id][rotation]
            
            min_x = piece[:,0].min()
            max_x = piece[:,0].max()
            
            # For all positions
            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                new_column_heights = column_heights.copy()
                new_column_heights[ x+min_x : x+max_x+1 ] += min_y
                y = Tetris.BOARD_HEIGHT - ( new_column_heights ).max()

                # Valid move
                if y >= 0:
                    board = self._add_piece_to_board(piece, [x,y])
                    states[(x, rotation)] = self._get_board_props(board, new_column_heights)

        return states



    def play(self, x, rotation, render=False, render_delay=None):
        '''Makes a play given a position and a rotation, returning the reward and if the game is over'''
        self.current_pos = [x, 0]
        self.current_rotation = rotation

        # Drop piece
        if render:
            while not self._check_collision(self._get_rotated_piece(), self.current_pos):
                self.render()
                if render_delay:
                    sleep(render_delay)
                self.current_pos[1] += 1
            self.current_pos[1] -= 1
        else:
            min_x = self._get_rotated_piece()[:,0].min()
            max_x = self._get_rotated_piece()[:,0].max()
            min_y = Tetris.TETROMINOS_MIN_Y[self.current_piece][rotation]
            y = Tetris.BOARD_HEIGHT - ( self.column_heights[ x+min_x : x+max_x+1 ] + min_y ).max()
            self.current_pos = [x,y]

        # Update board and calculate score        
        new_board = self._add_piece_to_board(self._get_rotated_piece(), self.current_pos)
        lines_cleared, new_board = self._clear_lines(new_board)
        self.set_board(new_board)
        score = 1 + (lines_cleared ** 2) * Tetris.BOARD_WIDTH
        self.score += score

        # Start new round
        self._new_round()
        if self.game_over:
            score -= 2

        return score, self.game_over


    def render(self):
        '''Renders the current board'''
        img = [Tetris.COLORS[p] for row in self._get_complete_board() for p in row]
        img = np.array(img).reshape(Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH, 3).astype(np.uint8)
        img = img[..., ::-1] # Convert RRG to BGR (used by cv2)
        img = Image.fromarray(img, 'RGB')
        img = img.resize((Tetris.BOARD_WIDTH * 25, Tetris.BOARD_HEIGHT * 25))
        img = np.array(img)
        cv2.putText(img, str(self.score), (22, 22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.imshow('image', np.array(img))
        cv2.waitKey(1)
