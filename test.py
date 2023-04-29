import asyncio
import argparse

import threading
import socket
import json
# from .. import game
import copy
from collections import deque
import time as time
import random

B = {'message': 'This is a Bad Move. Your new_position is unreachable',
    'state': {'players': ['Player_1', 'Player_2'],
                'current': 0,
                'positions': [45, 6], 
                'board': [{'N': False, 'E': True, 'S': True, 'W': False, 'item': None}, {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, {'N': False, 'E': True, 'S': True, 'W': True, 'item': 0}, {'N': False, 'E': False, 'S': True, 'W': True, 'item': 13}, {'N': False, 'E': True, 'S': True, 'W': True, 'item': 1}, {'N': False, 'E': True, 'S': False, 'W': True, 'item': None}, {'N': False, 'E': False, 'S': True, 'W': True, 'item': None}, {'N': True, 'E': False, 'S': False, 'W': True, 'item': 12}, {'N': False, 'E': False, 'S': True, 'W': True, 'item': None}, {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, {'N': False, 'E': True, 'S': False, 'W': True, 'item': None}, {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, {'N': True, 'E': True, 'S': True, 'W': False, 'item': 2}, {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, {'N': True, 'E': True, 'S': True, 'W': False, 'item': 3}, {'N': True, 'E': True, 'S': False, 'W': True, 'item': 19}, {'N': False, 'E': True, 'S': True, 'W': True, 'item': 4}, {'N': False, 'E': False, 'S': True, 'W': True, 'item': 16}, {'N': True, 'E': False, 'S': True, 'W': True, 'item': 5}, {'N': True, 'E': True, 'S': True, 'W': False, 'item': 20}, {'N': False, 'E': True, 'S': True, 'W': True, 'item': 23}, {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, {'N': False, 'E': True, 'S': True, 'W': False, 'item': None}, {'N': True, 'E': True, 'S': False, 'W': False, 'item': None}, {'N': False, 'E': False, 'S': True, 'W': True, 'item': 17}, {'N': False, 'E': True, 'S': True, 'W': False, 'item': None}, {'N': True, 'E': True, 'S': True, 'W': False, 'item': 6}, {'N': True, 'E': True, 'S': False, 'W': True, 'item': 21}, {'N': True, 'E': True, 'S': False, 'W': True, 'item': 7}, {'N': True, 'E': False, 'S': False, 'W': True, 'item': None}, {'N': True, 'E': False, 'S': True, 'W': True, 'item': 8}, {'N': False, 'E': False, 'S': True, 'W': True, 'item': None}, {'N': True, 'E': False, 'S': True, 'W': True, 'item': 9}, {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, {'N': True, 'E': True, 'S': False, 'W': True, 'item': 18}, {'N': True, 'E': True, 'S': False, 'W': False, 'item': 15}, {'N': True, 'E': True, 'S': False, 'W': False, 'item': 14}, {'N': True, 'E': False, 'S': True, 'W': True, 'item': 22}, {'N': False, 'E': True, 'S': True, 'W': False, 'item': None}, {'N': False, 'E': False, 'S': True, 'W': True, 'item': None}, {'N': True, 'E': True, 'S': False, 'W': False, 'item': None}, {'N': False, 'E': False, 'S': True, 'W': True, 'item': None}, {'N': True, 'E': True, 'S': False, 'W': True, 'item': 10}, {'N': False, 'E': True, 'S': True, 'W': False, 'item': None}, {'N': True, 'E': True, 'S': False, 'W': True, 'item': 11}, {'N': False, 'E': True, 'S': False, 'W': True, 'item': None}, {'N': True, 'E': False, 'S': False, 'W': True, 'item': None}],
                'tile': {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, 
                'target': 15, 
                'remaining': [3, 4]}, 
                'move': {'tile': {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, 
                            'gate': 'B', 
                            'new_position': 45}
        }

my_index = B['state']['current']
current_pos = B['state']['positions'][my_index]
next_pos = B['move']['new_position']
moves =  [46]
chosen_move =  46


board = B['state']['board']


# print("len : ", len(board))
print("current_pos :", "index = ",current_pos, ' ', board[current_pos])


print("next_pos :", "index = ",next_pos, ' ', board[next_pos])



#----------------------------

def successors(index, state):
    """Check all possible movements starting from the index of the tile."""
    # board = state["board"]
    res = []

    for direction in ["N", "S", "E", "W"]:
        print("current_tile : ", direction, " = ", board[current_pos][direction])
        if board[current_pos][direction]: #check if current tile is open in the asked direction
            coords = add(index2coords(current_pos), DIRECTIONS[direction]["coords"]) #coords of the next tile
            next_tile_index = coords2index(*coords)
            print("coord_valid : ", isCoordsValid(*coords))
            if isCoordsValid(*coords):
                next_tile = board[next_tile_index] #retrieve the NSWE of the next tile
                opposite_dir = DIRECTIONS[direction]["opposite"]

                print("next_tile : ",opposite_dir, " = ", next_tile[opposite_dir])
                if next_tile[opposite_dir]:
                    res.append(next_tile_index)
                    print(res)
                    print("added index : ",next_tile_index, "  ", board[next_tile_index])
        print('------------')

    return res




def next(state, move):
        """Create a copy of the board accounting for the changes inserting a tile in a gate makes."""
        new_state = copy.deepcopy(state)

        new_board, new_free = slideTiles(state["board"], move["tile"], move["gate"])
        new_state["board"] = new_board
        new_state["tile"] = new_free

        new_positions = []
        for position in state["positions"]:
            if onTrack(position, move["gate"]):
                if position == GATES[move["gate"]]["end"]:
                    new_positions.append(GATES[move["gate"]]["start"])
                    continue
                new_positions.append(position + GATES[move["gate"]]["inc"])
                continue
            new_positions.append(position)

        new_state["positions"] = new_positions

        return new_state









#-----------------LEVEL_2------------------- 
"""_______Operator functions_________"""


DIRECTIONS = {
    "N": {"coords": (-1, 0), "inc": -7, "opposite": "S"},
    "S": {"coords": (1, 0), "inc": 7, "opposite": "N"},
    "W": {"coords": (0, -1), "inc": -1, "opposite": "E"},
    "E": {"coords": (0, 1), "inc": 1, "opposite": "W"},
    (-1, 0): {"name": "N"},
    (1, 0): {"name": "S"},
    (0, -1): {"name": "W"},
    (0, 1): {"name": "E"},
}
GATES = {
    "A": {"start": 1, "end": 43, "inc": 7},
    "B": {"start": 3, "end": 45, "inc": 7},
    "C": {"start": 5, "end": 47, "inc": 7},
    "D": {"start": 13, "end": 7, "inc": -1},
    "E": {"start": 27, "end": 21, "inc": -1},
    "F": {"start": 41, "end": 35, "inc": -1},
    "G": {"start": 47, "end": 5, "inc": -7},
    "H": {"start": 45, "end": 3, "inc": -7},
    "I": {"start": 43, "end": 1, "inc": -7},
    "J": {"start": 35, "end": 41, "inc": 1},
    "K": {"start": 21, "end": 27, "inc": 1},
    "L": {"start": 7, "end": 13, "inc": 1},
}
def slideTiles(board, free, gate):
    start = GATES[gate]["start"]
    end = GATES[gate]["end"]
    inc = GATES[gate]["inc"]
    new_free = board[end]
    new_board = copy.deepcopy(board)
    dest = end
    src = end - inc
    while dest != start:
        new_board[dest] = new_board[src]
        dest = src
        src -= inc
    new_board[start] = free
    return new_board, new_free
def onTrack(index, gate):
    return index in range(
        GATES[gate]["start"],
        GATES[gate]["end"] + GATES[gate]["inc"],
        GATES[gate]["inc"],
    )
def turn_tile(tile):
    res = copy.deepcopy(tile)
    res["N"] = tile["E"]
    res["E"] = tile["S"]
    res["S"] = tile["W"]
    res["W"] = tile["N"]
    return res
def isSameTile(t1, t2):
    for _ in range(4):
        if t1 == t2:
            return True
        t2 = turn_tile(t2)
    return False
def random_turn_tile(tile):
    for _ in range(random.randint(1, 4)):
        tile = turn_tile(tile)
    return tile
def makeTiles():
    """creation of labyrinth"""
    tiles = []
    straight = {"N": True, "E": False, "S": True, "W": False, "item": None}
    corner = {"N": True, "E": True, "S": False, "W": False, "item": None}
    tee = {"N": True, "E": True, "S": True, "W": False, "item": None}
    for _ in range(12):
        tiles.append(random_turn_tile(straight))
    for _ in range(10):
        tiles.append(random_turn_tile(corner))
    treasure = 12
    for _ in range(6):
        tiles.append(random_turn_tile(dict(corner, item=treasure)))
        treasure += 1
    for _ in range(6):
        tiles.append(random_turn_tile(dict(tee, item=treasure)))
        treasure += 1
    random.shuffle(tiles)
    return tiles
def index2coords(index):
    return index // 7, index % 7
def coords2index(i, j):
    return i * 7 + j
def isCoordsValid(i, j):
    return i >= 0 and i < 7 and j >= 0 and i < 7
def add(A, B):
    return tuple(a + b for a, b in zip(A, B))


#--------------------



print(successors(current_pos, board))

