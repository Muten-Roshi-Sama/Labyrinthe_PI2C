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

#----------------------------
"""
-implement timeout at 3000ms
-debug fetch error when connecting to server



"""

#-----------------LEVEL_1------------------- 
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


#-------------LEVEL_2--------------------------
"""_______MY_Operator functions_________"""

def tile_finder(tile, state):
    board = state["board"]  # board is a LIST of dictionnaries
    return board.index(tile)
        
def target_finder(state):
    target_ID = state["target"]
    board = state["board"]
    for i in board:
        if i['item'] == target_ID:
            return board.index(i)
    print('Could not find target')

def possible_orientations(tile):
    """Generate all possible orientations of a given tile.{'N': False, 'E': True, 'S': True, 'W': False, 'item': None}"""
    res = []
    new_tile = tile
    for i in range(4):
        if new_tile not in res:
            res.append(new_tile)
        new_tile = turn_tile(tile)
    return res

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time} seconds.")
        return result
    return wrapper


#-------------LEVEL_3--------------------------
"""_______Main_functions_________"""

def successors(state, index):
    """Check all possible movements starting from the index of the tile."""
    res = []
    board = state['board']
    # tile = board[index]
    # tile_ID = tile_finder(tile, state)
    print('current tile = ', index)
    for direction in ["N", "S", "E", "W"]:
        # print("current_tile : ",)  # direction, " = ", board[index][direction]
        if board[index][direction]: #check if current tile is open in the asked direction
            coords = add(index2coords(index), DIRECTIONS[direction]["coords"]) #coords of the next tile
            next_tile_index = coords2index(*coords)
            # print("coord_valid : ", isCoordsValid(*coords))
            if isCoordsValid(*coords):
                next_tile = board[next_tile_index] #retrieve the NSWE-direction of the next tile
                opposite_dir = DIRECTIONS[direction]["opposite"]
                # print("next_tile : ",opposite_dir, " = ", next_tile[opposite_dir])
                if next_tile[opposite_dir]:
                    res.append(next_tile_index)
                    # print("added index : ",next_tile_index, "  ", board[next_tile_index])
        print('------------')
    print('poss_moves_list = ', res)
    return res

def next(state, move):
        """Create a copy of the board accounting for the changes inserting a tile in a gate makes."""
        new_state = copy.deepcopy(state)

        new_board, new_free = slideTiles(board=state["board"], free=move["tile"], gate=move["gate"])
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

def showBoard(board):
    mat = []
    for i in range(28):
        mat.append([])
        for j in range(28):
            mat[i].append(" ")
    for index, value in enumerate(board):
        i = (index // 7) * 4
        j = (index % 7) * 4
        mat[i][j] = "#"
        mat[i][j + 1] = "#" if not value["N"] else " "
        mat[i][j + 2] = "#"
        mat[i][j + 3] = "|"
        mat[i + 1][j] = "#" if not value["W"] else " "
        mat[i + 1][j + 1] = (
            " " if value["item"] is None else chr(ord("A") + value["item"])
        )
        mat[i + 1][j + 2] = "#" if not value["E"] else " "
        mat[i + 1][j + 3] = "|"
        mat[i + 2][j] = "#"
        mat[i + 2][j + 1] = "#" if not value["S"] else " "
        mat[i + 2][j + 2] = "#"
        mat[i + 2][j + 3] = "|"
        mat[i + 3][j] = "-"
        mat[i + 3][j + 1] = "-"
        mat[i + 3][j + 2] = "-"
        mat[i + 3][j + 3] = "-"

    print("\n".join(["".join(line) for line in mat]))

class PriorityQueue:
    def __init__(self):
        self.data = []
        self.historic = []

    def enqueue(self, value, priority):
        self.data.append({'value': value, 'priority': priority})
        self.data.sort(key=lambda elem: elem['priority'], reverse=True)

    def add_to_historic(self, value, priority):
        self.historic.append({'value': value, 'priority': priority})
        self.historic.sort(key=lambda elem: elem['priority'], reverse=True)

    def dequeue(self):
        return self.data.pop(0)['value']

    def isEmpty(self):
        return len(self.data) == 0

    def show_list(self):
        return self.data
    
    def show_historic(self):
        return self.historic

def manhattan_distance(current_pos, goal_pos):
    """Calculate the Manhattan distance heuristic between 2 given tiles."""
    start = index2coords(current_pos)
    end = index2coords(goal_pos)
    return abs(start[0] - end[0]) + abs(start[1] - end[1])


def BestFS(start, state, successors, target_tile, heuristic):
    """ start= my_index
        node : also an index
        parent = {tile : parent_of tile}
    """
    q = PriorityQueue()  #stores the nodes to be visited
    parent = {}  # stores the parent nodes of each visited node
    parent[start] = None
    q.enqueue(start, heuristic(start, target_tile))  # add the initial node (the board passed as parameter) to the queue q with a priority value of heuristic(board, target)

    while not q.isEmpty():
        node = q.dequeue() # dequeues the node with the LOWEST priority from the priority queue
        if node == target_tile:
            return (node, None, True)  #TODO TRUE ??
        for successor in successors(state, index=node): #for move in possible_moves_list (given by the successors function)
            if successor not in parent: #checks if node has already been visited
                parent[successor] = node  #adds it to the parent dictionary, 
                q.enqueue(successor, heuristic(successor, target_tile))
                q.add_to_historic(successor, heuristic(successor, target_tile))
                # print('Priority list = ')
                # q.show_list()
        node = None

    Best = {'value': None, 'priority': 9999}
    for i in q.show_historic():
        if i['priority'] <= Best['priority']:
            Best = i

    Best_tile = Best['value']
    Priority = Best['priority']
    return (Best_tile, Priority, False)


@timeit
def find_best_move(start, state, successors, target_tile, heuristic):
    """Call BestFS for all possible gate placements and tile orientations,
        returns a path to target if not the path with the best priority.
    """
    res = []
    best = {'choice': None, 'priority': 9999}
    for chosen_gate in GATES.keys():
        tile_to_insert = state['tile']  #the RAW tile we have just received
        for tile in possible_orientations(tile_to_insert):
            new_board = next(state, move={"tile": tile, "gate": chosen_gate})
            chosen_move = BestFS(start, state, successors, target_tile, heuristic)   #returns (chosen_tile=7 , priority=1, path_to_target=True)
            choice = {"tile": tile, "gate": chosen_gate, "new_position": chosen_move[0]}
            res.append({'choice': choice, 'priority': chosen_move[1]})
        
        for i in res:
            if i['priority'] <= best['priority']:
                best = i
    print(best['choice'] )
    return best['choice']    # return chosen_move = {"tile": tile, "gate": chosen_gate, "new_position": chosen_move}




#======RUN================================


# 'board': [{'N': False, 'E': True, 'S': True, 'W': False, 'item': None}, {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, {'N': False, 'E': True, 'S': True, 'W': True, 'item': 0}, {'N': False, 'E': False, 'S': True, 'W': True, 'item': 13}, {'N': False, 'E': True, 'S': True, 'W': True, 'item': 1}, {'N': False, 'E': True, 'S': False, 'W': True, 'item': None}, {'N': False, 'E': False, 'S': True, 'W': True, 'item': None}, {'N': True, 'E': False, 'S': False, 'W': True, 'item': 12}, {'N': False, 'E': False, 'S': True, 'W': True, 'item': None}, {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, {'N': False, 'E': True, 'S': False, 'W': True, 'item': None}, {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, {'N': True, 'E': True, 'S': True, 'W': False, 'item': 2}, {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, {'N': True, 'E': True, 'S': True, 'W': False, 'item': 3}, {'N': True, 'E': True, 'S': False, 'W': True, 'item': 19}, {'N': False, 'E': True, 'S': True, 'W': True, 'item': 4}, {'N': False, 'E': False, 'S': True, 'W': True, 'item': 16}, {'N': True, 'E': False, 'S': True, 'W': True, 'item': 5}, {'N': True, 'E': True, 'S': True, 'W': False, 'item': 20}, {'N': False, 'E': True, 'S': True, 'W': True, 'item': 23}, {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, {'N': False, 'E': True, 'S': True, 'W': False, 'item': None}, {'N': True, 'E': True, 'S': False, 'W': False, 'item': None}, {'N': False, 'E': False, 'S': True, 'W': True, 'item': 17}, {'N': False, 'E': True, 'S': True, 'W': False, 'item': None}, {'N': True, 'E': True, 'S': True, 'W': False, 'item': 6}, {'N': True, 'E': True, 'S': False, 'W': True, 'item': 21}, {'N': True, 'E': True, 'S': False, 'W': True, 'item': 7}, {'N': True, 'E': False, 'S': False, 'W': True, 'item': None}, {'N': True, 'E': False, 'S': True, 'W': True, 'item': 8}, {'N': False, 'E': False, 'S': True, 'W': True, 'item': None}, {'N': True, 'E': False, 'S': True, 'W': True, 'item': 9}, {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, {'N': True, 'E': True, 'S': False, 'W': True, 'item': 18}, {'N': True, 'E': True, 'S': False, 'W': False, 'item': 15}, {'N': True, 'E': True, 'S': False, 'W': False, 'item': 14}, {'N': True, 'E': False, 'S': True, 'W': True, 'item': 22}, {'N': False, 'E': True, 'S': True, 'W': False, 'item': None}, {'N': False, 'E': False, 'S': True, 'W': True, 'item': None}, {'N': True, 'E': True, 'S': False, 'W': False, 'item': None}, {'N': False, 'E': False, 'S': True, 'W': True, 'item': None}, {'N': True, 'E': True, 'S': False, 'W': True, 'item': 10}, {'N': False, 'E': True, 'S': True, 'W': False, 'item': None}, {'N': True, 'E': True, 'S': False, 'W': True, 'item': 11}, {'N': False, 'E': True, 'S': False, 'W': True, 'item': None}, {'N': True, 'E': False, 'S': False, 'W': True, 'item': None}]


message ={'request': 'play',
        'lives': 3, 
        'errors': [], 
        'state': {'players': ['Player_1', 'Player_2'],
                'current': 0, 
                'positions': [0, 48], 
                'board': [{'N': False, 'E': True, 'S': True, 'W': False, 'item': None}, {'N': True, 'E': True, 'S': False, 'W': True, 'item': None}, {'N': False, 'E': True, 'S': True, 'W': True, 'item': 0}, {'N': True, 'E': True, 'S': False, 'W': False, 'item': 19}, {'N': False, 'E': True, 'S': True, 'W': True, 'item': 1}, {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, {'N': False, 'E': False, 'S': True, 'W': True, 'item': None}, {'N': False, 'E': True, 'S': True, 'W': False, 'item': None}, {'N': False, 'E': False, 'S': True, 'W': True, 'item': 15}, {'N': True, 'E': True, 'S': False, 'W': True, 'item': 22}, {'N': False, 'E': False, 'S': True, 'W': True, 'item': None}, {'N': False, 'E': True, 'S': False, 'W': True, 'item': None}, {'N': False, 'E': True, 'S': False, 'W': True, 'item': None}, {'N': True, 'E': False, 'S': False, 'W': True, 'item': 16}, {'N': True, 'E': True, 'S': True, 'W': False, 'item': 2}, {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, {'N': True, 'E': True, 'S': True, 'W': False, 'item': 3}, {'N': True, 'E': True, 'S': False, 'W': True, 'item': 18}, {'N': False, 'E': True, 'S': True, 'W': True, 'item': 4}, {'N': True, 'E': False, 'S': False, 'W': True, 'item': None}, {'N': True, 'E': False, 'S': True, 'W': True, 'item': 5}, {'N': False, 'E': True, 'S': False, 'W': True, 'item': None}, {'N': True, 'E': False, 'S': False, 'W': True, 'item': None}, {'N': False, 'E': True, 'S': True, 'W': False, 'item': None}, {'N': False, 'E': True, 'S': True, 'W': True, 'item': 21}, {'N': True, 'E': True, 'S': False, 'W': False, 'item': 14}, {'N': True, 'E': True, 'S': False, 'W': False, 'item': None}, {'N': True, 'E': True, 'S': False, 'W': True, 'item': 20}, {'N': True, 'E': True, 'S': True, 'W': False, 'item': 6}, {'N': True, 'E': True, 'S': False, 'W': False, 'item': 12}, {'N': True, 'E': True, 'S': False, 'W': True, 'item': 7}, {'N': False, 'E': False, 'S': True, 'W': True, 'item': 17}, {'N': True, 'E': False, 'S': True, 'W': True, 'item': 8}, {'N': False, 'E': True, 'S': False, 'W': True, 'item': None}, {'N': True, 'E': False, 'S': True, 'W': True, 'item': 9}, {'N': True, 'E': False, 'S': False, 'W': True, 'item': None}, {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, {'N': False, 'E': True, 'S': True, 'W': False, 'item': None}, {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, {'N': False, 'E': False, 'S': True, 'W': True, 'item': None}, {'N': True, 'E': True, 'S': False, 'W': False, 'item': None}, {'N': False, 'E': True, 'S': False, 'W': True, 'item': None}, {'N': True, 'E': True, 'S': False, 'W': True, 'item': 10}, {'N': False, 'E': False, 'S': True, 'W': True, 'item': 13}, {'N': True, 'E': True, 'S': False, 'W': True, 'item': 11}, {'N': False, 'E': True, 'S': True, 'W': True, 'item': 23}, {'N': True, 'E': False, 'S': False, 'W': True, 'item': None}], 
                'tile': {'N': True, 'E': False, 'S': True, 'W': False, 'item': None}, 
                'target': 19, 
                'remaining': [4, 4]}}     


state = message['state']


board = state['board']
my_index = state['current']
current_pos = state['positions'][my_index]
my_target = target_finder(state)



print("start_pos :",current_pos)
print('my target :',my_target)

showBoard(board)



print('############################')
find_best_move(current_pos, state, successors, my_target, manhattan_distance)




"""NEXT_DEBUGGING"""

# tile = state['tile']

# chosen_gate = 'A'

# new_board = next(state, move={"tile": tile, "gate": chosen_gate})

# showBoard(new_board)

"""
###|# #|###|# #|###|# #|###|
#  |   | A |#T | B |# #|  #|
# #|###|# #|###|# #|# #|# #|
----------------------------
###|###|# #|###|###|###|# #|
#  | P#| W |  #|   |   | Q#|
# #|# #|###|# #|###|###|###|
----------------------------
# #|# #|# #|# #|###|# #|# #|
#C |# #|#D | S | E |  #| F#|
# #|# #|# #|###|# #|###|# #|
----------------------------
###|# #|###|###|# #|# #|# #|
   |  #|#  | V |#O |#  | U |
###|###|# #|# #|###|###|###|
----------------------------
# #|# #|# #|###|# #|###|# #|
#G |#M | H | R#| I#|   | J#|
# #|###|###|# #|# #|###|# #|
----------------------------
# #|# #|# #|# #|###|# #|###|
  #|# #|# #|# #|#  |# #|  #|
###|# #|# #|# #|# #|# #|# #|
----------------------------
# #|###|# #|###|# #|###|# #|
#  |   | K | N#| L | X |  #|
###|###|###|# #|###|# #|###|
#---------------------------

"""