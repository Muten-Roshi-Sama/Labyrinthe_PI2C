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


#GIT : /OneDrive/Bureau/Cours/BA2/Q2/Projet_Info/Labo_5_Projet/Labyrinthe_PI2C



#-----Variables----------
player = 2

if player == 1:
    player_name = "Player_1"
    request_port = 8880
    Matricules = ["1000", "2000"]
elif player == 2 :
    player_name = "Player_2"
    request_port = 2000
    Matricules = ["21245", "3333"]


max_recv_length = 10000
timeout = 3000 #milliseconds
serverAddress = ('localhost', 3000) 




#--------------Level_1--------------------------
"""Connecting to the server"""


request = {
"request": "subscribe",
"port": request_port,
"name": player_name,
"matricules": Matricules
}

with socket.socket() as s:
    s.connect(serverAddress)
    s.send(json.dumps(request).encode())
    response = s.recv(max_recv_length).decode()
print(response)

def main():
    print("------------------------")
    with socket.socket() as s:
        s.bind(('', request_port))
        s.settimeout(1)
        s.listen()
        try:
            client, address = s.accept()
            with client:
                request = client.recv(max_recv_length).decode()
                # print("request =  ",request)
                req = json.loads(request)
                message = req["request"]
                
                # print(message, 'hohoh')
                if message == "ping":
                    client.send(json.dumps({'response': 'pong'}).encode())

                elif message == "state":
                    time_start = time.time()
                    state = message
                    # print(state)
                    my_index = state["current"]
                    current_pos = state["position"][my_index]
                    # print(state["positions"][my_index])
                    # print(BFS(state["positions"][0]))

                elif message == "play":            #TODO
                    lives = req["lives"]
                    error_list = req["errors"]
                    print("ERRORS : ", error_list)
                    state = req["state"]
                    my_index = state["current"]
                    current_pos = state["positions"][my_index]
                    print("Current_pos : ", current_pos)
                    #
                    #-------------
                    #
                    tile = state["tile"]

                    chosen_gate = random.choice(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K","L"])
                    move_gate = {"tile": tile, "gate": chosen_gate}
                    new_state = next(state, move_gate)
                    moves = successors(my_index, new_state)
                    print("moves = ",moves)
                    #
                    if len(moves) == 1:  # if seq contains 0 or 1 element
                        random_move = moves[0] if moves else None  # select the first element, or None if seq is empty
                    elif moves == []:
                        random_move = current_pos
                    else:
                        random_move = random.choice(moves)
                    #
                    # print(state["board"])
                    #
                    chosen_move = {"tile": tile, "gate": chosen_gate, "new_position": random_move}

                    print("chosen_move = ", chosen_move["new_position"])
                    print()
                    client.send(json.dumps({'response': 'move', 'move': chosen_move, "message" : "Hoho" }).encode())
                
                
        except socket.timeout:
            pass
        except OSError:
            print("Server address not reachable.")








#-----------------LEVEL_2------------------- 
"""Operator functions"""


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





#---------LEVEL_3--------------------
"""Moving and AI path """

def successors(index, state):
    """Check all possible movements starting from the index of the tile."""
    current_pos = state["positions"][index]
    board = state["board"]
    res = []
    for direction in ["N", "S", "E", "W"]:
        if board[current_pos][direction]: #check if current tile is open in the asked direction
            coords = add(index2coords(current_pos), DIRECTIONS[direction]["coords"]) #coords of the next tile
            if isCoordsValid(*coords):
                next_tile = board[coords2index(*coords)] #retrieve the NSWE of the next tile
                opposite_dir = DIRECTIONS[direction]["opposite"]
                if next_tile[opposite_dir]:
                    res.append(coords2index(*coords))
    return res







def next(state, move):

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

        # if (
        #     path(
        #         new_state["positions"][state["current"]],
        #         move["new_position"],
        #         new_state["board"],
        #     )
        #     is None
        # ):
        #     raise game.BadMove("Your new_position is unreachable")

        # new_state["positions"][state["current"]] = move["new_position"]

        # if (
        #     new_state["board"][new_state["positions"][state["current"]]]["item"]
        #     == targets[state["current"]][-1]
        # ):
        #     targets[state["current"]].pop()
        #     if len(targets[state["current"]]) == 0:
        #         new_state["remaining"] = 0
        #         new_state["target"] = None
        #         raise game.GameWin(state["current"], new_state)

        # new_state["current"] = (new_state["current"] + 1) % 2

        # new_state["target"] = targets[new_state["current"]][-1]
        # new_state["remaining"] = [len(trg) for trg in targets]

        return new_state






























#--------------RUN-------------------

while __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('player', help='The player you want to be (1 or 2)')
    # args = parser.parse_args()
    # asyncio.run(init_variables(args.player))
    main()




#-----------Temporary_Storage-----------------

class PriorityQueue:
	def __init__(self):
		self.data = []

	def enqueue(self, value, priority):
		# Could be better
		self.data.append({'value': value, 'priority': priority})
		self.data.sort(key=lambda elem: elem['priority'])

	def dequeue(self):
		return self.data.pop(0)['value']

	def isEmpty(self):
		return len(self.data) == 0



def BestFS(start,successors, goals, heuristic, state):
    q = PriorityQueue()
    parent = {}
    parent[start] = None
    q.enqueue(start, heuristic(start))
    while not q.isEmpty():
        node = q.dequeue()
        if path(state["positions"][0], state["target"], node, successors):
            break
        for successor in successors(node):
            if successor not in parent:
                parent[successor] = node
                q.enqueue(successor, heuristic(successor))
        node = None

    res = []
    while node is not None:
        res.append(node)
        node = parent[node]

    return list(reversed(res))  


def manhattan_distance(current_pos, goal_pos):
    """
    Calculate the Manhattan distance heuristic between current_pos and goal_pos.
    """
    return abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])



"""Heuristic: maximiser les nombre de tresors auquels j'ai acces et minimiser ceux de l'adversaire"""

