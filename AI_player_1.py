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
import math 
import heapq


#GIT : /OneDrive/Bureau/Cours/BA2/Q2/Projet_Info/Labo_5_Projet/Labyrinthe_PI2C

# C:/Users/vassi/OneDrive/Bureau/Cours/BA2/Q2/Projet_Info/Labo_5_Projet/Labyrinthe_PI2C

# C:\Users\MediaMonster\OneDrive\Bureau\Labyrinthe_PI2C

#----------------------------
#TODO:
"""
-debug fetch error when connecting to server
-debugging when remaining on the same tile. res = []
-add timeout to find_best_move() of 2900ms

"""





#-----Variables----------
player = 1

if player == 1:
    player_name = "Player_1"
    request_port = 8880
    Matricules = ["1000", "2000"]
elif player == 2 :
    player_name = "Player_2"
    request_port = 2000
    Matricules = ["21245", "3333"]


max_recv_length = 10000
timeout = 2.95 #milliseconds
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

                elif message == "play":
                    # print("MESSAGE : ", req)
                    state = req['state']

                    board = state['board']
                    my_index = state['current']
                    current_pos = state['positions'][my_index]
                    my_target = target_finder(state)

                    print("start_pos :",current_pos)
                    print('my target :',my_target)

                    showBoard(board)

                    lives = req["lives"]
                    error_list = req["errors"]
                    print("ERRORS : ", error_list)

                    chosen_move = find_best_move(current_pos, state, successors, my_target, manhattan_distance)

                    client.send(json.dumps({'response': 'move', 'move': chosen_move, "message" : "Hoho" }).encode())

                    print('############################')
                
        except socket.timeout:
            pass
        except OSError:
            print("Server address not reachable.")








#-----------------LEVEL_2------------------- 
"""_______Operator functions_________"""


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

def index2coords(index):
    return index // 7, index % 7

def coords2index(i, j):
    return i * 7 + j

def isCoordsValid(i, j):
    return i >= 0 and i < 7 and j >= 0 and i < 7

def add(A, B):
    return tuple(a + b for a, b in zip(A, B))





#-------------LEVEL_3--------------------------
"""_______MY_Operator functions_________"""


def target_finder(state):
    target_ID = state["target"]
    board = state["board"]
    for i in board:
        if i['item'] == target_ID:
            return board.index(i)
    print('Could not find target')

def possible_orientations(tile):   #TODO use tuple to erase dubbles. 
    """Generate all possible orientations of a given tile.{'N': False, 'E': True, 'S': True, 'W': False, 'item': None}"""
    res = []
    new_tile = tile
    for i in range(4):
        if new_tile not in res:
            res.append(new_tile)
        new_tile = turn_tile(tile)
    return res


#-------------LEVEL_4--------------------------
"""_______Main_functions_________"""

def successors(state, index):
    """Check all possible movements starting from the index of the tile."""
    res = []
    board = state['board']
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
    Best = []
    q = PriorityQueue()  #stores the nodes to be visited
    parent = {}  # stores the parent nodes of each visited node
    parent[start] = None
    q.enqueue(start, heuristic(start, target_tile))  # add the initial node (the board passed as parameter) to the queue q with a priority value of heuristic(board, target)

    while not q.isEmpty():
        node = q.dequeue() # dequeues the node with the LOWEST priority from the priority queue
        # print('node = ', node)
        # print('target_tile = ', target_tile)
        if node == target_tile:
            return (node, None)
        for successor in successors(state, index=node): #for move in possible_moves_list (given by the successors function)
            if successor not in parent: #checks if node has already been visited
                parent[successor] = node  #adds it to the parent dictionary, 
                q.enqueue(successor, heuristic(successor, target_tile))
                q.add_to_historic(successor, heuristic(successor, target_tile))
                print('Priority list = ')
                q.show_list()
        node = None

    Best = {'value': None, 'priority': 9999}
    for i in q.show_historic():
        if i['priority'] <= Best['priority']:
            Best = i

    Best_tile = Best['value']
    Priority = Best['priority']
    return (Best_tile, Priority)



def find_best_move(start, state, successors, target_tile, heuristic):
    """Call BestFS for all possible gate placements and tile orientations,
        returns a path to target if not the path with the best priority.
    """
    start_time=time.time()
    res = []
    best = {'choice': None, 'priority': 9999}
    while time.time() - start_time > timeout:
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
        
        else:
            raise Exception("Temps imparti dépassé")

    

#--------------RUN-------------------

while __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('player', help='The player you want to be (1 or 2)')
    # args = parser.parse_args()
    # asyncio.run(init_variables(args.player))
    main()




#-----------Temporary_Storage-----------------

def a_star(state, start, goal):
    """A* algorithm to find the shortest path between two positions"""
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    distance = []
    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break
        for next_pos in successors(current, state):
            new_cost = cost_so_far[current] + 1  # assuming all moves have the same cost
            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                priority = new_cost + distance(next_pos, goal)
                heapq.heappush(frontier, (priority, next_pos))
                came_from[next_pos] = current

    path = []
    pos = goal
    while pos != start:
        path.append(pos)
        pos = came_from[pos]
    path.append(start)
    path.reverse()

    return path

def evaluate_move(state, move):
    current_pos = state["positions"][state["current"]]
    goal_pos = move["new_position"]
    path = a_star(state, current_pos, goal_pos)
    return -len(path)  # negative because we want the shortest path to have a higher score

def best_first_choice(state, moves, treasure_pos):
    my_index = state["current"]
    current_pos = state["positions"][my_index]
    best_move = None
    best_score = -float('inf')
    treasure_pos_2D = index2coords(treasure_pos)  
    for move in moves:
        new_pos = tuple(map(sum, zip(current_pos, move)))
        distance = math.sqrt((new_pos[0] - treasure_pos_2D[0])**2 + (new_pos[1] - treasure_pos_2D[1])**2) #TODO
        score = -distance  # negative because we want the shortest distance to have a higher score
        if score > best_score:
            best_move = move
            best_score = score
    return best_move



#-------------------------------------------------