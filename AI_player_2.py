import threading
import socket
import json
# from .. import game
import copy
from collections import deque
import time as time
import random as r
#--
# import sys
# sys.path.append('C:\Users\vassi\OneDrive\Bureau\Cours\BA2\Q2\Projet_Info\Labo_5_Projet\PI2CChampionshipRunner\games\labyrinthe\game.py')
# from PI2CChampionshipRunner import games\labyrinthe\game.py

"""LAUNCH SERVER"""
# Work PC : & C:/Users/vassi/AppData/Local/Programs/Python/Python39/python.exe c:/Users/vassi/OneDrive/Bureau/Cours/BA2/Q2/Projet_Info/Labo_5_Projet/PI2CChampionshipRunner/server.py labyrinthe
# Home PC : & c:/Users/Vass/OneDrive/Bureau/Cours/BA2/Q2/Projet_Info/Labo_5_Projet/PI2CChampionshipRunner/server.py labyrinthe


#-----Variables----------
Player = 2

if Player == 1:
    player_name = "BobbyChapman"
    request_port = 8880
    Matricules = ["1000", "2000"]


elif Player == 2 :
    player_name = "Giorgio"
    request_port = 2000
    Matricules = ["21245", "3333"]

max_recv_length = 10000
timeout = 3000 #milliseconds

# serverAddress = ('177.17.10.59', 3000)
serverAddress = ('localhost', 3000) 
# serverAddress = ('192.168.0.138', 3000) #work PC


#-----------------------------------------------
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


#---------------------------------------------------------



#--------------Level_1--------------------------
"""Connecting to the server"""


"""
move = {"tile": turn_tile(turn_tile(turn_tile(turn_tile(state["tile"])))), "gate": "C", "new_position": 8})
"""
# s.connect(serverAddress)

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
    with socket.socket() as s:
        s.bind(('', request_port))
        s.settimeout(1)
        s.listen()
        try:
            client, address = s.accept()
            with client:
                request = client.recv(max_recv_length).decode()
                # print("1", request)

                message = json.loads(request)["request"]
                # print(message, 'hohoh')

                if message == "ping":
                    client.send(json.dumps({'response': 'pong'}).encode())

                elif message == "play":
                    print(message)
                    # lives = request["lives"]
                    # error_list = request["errors"]
                    # state = request["state"]

                    chosen_move = {"tile": 6, "gate": "C", "new_position": 8}
                    # client.send(json.dumps({'response': 'move', 'move': chosen_move, "message" : "Hoho" }))

                elif message == "state":
                    time_start = time.time()
                    state = message
                    # print(state)
                    print(state["positions"][0])
                    # print(BFS(state["positions"][0]))
        except socket.timeout:
            pass
        except OSError:
            print("Server address not reachable.")



#---------LEVEL_2--------------------
"""Moving and AI path """

"""
fonction heuristique : mesure de idistance de l'objectif

negamax en adaptant que notre jeu n'est pas a information parfaite
OU
on joue seul (Best-First meilleur choix selon prof + eliminer un max de poss au départ)

prof dans son code utilise du Breadth First mais peu importe car il joue en random)
"""

def BFS(start, successor):
    """Best_First_Search"""
    q = deque()
    # q.append(start)
    parent = {}
    parent[start]= None
    node = start
    while time.time() - time.start < timeout:
        for successor in successor(node):
            if successor not in parent:
                parent[successor] = node.enqueue(successor)
        node = q.dequeue()
    res = []
    while node is not None:
        res.append(node)
        node = parent[node]
    
    return list(reversed(res))


def BestFS(start,successors, goals, heuristic):
    q = deque()
    parent = {}
    parent[start] = None
    q.enqueue(start, heuristic(start))
    while not q.isEmpty():
        node = q.dequeue()
        if node in goals:
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



def successors(node, state):
    board = state["board"]
    current_pos = state['positions'][0]
    res = []
    directions = {'go_left' : (board[current_pos + 1]['W'], current_pos + 1),
                'go_right' : (board[current_pos - 1]['E'], current_pos - 1),
                'go_up' : (board[current_pos - 7]['S'], current_pos - 7),
                'go_down' : (board[current_pos + 7]['N'], current_pos + 7)}
    l, c = node
    for dl, dc in directions:
        nl = l + dl
        nc = c + dc
        if board[nl][nc] in [' ', 'E']:
            res.append((nl, nc))
    return res

def manhattan_distance(current_pos, goal_pos):
    """
    Calculate the Manhattan distance heuristic between current_pos and goal_pos.
    """
    return abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])


def heuristic_prof(node):
    l, c = node
    return (l-1)*2 + (c-9)**2





while __name__ == '__main__':
    main()













#-----------
'''

pos_history = []

def update_board(state, tile_number, new_tile, gate):
    state['board'][tile_number] = new_tile

def game_over(state):
    """Determines whether a player wins the game."""
    playerIndex = state['current']  #player_pos
    otherIndex = (playerIndex+1)%2  
    winner = False
    if state['remaining'][0]== '0': #numbers of treasures each player still has to reach
            winner = playerIndex
    elif state['remaining'][1]== '0':
        winner = otherIndex
    return winner


def find_free_tiles(state):
    """use alpha-pruning to find all free tiles"""
    res = []
    current_pos = state['positions'][0]

    go_left = (state['board'][current_pos + 1]['W'], current_pos + 1)
    go_right = (state['board'][current_pos - 1]['E'], current_pos - 1)
    go_up = (state['board'][current_pos - 7]['S'], current_pos - 7)
    go_down = (state['board'][current_pos + 7]['N'], current_pos + 7)

    for i in [go_left, go_right, go_up, go_down]:
        if i[1] == "true":
            i.append(res[0]) #list with first degree moves possible 

    for i in res:
        simulated_pos = i[1]
        go_left = (state['board'][simulated_pos + 1]['W'], simulated_pos + 1)
        go_right = (state['board'][simulated_pos - 1]['E'], simulated_pos - 1)
        go_up = (state['board'][simulated_pos - 7]['S'], simulated_pos - 7)
        go_down = (state['board'][simulated_pos + 7]['N'], simulated_pos + 7)

    return res


def all_moves(state):
    res = []
    tiles_placement = ['A','B','C','D','E','F','G','H','I','J','K','L']
    for i in tiles_placement:
        pass




def next(state, move):
    newState = copy.deepcopy(state)
    playerIndex = state['current']
    otherIndex = (playerIndex+1)%2

    if game_over() is not None:
        pass
    

#------------


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

    new_free = board[end]  #opposite tile pushed out of the game
    new_board = copy.deepcopy(board)
    dest = end
    src = end - inc # 43 - 7
    while dest != start:  #while 43 != 1
        new_board[dest] = new_board[src]  #last tile (43) becomes the precedent tile (43-7) etc.
        dest = src #43 becomes 43-7
        src -= inc # 43 - 7 -7
    new_board[start] = free # ejected tile
    return new_board, new_free






'''








