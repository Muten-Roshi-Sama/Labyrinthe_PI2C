# Project Name : Labyrinthe_PI2C

##Project Description : 
Uses AI (Best First Search) to explores the search space by selecting the most promising node to visit next based on some heuristic evaluation function. There is a 3 seconds constraint to answer.


##Prerequisites
To run the agent, Python 3.8 or a newer version must be installed. Additionally, the following libraries must be installed:

socket: client-server communication
json: for loading and saving data in JSON format
copy: for making deep copies of objects
time: for measuring execution time


##Installation
Clone this repository.
Install the required dependencies using pip install -r requirements.txt.
Run the project using python : server.py, AI_player_1.py and AI_player_2.py


##Level_1 : """Connecting to the server"""
We connect to the server using the socket library.

##Level_2 : """Operator functions"""
Operator functions, credits : PI2CChampionship > Labyrinthe > game.py

##Level_3 : """MY_operator functions"""
Our own Operator functions to simplify the main functions.

##Level_4 : """Main_functions"""
successor(state, index): generates all possible moves from a given state (state = current game state). It generates
     the successors of the node (the following states) in the search tree. It uses the function "next" to calculate the next state for each possible move. It plays the role of a path function: a function that checks if there is a path (a sequence of positions) between the current player position and the nearest treasure. It returns the list of positions of the path (res).

next(state, move): a function that makes a deepcopy of the original state and insert a tile. 
    Credits : PI2CChampionship > Labyrinthe > game.py

find_best_move(state, successors, heuristic, flag): : Call BestFS for all possible gate placements and tile 
    orientations. Returns the new position with the lowest priority (i.e. 1, 2, represents the distance to target) using manhattan_distance as heuristic. Uses a flag to prevent sending the same Badmove again.


##Pytest
While running the plain server.py and AI_player_2.py in the background, we run test_player_1.
Using coverage, we obtain a +90% coverage.


##Authors
Flioris Vassily, 21245
Hamzi Loubna, 20026


##Credits
PI2CChampionship
ChatGPT-OpenAI
Stackoverflow

