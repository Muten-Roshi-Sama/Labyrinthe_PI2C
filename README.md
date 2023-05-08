# Labyrinthe_PI2C

###1###
Algorithme de recherche choisi(stratégie): Best First Searh   
L'algorithme explore d'abord les nœuds les plus proches du trésor ---\ but=trouver une solution plus rapidement.
Parcour l'espace de recherche et trouver le chemin le plus court qui mène à un trésor = évalue les nœuds (ou positions) en fonction de leur distance (heuristique) par rapport au trésor.

###2###
target_finder : fonction qui vérifie si un nœud(state) donné (ou position) correspond à un trésor

###3###
successor(state,index) : génère tout les coups possibles à partir d'un état donné (state= etat atuel du jeu) elle génère les successeur du noeud (les états suivants) dans l'arbre de recherche. Elle utilise la fonction next pour calculer l'état suivant pour chaque coup possible(voir ###4### pour next) 
    +
joue le role d'une fontion path : fonction qui vérifie si il existe un chemin (suite de positions) entre la position actuelle du joueur et le trésor le plus proche. Renvoie la liste des positions du chemin (res)

###4###
next : fonction qui calcule l'état suivant en fonction du coup joué (1 coup= inserer une tuile dans le board + Bouger le joueur)

###5###
Find_best_move : choisi le meilleur mouvement à jouer à chaque étape du jeu. 
INPUT = la liste des mouvements possible pour le joueur (générée par la fonction successor)
OUTPUT = renvoie le mouvement qui se raproche le plus du trésor

###6###
position_successive : calcule la position suivante du joueur en fonction de son mouvement (vers le haut, le bas, la gauche ou la droite)

----Bibliothèques utilisées----
socket : communication serveur-client
json : pour charger et sauvegarder des données au format JSON
copy : pour faire copies profondes d'objets
collections : utilisé pour des conteneurs spécialisés
time : pour mesurer le temps d'exécution
random : génère nombres aléatoires
sys :  interagi avec le système d'exploitation
math :  pour fonctions mathématiques
heapq : implémentation d'une file de priorité

