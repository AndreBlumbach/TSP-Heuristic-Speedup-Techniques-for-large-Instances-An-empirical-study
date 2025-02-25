import math
import random
import numpy as np
import time

class Node:
    def __init__(self, node_id, x, y):
        self.id = node_id
        self.x = x
        self.y = y

    def distance_to(self, other_node):
        return math.sqrt((self.x - other_node.x) ** 2 + (self.y - other_node.y) ** 2)

nodes = []
with open("Testinstanz_100_1.txt", 'r') as file:
    lines = file.readlines()
    for line in lines[6:]: 
        if line.strip() == "EOF":
            break
        parts = line.split()
        node_id = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        nodes.append(Node(node_id, x, y))

total_distance = 0
num_nodes = len(nodes)
dmatrix = np.zeros((len(nodes), len(nodes)))
not_visited = set(range(1, num_nodes))
tour = [0]  # Startknoten

while not_visited:
    last_node = nodes[tour[-1]]
    closest_node = None
    closest_distance = float('inf')
        
    for i in not_visited:
        dst = last_node.distance_to(nodes[i])
        a = nodes[tour[-1]].id
        dmatrix[i][a] = dst
        dmatrix[a][i] = dst
        if dst < closest_distance:
            closest_node = i
            closest_distance = dst
        
    tour.append(closest_node)
    not_visited.remove(closest_node)
    total_distance += closest_distance
    
# Rückkehr zum Startknoten
total_distance += nodes[tour[-1]].distance_to(nodes[tour[0]])
#tour.append(0)  

Startlösung = tour #NN-Tour als Startlösung

def zwei_opt_delta(tour, i, k, dist_matrix):   
    if k+1 < len(tour):    
        # Knotenpaare vor Tausch
        A, B = tour[i-1], tour[i]
        C, D = tour[k], tour[k+1]    
        # delta für neue Kanten: A-C und B-D
        delta = dist_matrix[A][C] + dist_matrix[B][D] - (dist_matrix[A][B] + dist_matrix[C][D])
    else:
        A, B = tour[i-1], tour[i]
        C, D = tour[k], tour[0]
        delta = dist_matrix[A][C] + dist_matrix[B][D] - (dist_matrix[A][B] + dist_matrix[C][D])     
    return delta

def zwei_opt_move(tour, i, k):
    if k+1 < len(tour):
        return tour[:i] + tour[i:k+1][::-1] + tour[k+1:]
    else:
        return tour[:i] + tour[i:][::-1] 

start_time = time.perf_counter()

permut = Startlösung #NN-Tour als Startlösung
permut_Kosten = total_distance #Distanz der Startlösung
length = len(permut)

d_l_b = np.zeros(len(nodes))
for i in range(len(nodes)):
    d_l_b[i] = 1
  
improved = True
while improved:
    improved = False      
    # 2-opt mit Delta-Evaluations
    for i in range(1, length - 1):
        if not d_l_b[i]:
            continue
        d_l_b[i] = 0
        for k in range(i + 1, length):    

            delta = zwei_opt_delta(permut, i, k, dmatrix)
                        
            if delta < -0.000001:
                
                permut = zwei_opt_move(permut, i, k)
                permut_Kosten += delta
                improved = True
                if k < length -1:
                    d_l_b[i]=1
                    d_l_b[i-1]=1
                    d_l_b[k]=1
                    d_l_b[k+1]=1
                else:
                    d_l_b[i]=1
                    d_l_b[i-1]=1
                    d_l_b[k]=1
                    d_l_b[0]=1
            
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(elapsed_time)
print(permut_Kosten)
