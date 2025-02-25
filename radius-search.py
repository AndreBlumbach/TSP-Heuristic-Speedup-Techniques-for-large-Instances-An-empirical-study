import numpy as np
from scipy.spatial import KDTree
import time

def delta_for_candidates(tour, i, k):   
    if k+1 < len(tour):    
        # Knotenpaare vor Tausch
        A, B = tour[i-1], tour[i]
        C, D = tour[k], tour[k+1]
        
        x_ab = B[0] - A[0]
        y_ab = B[1] - A[1]
        distance_ab = (x_ab ** 2 + y_ab ** 2) ** 0.5
        x_ac = C[0] - A[0]
        y_ac = C[1] - A[1]
        distance_ac = (x_ac ** 2 + y_ac ** 2) ** 0.5
        x_bd = D[0] - B[0]
        y_bd = D[1] - B[1]
        distance_bd = (x_bd ** 2 + y_bd ** 2) ** 0.5
        x_cd = D[0] - C[0]
        y_cd = D[1] - C[1]
        distance_cd = (x_cd ** 2 + y_cd ** 2) ** 0.5
        
        # delta für neue Kanten: A-C und B-D
        delta = distance_ac + distance_bd - (distance_ab + distance_cd)
    else:
        A, B = tour[i-1], tour[i]
        C, D = tour[k], tour[0]
        
        x_ab = B[0] - A[0]
        y_ab = B[1] - A[1]
        distance_ab = (x_ab ** 2 + y_ab ** 2) ** 0.5
        x_ac = C[0] - A[0]
        y_ac = C[1] - A[1]
        distance_ac = (x_ac ** 2 + y_ac ** 2) ** 0.5
        x_bd = D[0] - B[0]
        y_bd = D[1] - B[1]
        distance_bd = (x_bd ** 2 + y_bd ** 2) ** 0.5
        x_cd = D[0] - C[0]
        y_cd = D[1] - C[1]
        distance_cd = (x_cd ** 2 + y_cd ** 2) ** 0.5
        
        # delta für neue Kanten: A-C und B-D
        delta = distance_ac + distance_bd - (distance_ab + distance_cd)
    
    return delta

def zwei_opt_move(tour, i, k):
    if k+1 < len(tour):
        return tour[:i] + tour[i:k+1][::-1] + tour[k+1:]
    else:
        return tour[:i] + tour[i:][::-1] 

def nearest_neighbor_kdtree(start_index, nodes, tree,  initial_radius=100, radius_step=200):
    tour_distance = 0
    first_node = nodes[start_index]
    tour = [first_node]
    n = len(nodes) 
    unvisited = set(range(n)) - {start_index}    
    current_index = start_index
    
    while unvisited:
        radius = initial_radius
        nearest_neighbor = None
        nearest_distance = float('inf') 
        
        # Radius-Erweiterungsschleife
        while nearest_neighbor is None:
            # Überprüfen der Nachbarn im aktuellen Radius
            candidates = tree.query_ball_point(nodes[current_index], r=radius)
            
            for index in candidates:
                if index in unvisited:
                    x = nodes[current_index][0] - nodes[index][0]
                    y = nodes[current_index][1] - nodes[index][1]
                    distance = (x ** 2 + y ** 2) ** 0.5
                    
                    if distance < nearest_distance:
                        nearest_neighbor = index
                        nearest_distance = distance
                        
            if nearest_neighbor is None:
                radius += radius_step
                
        tour.append(nodes[nearest_neighbor])
        unvisited.remove(nearest_neighbor)
        current_index = nearest_neighbor
        tour_distance += nearest_distance
        
    return tour,tour_distance

#Funktion zur radius Nachbarschaftssuche
def radius_search(index, tour, tree,  search_radius):
    n = len(tour)    
    node_index = index
    radius = search_radius
    
    candidates = tree.query_ball_point(tour[node_index], r=radius)
    return candidates

#Einlesen
nodes = []
with open("Testinstanz_1000_3.txt", "r") as file:
    lines = file.readlines()
    for line in lines[6:]: 
        if line.strip() == "EOF":
            break
        values = line.split()
        x, y = float(values[1]), float(values[2])
        nodes.append([x, y])

#kd-tree erstellen
nodes = np.array(nodes)
tree = KDTree(nodes)
nodes_dict = {tuple(coord): index for index, coord in enumerate(nodes)}

start_node = 0
tour, total_distance = nearest_neighbor_kdtree(start_node, nodes, tree)

total_distance += ((tour[-1][0] - tour[0][0]) ** 2 + (tour[-1][1] - tour[0][1]) ** 2) ** 0.5

Startlösung = tour #NN-Tour als Startlösung

start_time = time.perf_counter()

permut = Startlösung #NN-Tour als Startlösung
permut_Kosten = total_distance #Distanz der Startlösung
length = len(permut)
print(length)

permut_dict = {tuple(coord): index for index, coord in enumerate(permut)}

improved = True
while improved:
    improved = False      
    # 2-opt mit Delta-Evaluations          
    for i in range(1, length-1):
            
        best_delta = 0
        x = permut[i-1][0] - permut[i][0]
        y = permut[i-1][1] - permut[i][1]
        distance = (x ** 2 + y ** 2) ** 0.5
        radius = distance+1
        candidates = radius_search(i-1,permut,tree,radius)
            
        for candidate in candidates:
                     
            coords = nodes[candidate]
            
            k = permut_dict.get(tuple(coords))
            
            if k <= i:
                continue
            
            delta = delta_for_candidates(permut, i, k)
                
            if delta < best_delta:
                
                best_delta = delta
                best_k = k
        if best_delta < -0.0000001:
        
            permut = zwei_opt_move(permut, i, best_k)
            permut_Kosten += best_delta
            print("permut Update:"+str(permut_Kosten))
            improved = True
            permut_dict = {tuple(coord): index for index, coord in enumerate(permut)}
        
        
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(elapsed_time)
print(permut_Kosten)