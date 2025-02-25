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


#Einlesen
nodes = []
with open("Testinstanz_1000_cluster.txt", "r") as file:
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

upper_left = []
upper_right = []
lower_left = []
lower_right = []

for node in nodes:
    if node[0] < 50000 and node[1] < 50000:
        lower_left.append(node)
    elif node[0] < 50000 and node[1] > 50000:
        upper_left.append(node)    
    elif node[0] > 50000 and node[1] < 50000:
        lower_right.append(node)
    elif node[0] > 50000 and node[1] > 50000:
        upper_right.append(node)

tree_u_l = KDTree(upper_left)
tree_u_r = KDTree(upper_right)
tree_l_l = KDTree(lower_left)
tree_l_r = KDTree(lower_right)

# Initialisiere die Matrix für die k nächsten Nachbarn
nn_number = 40
nn_matrix = np.zeros((len(nodes), nn_number),dtype=int)
# Fülle die Matrix für die 5 nächsten Nachbarn
for i in range(len(nodes)):
    u_l = tree_u_l.query(nodes[i], k=10)
    u_l_coords =[]
    for index in u_l[1]:
        u_l_coords.append(upper_left[index])
    u_l_index = []  
    for coord in u_l_coords:
        main_index = nodes_dict.get(tuple(coord))
        u_l_index.append(main_index)
      
    u_r = tree_u_r.query(nodes[i], k=10)
    u_r_coords =[]
    for index in u_r[1]:
        u_r_coords.append(upper_right[index])
    u_r_index = []  
    for coord in u_r_coords:
        main_index = nodes_dict.get(tuple(coord))
        u_r_index.append(main_index)
    
    l_l = tree_l_l.query(nodes[i], k=10)
    l_l_coords =[]
    for index in l_l[1]:
        l_l_coords.append(lower_left[index])
    l_l_index = []  
    for coord in l_l_coords:
        main_index = nodes_dict.get(tuple(coord))
        l_l_index.append(main_index)
    
    l_r = tree_l_r.query(nodes[i], k=10)
    l_r_coords =[]
    for index in l_r[1]:
        l_r_coords.append(lower_right[index])
    l_r_index = []  
    for coord in l_r_coords:
        main_index = nodes_dict.get(tuple(coord))
        l_r_index.append(main_index) 
        
    nearest_neighbors = np.concatenate((u_l_index, u_r_index, l_l_index, l_r_index))

    nn_matrix[i, :] = nearest_neighbors

start_node = 0
tour, total_distance = nearest_neighbor_kdtree(start_node, nodes, tree)

total_distance += ((tour[-1][0] - tour[0][0]) ** 2 + (tour[-1][1] - tour[0][1]) ** 2) ** 0.5
#tour.append(nodes[0])  
Startlösung = tour #NN-Tour als Startlösung

start_time = time.perf_counter()

permut = Startlösung #NN-Tour als Startlösung
permut_Kosten = total_distance #Distanz der Startlösung
length = len(permut)

permut_dict = {tuple(coord): index for index, coord in enumerate(permut)}
Gesamtersparnis = 0
Move_counter = 0
improved = True
while improved:
    improved = False      
    # 2-opt mit Delta-Evaluations
    for i in range(1, length - 1):
        
        best_delta = 0
        coord_i = permut[i]
        l = nodes_dict.get(tuple(coord_i)) 
        for k in range(40):
            
            candidate = nn_matrix[l][k]
            coords = nodes[candidate]
            j = permut_dict.get(tuple(coords))
            
            if j <= i:
                continue
            
            delta = delta_for_candidates(permut, i, j)
                
            if delta < best_delta:
                
                best_delta = delta
                best_j = j
                
        if best_delta < -0.0000001:
        
            permut = zwei_opt_move(permut, i, best_j)
            permut_Kosten += best_delta
            improved = True
            permut_dict = {tuple(coord): index for index, coord in enumerate(permut)}
            
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(elapsed_time)
print(permut_Kosten)