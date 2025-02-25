import numpy as np
from scipy.spatial import KDTree
import time

start_time = time.perf_counter()
nodes = []
with open("Testinstanz_5000_1.txt", "r") as file:
    lines = file.readlines()
    for line in lines[6:]: 
        if line.strip() == "EOF":
            break
        values = line.split()
        x, y = float(values[1]), float(values[2])
        nodes.append([x, y])

nodes = np.array(nodes)
tree = KDTree(nodes)

def nearest_neighbor_kdtree(start_index, nodes, tree,  initial_radius=100, radius_step=200):
    tourlänge = 0
    n = len(nodes)
    visited = [start_index] 
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
                
        visited.append(nearest_neighbor)
        unvisited.remove(nearest_neighbor)
        current_index = nearest_neighbor
        tourlänge += nearest_distance

    return visited,tourlänge

start_node = 0
tour, tourlänge = nearest_neighbor_kdtree(start_node, nodes, tree)


end_time = time.perf_counter()

tourlänge += ((nodes[tour[-1]][0] - nodes[tour[0]][0]) ** 2 + (nodes[tour[-1]][1] - nodes[tour[0]][1]) ** 2) ** 0.5
tour.append(0)                
elapsed_time = end_time - start_time
print(elapsed_time)
print(tourlänge)

