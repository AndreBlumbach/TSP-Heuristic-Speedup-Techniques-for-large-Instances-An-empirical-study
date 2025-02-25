import math
import random

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
distance_matrix = [[0] * num_nodes for _ in range(num_nodes)]
not_visited = set(range(1, num_nodes))
tour = [0]  # Startknoten
    
while not_visited:
    last_node = nodes[tour[-1]]
    closest_node = None
    closest_distance = float('inf')
        
    for i in not_visited:
        dst = last_node.distance_to(nodes[i])
        if dst < closest_distance:
            closest_node = i
            closest_distance = dst
        
    tour.append(closest_node)
    not_visited.remove(closest_node)
    total_distance += closest_distance
    
# Rückkehr zum Startknoten
total_distance += nodes[tour[-1]].distance_to(nodes[tour[0]])
tour.append(0) 

print(total_distance)
print(tour)
