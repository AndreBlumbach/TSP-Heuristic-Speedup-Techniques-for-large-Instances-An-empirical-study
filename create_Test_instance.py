import random, math
import numpy as np

Vertices = []

node_number = 10000
for i in range(node_number):
    
    x =  round(random.uniform(0, 10000),2)
    y =  round(random.uniform(0, 10000),2)
    
    vertex = (i,x,y)
    Vertices.append(vertex)
    

with open(f"Testinstance_{node_number}_x.txt", "a") as file:
    for node in Vertices:
        file.write(f"{node[0]}    {node[1]}    {node[2]}\n")
    file.write("EOF")




