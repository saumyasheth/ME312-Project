import networkx as nx
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import random
import matplotlib.pyplot as plt
# random.seed(9001)

def drone_price(distance):
    return  distance/10

def local_price(distance):
    return distance

G=nx.MultiDiGraph()

vertices = range(3)

# Define edges with information about whether it's local or auto

drone_edges = [(i, j, {'distance': np.random.randint(10,30), 'is_Truck': False}) for i in vertices for j in vertices if i != j]
Truck_routes = [(0, 1),  (1,2),(2, 3), (3, 4)]  # Updated local_train_routes
Truck_edges = [(u, v, {'distance': np.random.randint(10, 30), 'is_Truck':True}) for u, v in Truck_routes]
#drone_edges = [(i, j, {'distance': random.randint(1, 10), 'is_Truck': False}) for i in vertices for j in vertices if i < j and (i, j) not in [(3, 8)]]
#drone_edges.append((3, 8, {'distance': 100, 'is_Truck': False}))

# Add nodes and edges to the graph
G.add_nodes_from(vertices)

G.add_edges_from(drone_edges)
G.add_edges_from(Truck_edges)

# Assign prices to edges
for u, v, d in G.edges(data=True): 
    d['price'] = drone_price(d['distance'])

    print(f"Edge: ({u}, {v}), Attributes: {d}")    
 
# Define optimization model (code omitted for brevity)

# Optimize the model (code omitted for brevity)

# Print solution (code omitted for brevity)

plt.show()
def my_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):

    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items
        

# Calculate positions of nodes using spring layout
pos = nx.spring_layout(G)
pos1 = nx.spring_layout(G)

# Visualizing the graph
plt.figure(figsize=(10, 6))



        
model = gp.Model("Path_Finding")

node_A = 1
node_B = 3

# # Draw auto edges
drone_edges = [(u, v) for u, v, d in G.edges(data=True) if not d['is_Truck']]
nx.draw_networkx_edges(G, pos, edgelist=drone_edges, edge_color='black', width=1.0, alpha=0.5)


Truck_labels = {(u, v): f"{d['distance']}" for u, v, d in G.edges(data=True) if d['is_Truck']}

# Draw Truck edges with different curvature
Truck_edges = [(u, v) for u, v, d in G.edges(data=True) if d['is_Truck']]
edge_weights = nx.get_edge_attributes(G,'distance')

# curved_edge_labels = {edge: edge_weights[edge] for edge in Truck_edges}
curved_edge_labels = {(u, v): f"{d['distance']}" for u, v, d in G.edges(data=True) if d['is_Truck']}


for u, v in Truck_edges:
    curve_factor = 0.2  # Adjust the curve factor for local edges
    edge_pos = [(pos[u][0], pos[v][0]),
                ((1 - curve_factor) * pos[u][0] + curve_factor * pos[v][0],
                 (1 - curve_factor) * pos[u][1] + curve_factor * pos[v][1]),
                ((1 - curve_factor) * pos[v][0] + curve_factor * pos[u][0],
                 (1 - curve_factor) * pos[v][1] + curve_factor * pos[u][1]),
                (pos[v][0], pos[u][0])]
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], connectionstyle=f'arc3, rad = {curve_factor}',
                           edge_color='red', width=1.0, alpha=0.5,arrows=True)
    my_draw_networkx_edge_labels(G, pos, edge_labels=curved_edge_labels,rotate=False,rad = curve_factor)



# Draw nodes with labels
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=2000)
nx.draw_networkx_labels(G, pos, font_size=10)

# Add edge labels
edge_labels = {(u, v): f"{d['distance']}" for u, v, d in G.edges(data=True) if not d['is_Truck']}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)


plt.title('Red is Truck black is drone')
plt.axis('off')

        
model = gp.Model("Path_Finding")



# Decision variables
x = {}
# y = {}
for i, j, data in G.edges(data=True):
    x[i, j , str(data["is_Truck"])] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    x[j, i , str(data["is_Truck"])] =  model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
    # y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")





# Define cost and time factors
cost_truck = 1  # Adjust based on relative cost
cost_drone = 0  # Adjust based on relative cost
time_truck = 2  # Adjust based on relative speed
time_drone = 0  # Adjust based on relative speed

# Objective function: minimize total cost considering time
# Objective function: minimize total cost considering time
model.setObjective(gp.quicksum((cost_truck * data['price'] * time_truck * x[i, j, str(data["is_Truck"])] +
                                cost_drone * data['price'] * time_drone * x[i, j, str(data["is_Truck"])])
                   for i, j, data in G.edges(data=True)), GRB.MINIMIZE)




list_of_xs_for_c1 = []

for neighbor in vertices:
    if neighbor!=node_A:
        for i, j, data in G.edges(data=True):
            if i==node_A and j==neighbor:
                list_of_xs_for_c1.append([node_A , neighbor , str(data["is_Truck"])])
            if j==node_A and i==neighbor:
                list_of_xs_for_c1.append([node_A , neighbor , str(data["is_Truck"])])

# print(list_of_xs_for_c1)

model.addConstr( gp.quicksum(x[i[0], i[1] , i[2]] for i in list_of_xs_for_c1)   == 1 , f"StartNode{node_A}")


list_of_xs_for_c2 = []

for neighbor in vertices:
    if neighbor!=node_B:
        for i, j, data in G.edges(data=True):
            if i==neighbor and j==node_B:
                list_of_xs_for_c2.append([neighbor , node_B , str(data["is_Truck"])])
            if j==neighbor and i==node_B:
                list_of_xs_for_c2.append([neighbor , node_B , str(data["is_Truck"])])

# print(list_of_xs_for_c2)

model.addConstr( gp.quicksum(x[i[0], i[1] , i[2]] for i in list_of_xs_for_c2)   == 1 , f"EndNode_{node_B}")  


print()
print()

list_of_xs_for_c3 = []

for (i_mid, j_mid,Truck_mid) in x:
    for (i_before, j_before,Truck_before) in x:
        for (i_after, j_after,Truck_after) in x:
            if j_before==i_mid and i_after==j_mid and i_before!=j_mid and j_after!=i_mid:
                list_of_xs_for_c3.append([ [i_before, j_before,Truck_before] ,[i_mid, j_mid,Truck_mid] , [i_after, j_after,Truck_after]])

                # print([ [i_before, j_before,local_before] ,[i_mid, j_mid,local_mid] , [i_after, j_after,local_after]])
                

for i in list_of_xs_for_c3: 
    if i[1][0]!=node_A and i[1][1]!=node_B:
        model.addConstr((x[i[1][0], i[1][1] , i[1][2]]==1) >> (gp.quicksum(x[j[0][0], j[0][1] , j[0][2]]+x[j[2][0], j[2][1] , j[2][2]] for j in list_of_xs_for_c3 if j[1]==i[1] ) >= 2 ) , f"c3")  
    if i[1][0]==node_A and i[1][1]!=node_B:
        model.addConstr((x[i[1][0], i[1][1] , i[1][2]]==1) >> (gp.quicksum(x[j[2][0], j[2][1] , j[2][2]] for j in list_of_xs_for_c3 if j[1]==i[1] ) >= 1) , f"c4")  
    if i[1][0]!=node_A and i[1][1]==node_B:
        model.addConstr((x[i[1][0], i[1][1] , i[1][2]]==1) >> (gp.quicksum(x[j[0][0], j[0][1] , j[0][2]] for j in list_of_xs_for_c3 if j[1]==i[1] ) >= 1) , f"c5")  

# # Add constraint for switching between truck and drone
# for i in vertices:
#     for j in vertices:
#         if i != j:
#             for is_Truck in ['True', 'False']:
#                 model.addConstr(x[i, j, is_Truck] + x[i, j, str(not eval(is_Truck))] <= 1, f"Switch_Constraint_{i}_{j}_{is_Truck}")


# Optimize the model
model.optimize()
# Print solution
if model.status == GRB.OPTIMAL:
    print("Optimal solution found.")
    for i, j , is_Truck in x:
         if x[i, j , is_Truck].x > 0.5:
              a1=""
              if is_Truck==True:
                a1 = "Truck"
              else:
                a1 = "Drone"


              print(f"Edge from node {i} to node {j} via ", a1 ," is selected.")
else:
    print("No solution found.")

plt.show()



# import networkx as nx
# import numpy as np
# import gurobipy as gp
# from gurobipy import GRB
# import matplotlib.pyplot as plt

# def drone_price(distance):
#     return distance*10

# def local_price(distance):
#     return distance

# # Create a graph
# G = nx.MultiDiGraph()

# vertices = range(7)

# # Define edges with information about whether it's local or auto
# drone_edges = [(i, j, {'distance': np.random.randint(10, 30), 'is_Truck': False}) for i in vertices for j in vertices if i != j]
# Truck_routes = [(0, 1), (2, 3), (3, 4)]  # Updated local_train_routes
# Truck_edges = [(u, v, {'distance': np.random.randint(10, 30), 'is_Truck': True}) for u, v in Truck_routes]

# # Add nodes and edges to the graph
# G.add_nodes_from(vertices)
# G.add_edges_from(drone_edges)
# G.add_edges_from(Truck_edges)

# # Assign prices to edges
# for u, v, d in G.edges(data=True): 
#     d['price'] = drone_price(d['distance'])

# # Define optimization model
# model = gp.Model("Path_Finding")

# # Decision variables
# x = {}
# for i, j, data in G.edges(data=True):
#     for is_Truck in [True, False]:
#         x[i, j, is_Truck] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{is_Truck}")

# # Objective function: minimize total cost considering time
# cost_truck = 1  # Adjust based on relative cost
# cost_drone = 200  # Adjust based on relative cost
# time_truck = 2  # Adjust based on relative speed
# time_drone = 1  # Adjust based on relative speed
# model.setObjective(gp.quicksum((cost_truck * data['price'] * time_truck * x[i, j, True] +
#                                 cost_drone * data['price'] * time_drone * x[i, j, False])
#                    for i, j, data in G.edges(data=True)), GRB.MINIMIZE)

# # Add constraints for start and end nodes
# start_node = 0
# end_node = 6
# model.addConstr(gp.quicksum(x[start_node, neighbor, is_Truck] for neighbor in vertices if neighbor != start_node for is_Truck in [True, False]) == 1, f"StartNode_{start_node}")
# model.addConstr(gp.quicksum(x[neighbor, end_node, is_Truck] for neighbor in vertices if neighbor != end_node for is_Truck in [True, False]) == 1, f"EndNode_{end_node}")

# # Add constraint for switching between truck and drone
# for i, j, is_Truck in x:
#     model.addConstr(x[i, j, is_Truck] + x[i, j, not is_Truck] <= 1, f"Switch_Constraint_{i}_{j}_{is_Truck}")

# # Optimize the model
# model.optimize()

# # Print solution
# if model.status == GRB.OPTIMAL: 
#     print("Optimal solution found.")
#     for i, j, is_Truck in x:
#         if x[i, j, is_Truck].x > 0.5:
#             print(is_Truck)
#             vehicle = "Truck" if is_Truck else "Drone"
#             print(f"Edge from node {i} to node {j} via {vehicle} is selected.")
# else:
#     print("No solution found.")


# import networkx as nx
# import numpy as np
# import gurobipy as gp
# from gurobipy import GRB

# def drone_price(distance):
#     return distance * 10

# def local_price(distance):
#     return distance

# # Create a graph
# G = nx.MultiDiGraph()

# vertices = range(8)

# # Define edges with information about whether it's local or auto
# drone_edges = [(i, j, {'distance': np.random.randint(10, 30), 'is_Truck': False}) for i in vertices for j in vertices if i != j]
# Truck_routes = [(0, 1), (2, 3), (3, 4)]  # Updated local_train_routes
# Truck_edges = [(u, v, {'distance': np.random.randint(10, 30), 'is_Truck': True}) for u, v in Truck_routes]

# # Add nodes and edges to the graph
# G.add_nodes_from(vertices)
# G.add_edges_from(drone_edges)
# G.add_edges_from(Truck_edges)

# # Assign prices to edges
# for u, v, d in G.edges(data=True): 
#     d['price'] = drone_price(d['distance'])

# # Define optimization model
# model = gp.Model("Path_Finding")

# # Decision variables
# x = {}
# for i, j, data in G.edges(data=True):
#     for is_Truck in [True, False]:
#         x[i, j, is_Truck] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{is_Truck}")

# # Objective function: minimize total cost considering time
# cost_truck = 1  # Adjust based on relative cost
# cost_drone = 1 # Adjust based on relative cost
# time_truck = 2  # Adjust based on relative speed
# time_drone = 1  # Adjust based on relative speed
# model.setObjective(gp.quicksum((cost_truck * data['price'] * time_truck * x[i, j, True] +
#                                 cost_drone * data['price'] * time_drone * x[i, j, False])
#                    for i, j, data in G.edges(data=True)), GRB.MINIMIZE)

# # Add constraints for start and end nodes
# start_node = 0
# end_node = 7
# model.addConstr(gp.quicksum(x[start_node, neighbor, is_Truck] for neighbor in vertices if neighbor != start_node for is_Truck in [True, False]) == 1, f"StartNode_{start_node}")
# model.addConstr(gp.quicksum(x[neighbor, end_node, is_Truck] for neighbor in vertices if neighbor != end_node for is_Truck in [True, False]) == 1, f"EndNode_{end_node}")

# # Add constraints for hybrid delivery system
# # for i, j in G.edges():
# #     model.addConstr(gp.quicksum(x[i, j, is_Truck] for is_Truck in [True, False]) == 1, f"Hybrid_Constraint_{i}_{j}")

# # Add constraints for transition between truck and drone
# for i, j in G.edges():
#     for is_Truck in [True, False]:
#         if (i, j, is_Truck) in x:  # Check if decision variable exists
#             model.addConstr(x[i, j, is_Truck] <= gp.quicksum(x[u, i, not is_Truck] for u in G.neighbors(i) if (u, i, not is_Truck) in x), f"Transition_Constraint_{i}_{j}_{is_Truck}")

# # Optimize the model
# model.optimize()

# # Print solution
# if model.status == GRB.OPTIMAL:
#     print("Optimal solution found.")
#     for i, j, is_Truck in x:
#         if x[i, j, is_Truck].x > 0.5:
#             vehicle = "Truck" if is_Truck else "Drone"
#             print(f"Edge from node {i} to node {j} via {vehicle} is selected.")
# else:
#     print("No solution found.")

