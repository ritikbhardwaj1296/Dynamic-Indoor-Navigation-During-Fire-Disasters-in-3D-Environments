# Importing Libraries
import math
import sys
import random
import numpy as np
from initialization import *


# Fire Spread Estimation
def fire_spread_per_sec(a, b, c, d, e, start_time):
    
    x = graph[a]
    #print(x)
    y = set()
    for i in x:
        y.add(i)
    dct1 = {}
    lst1 = []
    c.add(a)
    #print(y)
    for h in c:
        if h in y:
            y.remove(h)
            
    #print(x)
    
    for i in y:
            dist_i = math.sqrt(((coor_dict[i][0] - coor_dict[a][0]) ** 2) + ((coor_dict[i][1] - coor_dict[a][1]) ** 2))
            #print(f'Distance between node {a} and node {i} is {dist_i:.2f}')
            dst = int(dist_i)
            dct1.update({i: dst})
            time = dist_i / b
            #print(f'The time after which node {i} become fire node is {time:.2f}')
            lst1.append(int(time))
            #s.remove(i)

#     print(f'The node connected to and distance of the node from fire node: \n {dct1}')
#     print(f'Time for nodes to become fire node: \n {lst1}')

    fire_nodes = {}

    for k,l in dct1.items():
        for j in range(start_time, e):
                d_fire_spread = (j - start_time) * b
                if d_fire_spread >= l:
                    ints = d/(d_fire_spread ** 2)
                    intensity = float("{:.3f}".format(ints))
                    fire_nodes.update({k: [j, intensity]})
                    break


#         for v in x:
#             fire_spread_per_sec(v, b, s)

    return fire_nodes
    #print(fire_nodes)

        
    #print(fire_nodes)
    
def fire_spread_baby_fn(a, b, c, d, e, f, start_time):
    x = graph[a]
    #print(x)
    y = set()
    for i in x:
        y.add(i)
    dct1 = {}
    lst1 = []
    c.add(a)
    
    for h in c:
        if h in y:
            y.remove(h)
            
    #print(x)
    
    for i in y:
            dist_i = math.sqrt(((coor_dict[i][0] - coor_dict[a][0]) ** 2) + ((coor_dict[i][1] - coor_dict[a][1]) ** 2))
            #print(f'Distance between node {a} and node {i} is {dist_i:.2f}')
            dst = int(dist_i)
            dct1.update({i: dst})
            time = dist_i / b
            #print(f'The time after which node {i} become fire node is {time:.2f}')
            lst1.append(int(time))
            #s.remove(i)

#     print(f'The node connected to and distance of the node from fire node: \n {dct1}')
#     print(f'Time for nodes to become fire node: \n {lst1}')

    fire_nodes = {}

    #run_t = int(input('Enter runtime for fire: '))
    
    
    for k,l in dct1.items():
        for j in range(start_time, f):
                d_fire_spread = (j - start_time) * b
                if d_fire_spread >= l:
                    ints = e / (d_fire_spread ** 2)
                    intensity = float("{:.6f}".format(ints))
                    fire_nodes.update({k: [(j + d), intensity]})
                    break

    for i in fire_nodes.keys():
        c.add(i)
    
#         for v in x:
#             fire_spread_per_sec(v, b, s)

    return fire_nodes
    #print(fire_nodes)

        
    #print(fire_nodes)
        
        
def multi_fire_spread_fn(fire_n, fire_spread_speed, fire_intensity, tme, start_time):
    
    fire_list = []
    passed_nodes = set()
    set1 = fire_spread_per_sec(fire_n, fire_spread_speed, passed_nodes, fire_intensity, tme, start_time)
    fire_list.append(set1)


    def loop1(passed_nodes, fire_list):
        set3 = {}
        for i, j in fire_list[-1].items():
            set2 = fire_spread_baby_fn(i, fire_spread_speed, passed_nodes, j[0], j[1], tme, start_time)
            set3.update(set2)
        fire_list.append(set3)
    
    
    for t in range(tme):
        loop1(passed_nodes, fire_list)
    
    #print(fire_list)
    
    
    fire_set = {}
    fire_set.update({fire_n : start_time})
    for i in fire_list:
        for j in i.items():
            fire_set.update({j[0]: j[1][0]})
            #print(j[0])
            
            
    return fire_set


def creating_fire_dict(tme, fire_nodes):
    fire_dict_final = {}

    
    for i, j in fire_nodes.items():
        if j[2] <= tme:
        
            fire_set_return = {}
            fire_set_return = multi_fire_spread_fn(i, j[0], j[1], tme, j[2])

            for k, l in fire_set_return.items():
                if k not in fire_dict_final.keys():
                    fire_dict_final.update({k : l})

                else:
                    if fire_set_return[k] < fire_dict_final[k]:
                        fire_dict_final.update({k: l})
    
    fire_dictionary = {}
    for r in fire_nodes.values():
        start_time = r[2]       

        
        for i in range(start_time, tme):
            set1 = set()
            for j, k in fire_dict_final.items():
                if k < i:
                    set1.add(j)
            fire_dictionary.update({i : set1})
        
    return fire_dictionary



# Evacuee Spread
def evacuee_set_new(graph, coor_dict, start_node, evacuee_speed, time):
    evac_dict = {start_node: time}
    all_nodes = [a for a in range(1, 80)]
    
    while len(evac_dict) != len(all_nodes):
        temp_dict = {}
        for i, j in evac_dict.items():
            #temp_dict = {}
            get_neighbors = list(graph[i] - evac_dict.keys())
            for k in get_neighbors:
                time = int(dist(i, k, coor_dict) / evacuee_speed)
                temp_dict.update({k: time + j})  
        
        for l, m in temp_dict.items():
            evac_dict.update({l: m})
        
        #print(evac_dict)    
    return evac_dict

# Path-Planning
def no_of_people(congestion_person_attime, graph):
    edges_with_cong = []
    for i in congestion_person_attime:
        f1 = i[1][0]
        f2 = i[1][1]
        edges_with_cong.append([f1, f2])
    d123 = {}
    for j in range(1, 80):
        d1 = {}
        for k in graph[j]:
            d1.update({k: 0})
        d123.update({j: d1})
    l_0, l_1 = 0, 0    
    for l in edges_with_cong:
        l_0 = l[0]
        l_1 = l[1]
        d123[l_0][l_1] += 1
        
    return d123
            
def congestion_fn(congestion_capacity, congestion_person_attime, graph):
    n = no_of_people(congestion_person_attime, graph)
    Vol_per_person = 0.06
    congestion_at_time = {}
    for i in range(1, 80):
        a1 = {}
        for j in graph[i]:
            con = ((n[i][j] * Vol_per_person) + (1 / congestion_capacity[i][j])) / congestion_capacity[i][j]
            a1.update({j: con})
        congestion_at_time.update({i: a1})
    
    return congestion_at_time

def fire_vs_evacuee(graph, safe_nodes, congestion_person_attime, congestion_capacity):
    
    congsss = congestion_fn(congestion_capacity, congestion_person_attime, graph)
    
#     rand_list = []

#     for k in range(1, 80):
#         a = float("{:.2f}".format(random.uniform(1, 10)))
#         rand_list.append(a)        
            
    weighted_graph = {}
    list299 = []
    
    for i in range(1, 80):
        #print(i)
        #print(duplicate_graph[i])
        list298 = []
        for j in graph[i]:
    #         print(j)
             if j in safe_nodes:
    #             print(j)
                 dist = math.sqrt(((coor_dict[i][0] - coor_dict[j][0]) ** 2) + ((coor_dict[i][1] - coor_dict[j][1]) ** 2))
                 list298.append({j: [dist, congsss[i][j]]})
             else:
                list298.append({j : [float('inf'), congsss[i][j]]})

        list299.append(list298)
            
        
    list300 = []
    for i in list299:
        #print(i)
        dct = {}
        for j in i:
            for p, q in j.items():
                dct.update({p: q})
                #print(p, q)
        list300.append(dct)
        
        
    for i in range(1, 80):
        weighted_graph.update({i: list300[i-1]})
    
    
    return weighted_graph
        
def exit_node_choose(init_node, exit_nodes, safe_nodes, graph):
    t_node = None
    exit_node_list = list(exit_nodes)
    dist_rec = {}
    a = init_node
    #min_dist = 0
    for i in exit_nodes:
        distance = math.sqrt(((coor_dict[i][0] - coor_dict[a][0]) ** 2) + ((coor_dict[i][1] - coor_dict[a][1]) ** 2))
        dist_rec.update({distance: i})
    
    #lookup = {value: key for key, value in dist_rec}
    prev_nodes = {68: 56, 77: 66, 78: 79} 
    while len(dist_rec) != 0:
        min_dist_node_value = min(dist_rec.keys())

        min_dist_node = dist_rec[min_dist_node_value]

        if min_dist_node in safe_nodes and prev_nodes[min_dist_node] in safe_nodes:
#             if prev_nodes[min_dist_node] in safe_nodes:
            t_node = min_dist_node
            break
#             if min_dist_node == 68:
#                 if 56 in safe_nodes:
#                     t_node = min_dist_node
#                     break
#             elif min_dist_node == 77:
#                 if 66 in safe_nodes:
#                     t_node = min_dist_node
#                     break
#             elif min_dist_node == 78:
#                 if 79 in safe_nodes:
#                     t_node = min_dist_node
#                     break
        else:
            dist_rec.pop(min_dist_node_value)

    
#     if t_node == None:
#         return None
            
    return t_node


def dijkstra_algorithm(start_node, weighted_graph):
    unvisited_nodes = []
    for i in weighted_graph.keys():
        unvisited_nodes.append(i)
 
    # We'll use this dict to save the cost of visiting each node and update it as we move along the graph   
    shortest_path = {}
 
    # We'll use this dict to save the shortest known path to a node found so far
    previous_nodes = {}
 
    # We'll use max_value to initialize the "infinity" value of the unvisited nodes   
    max_value = sys.maxsize
    for node in unvisited_nodes:
        shortest_path[node] = max_value
    # However, we initialize the starting node's value with 0   
    shortest_path[start_node] = 0
    
    # The algorithm executes until we visit all nodes
    while unvisited_nodes:
        # The code block below finds the node with the lowest score
        current_min_node = None
        for node in unvisited_nodes: # Iterate over the nodes
            if current_min_node == None:
                current_min_node = node
            elif shortest_path[node] < shortest_path[current_min_node]:
                current_min_node = node
                
        # The code block below retrieves the current node's neighbors and updates their distances
        neighbors = []
        for i in weighted_graph[current_min_node].keys():
            neighbors.append(i)
        for neighbor in neighbors:
            tentative_value = shortest_path[current_min_node] + 0.7 * 0.001 * weighted_graph[current_min_node][neighbor][0] + 0.3 * weighted_graph[current_min_node][neighbor][1]
            if tentative_value < shortest_path[neighbor]:
                shortest_path[neighbor] = tentative_value
                # We also update the best path to the current node
                previous_nodes[neighbor] = current_min_node
 
        # After visiting its neighbors, we mark the node as "visited"
        unvisited_nodes.remove(current_min_node)
    
    return previous_nodes, shortest_path

def print_result(previous_nodes, shortest_path, start_node, target_node):
    path = []
    node = target_node
    
    while node != start_node:
        path.append((node))
        node = previous_nodes[node]
 
    # Add the start node manually
    path.append((start_node))
    
    #print("We found the following best path with a value of {}.".format(shortest_path[target_node]))
    #print(" -> ".join(reversed(path)))
    y = list(reversed(path))
    
    return y, shortest_path[target_node]

def path_plan_do(safe_nodes, d_graph, start_node, target_node, congestion_person_attime, congestion_capacity):
    
    weighted_graph = fire_vs_evacuee(graph, safe_nodes, congestion_person_attime, congestion_capacity)
    
    #print(weighted_graph)
    
    previous_nodes, shortest_path = dijkstra_algorithm(start_node, weighted_graph)
    
    #print(previous_nodes, shortest_path)
    
    path_list, cost  = print_result(previous_nodes, shortest_path, start_node, target_node)
    
    #print(path_list)
    
    return path_list, cost

def person_number_attime(time, path_per_evacuee, time_per_evacuee):
    fin_lst = []
    for i in list(path_per_evacuee.keys()):
        #dct = {}
        edge_node_0 = 0
        edge_node_1 = 0
        for j in range(len(time_per_evacuee[i])):
            if time_per_evacuee[i][j] <= time:
                edge_node_0 = path_per_evacuee[i][j]
                try:
                    edge_node_1 = path_per_evacuee[i][j+1]
                except IndexError:
                    edge_node_1 = path_per_evacuee[i][j-1]
        fin_lst.append([i, [edge_node_0, edge_node_1]])
        
    return fin_lst

def dist(i, j, coor_dict):
    distance = math.sqrt(((coor_dict[i][0] - coor_dict[j][0]) ** 2) + ((coor_dict[i][1] - coor_dict[j][1]) ** 2))
    
    return distance
        
def inp_vect(start, fired):
    lst1 = []
    lst2 = []
    
    for i in range(1, 80):
        if i == start:
            lst1.append(1)
        else:
            lst1.append(0)
            
    for j in range(1, 80):
        if j in fired:
            lst2.append(i)
        else:
            lst2.append(0)
            
    lst1.extend(lst2)
    lst1.append(1)
    
    x_vec = np.array(lst1)
    
    return x_vec


def outtonode(output):
    count = 0
    for i in output:
        if i != 0:
            count += 1
            break
        else:
            count += 1
            
    return count

def probtohot(X):
    large = -10
    indx = 0
    for i, j in enumerate(X):
        if j > large:
            large = j
            indx = i
            
    h_vector = []
    
    for k in range(0, 80):
        if k == indx:
            h_vector.append(1)
        else:
            h_vector.append(0)
            
    return h_vector

def checktargetnde(ndes, e_nodes):
    bvg = {}
    for ihj in e_nodes:
        dfgt = dist(ihj, ndes, coor_dict)
        bvg.update({ihj: dfgt})
    ntochose = sorted(bvg, key=bvg.get)[0]
    return ntochose

def nextnodechose(startnode, outpp):
    dict125 = {}
    for cde, outpo in enumerate(outpp):
        dict125.update({cde+1: outpo})

    srt_lst = sorted(dict125, key=dict125.get)[:5]
    
    dist_dict_125 = {}
    for abcx in srt_lst:
        t_nde_chk = checktargetnde(abcx, exit_nodes)
        dist_targ = dist(abcx, t_nde_chk, coor_dict)
        dist_abcx = dist(startnode, abcx, coor_dict)
        totod = dist_targ + dist_abcx
        dist_dict_125.update({abcx: totod})
        
    final_node = sorted(dist_dict_125, key=dist_dict_125.get)[0]
    
    return final_node
    
    
    
    
    
