# Importing Libraries
import math
import sys
import random
import numpy as np
from functions import *
from initialization import *

# Input
# =============================================================================
# evacuee_node = int(input('Enter the evacuee node: '))
# evacuee_dict = {}
# 
# while evacuee_node != 0:
#     evacuee_speed = int(input('Enter the evacuee speed: '))
#     evacuee_dict.update({evacuee_node: evacuee_speed})
#     
#     evacuee_node = int(input('Enter the evacuee node: '))
#     
# fire_node = int(input('Enter the fire node: '))
# fire_dict = {}
# 
# while fire_node != 0:
#     fire_speed = int(input('Enter the fire speed: '))
#     fire_intensity = int(input('Enter the fire intensity: '))
#     fire_time = int(input('Enter the fire time: '))
#     fire_dict.update({fire_node: [fire_speed, fire_intensity, fire_time]})
#     
#     fire_node = int(input('Enter the fire node: '))
# =============================================================================
    
evacuee_dict = {11: 5, 4: 5, 7: 5, 22: 5, 52: 5, 54: 5, 5: 5, 31: 5, 73: 5, 27: 5}
fire_dict = {63: [2, 1000, 0]}

#Path-Planning
path_per_evacuee = {}
time_per_evacuee = {}
avg_path_cost = {}
for i, j in evacuee_dict.items():
    init_node = i
    final_path = []
    time_per_node = []
    init_evacu = init_node
    time = 0
    evacuee_speed = j
    cst, lp = 0, 0
    while final_path == [] or final_path[-1] not in exit_nodes:

        fire_dictionary = creating_fire_dict(time+1, fire_dict)

        fired = fire_dictionary[time]
        safe_nodes = all_nodes - fired
        #print(f'{safe_nodes}\n')
        # Congestion
        cong_person_attime = person_number_attime(time, path_per_evacuee, time_per_evacuee)
        
        if final_path == []:
            evacuee_set = evacuee_set_new(graph, coor_dict, init_node, evacuee_speed, time)
        elif init_node != init_evacu:
            init_evacu = init_node
            evacuee_set = evacuee_set_new(graph, coor_dict, init_evacu, evacuee_speed, time)


        target_node = exit_node_choose(init_node, exit_nodes, safe_nodes, graph)
        if target_node == None:
            print('Oops! no safe path available')
            break
        else:


            try:
                path_1, cost = path_plan_do(safe_nodes, graph, init_node, target_node, cong_person_attime, congestion_capacity)

                #print(f'The path for evacuee {i} at time {time} seconds is {path_1}')
                
                cst += cost
                

                for j in path_1:
                    if len(final_path) == 0:
                        if evacuee_set[j] <= time:
                            final_path.append(j)
                            time_per_node.append(time)
                            path_1.remove(j)
                        break
                    else:
                        if j != final_path[-1]:
                            if evacuee_set[j] <= time:
                                final_path.append(j)
                                time_per_node.append(time)
                                path_1.remove(j)
                            break          

                 #             else:
                #                 start_node = j
                #             break

            except KeyError:
                #final_path.append(0)
                print('Oops! no safe path available')
                break
            
            init_node = final_path[-1]
            time = time + 1
    path_per_evacuee.update({i:final_path})
    time_per_evacuee.update({i:time_per_node})
    avg_path_cost.update({i: cst / time})
    
print(f"The path per evacuee is: \n\t{path_per_evacuee}")
print(f"The time stamp is : \n\t{time_per_evacuee}")