# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 21:31:53 2021

@author: simon
"""

#Function can be run by uncommenting bottom line

def get_goal_input():
    
    #Stack dictionary with locations
    locations = {
        "a": ["APPLES", [100, 10]],
        "b": ["BREAD", [10, 65]],
        "c": ["CHECKOUT", [114, 73]],
        "d": ["DRY YEAST", [76, 116]],
        "e": ["EGGS", [93, 53]],
        "f": ["FOIE GRAS", [55, 94]],
        "g": ["GUM", [92, 124]],
        }
    
    #Print info text
    print("Please enter a new goal for the Supermarkt Robot: \n")
    locations_keys = list(locations.keys())
    for i in range(len(locations)):
        print(f'For the {locations[locations_keys[i]][0]}, type: {locations_keys[i]}')
    
    #Ask input
    goal_command = input()
    
    #Get them from string
    goal = locations[goal_command]
    chosen_product = goal[0]
    goal_coordinates = goal[1]
    print()
    print(f'You want the robot to go to the {chosen_product}, at: {goal_coordinates}')
    
    return goal_coordinates

# get_goal_input()