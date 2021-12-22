# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 21:31:53 2021

@author: simon
"""

#Function can be run by uncommenting bottom line

def get_goal_input():
    
    #Stack dictionary with locations
    locations = {
        "a": ["APPLES", [12, 100]],
        "b": ["BREAD", [19, 5.5]],
        "c": ["CHECKOUT", [95, 114]],
        "d": ["DRY YEAST", [40, 30]],
        "e": ["EGGS", [93, 32]],
        "f": ["FOIE GRAS", [125, 20]],
        "g": ["GUM", [73, 98]],
        } #connies??
    
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

get_goal_input()