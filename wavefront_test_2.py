import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image

def simple_obstacle_grid(GRIDSIZE=0, threshold=0):
    obstacle_grid = np.zeros((10, 15))
    obstacle_grid[4:8, 6:10] = 1
    return obstacle_grid

def get_obstacle_grid(GRIDSIZE=5, threshold=.125):
    # Open image and convert to numpy array
    image = Image.open('supermarket2.jpeg')
    image = image.convert("1")
    image_array = np.asarray(image)
    image_array = np.abs(image_array-1)
    
    
    # Create an empty obstacle grid
    HEIGHT, WIDTH = np.shape(image_array)
    obstacle_grid = np.zeros((HEIGHT // GRIDSIZE - 1, WIDTH // GRIDSIZE - 1))
    
    # If there is an obstacle in a cell of the grid, 
    # mark the whole cell as an obstacle
    for row in range(HEIGHT // GRIDSIZE - 1):
        for col in range(WIDTH // GRIDSIZE - 1):
            if (np.sum(image_array[row*GRIDSIZE : row*GRIDSIZE + GRIDSIZE, 
                                    col*GRIDSIZE : col*GRIDSIZE + GRIDSIZE])
                                    >= (threshold * GRIDSIZE * GRIDSIZE)):
                obstacle_grid[row][col] = 1
    
    return obstacle_grid


def generate_wave(array, HEIGHT, WIDTH, directions):
    # Perform wave algorithm untill no further changes
    RUN = True
    while RUN:
        RUN = False
        
        # Loop over all cells and check if the current value is 0
        for row in range(HEIGHT):
            for col in range(WIDTH):
                if array[row, col] == 0:
                    # Loop over all possible directions and check if a 
                    # neighbor is already set.
                    lowest_value = math.inf
                    for direction in directions:
                        # Check if not out of bounds
                        if (row + direction[0] >= 0 and 
                            row + direction[0] <= HEIGHT - 1 and
                            col + direction[1] >= 0 and 
                            col + direction[1] <= WIDTH - 1):
                            # Check if neighbor is set
                            if array[row + direction[0], 
                                     col + direction[1]] > 1:
                                # If the checked cell has a lower value than 
                                # than the neighbor with the current lowest 
                                # value, then change the lowest_value parameter
                                if array[row + direction[0], 
                                         col + direction[1]] + 1 < lowest_value:
                                    lowest_value = array[row + direction[0], 
                                                         col + direction[1]] + 1
                                # Notify that a change has been made
                                RUN = True       
                    
                    # Change the value of the cell to the lowest value of all 
                    # surrouncing cells + 1
                    if lowest_value != math.inf:
                        array[row, col] = lowest_value

    return array

def add_obstacle_gradient(array, HEIGHT, WIDTH, directions, value_increase=1):
    # Loop over all cells and check if the object is an obstacle (value = 1)
    for row in range(HEIGHT):
        for col in range(WIDTH):
            if array[row, col] == 1:
            # Loop over all possible directions
                for direction in directions:
                    # Check if not out of bounds and not an obstacle
                    if (row + direction[0] >= 0 and 
                        row + direction[0] <= HEIGHT - 1 and
                        col + direction[1] >= 0 and 
                        col + direction[1] <= WIDTH - 1 and
                        array[row + direction[0], 
                              col + direction[1]] != 1):
                        # Add to the value
                        array[row + direction[0], 
                              col + direction[1]] += value_increase
    
    return array


def generate_path(array, start, HEIGHT, WIDTH, directions):
    # Find optimal solution starting from start position:
    
    # Set the next head to check to be the start position
    next_head = start
    snake = []
    
    # Perform while loop untill no further changes are made
    RUN = True
    while RUN:
        RUN = False
        
        # Set the next cell to check
        snake.append(next_head)
        row, col = snake[-1]
        lowest_value = math.inf

        # Loop over all possible directions and check if a 
        for direction in directions:
            # Check if not out of bounds, not an obstacle and not already
            # in snake
            if (row + direction[0] >= 0 and 
                row + direction[0] <= HEIGHT - 1 and
                col + direction[1] >= 0 and 
                col + direction[1] <= WIDTH - 1 and 
                array[row + direction[0], col + direction[1]] != 1 and
                [row + direction[0], col + direction[1]] not in snake):
                # Check if the value of the neighbor has a lower value than
                # the current lowest value.
                if (array[row + direction[0], 
                          col + direction[1]] < lowest_value and
                    array[row + direction[0], col + direction[1]] <= 
                    array[row, col]):
                    lowest_value = array[row + direction[0], 
                                          col + direction[1]]
                    next_head = [row + direction[0], 
                                  col + direction[1]]
                    RUN = True

    return array, snake
        

def get_snake(start=[24, 17], end=[113, 126], diagonals=False, 
              show_obstacle_grid=False, show_wave=False, 
              obstacle_gradient=True):
    # By default, the algorithm checks in 4 directions: left, right, up, and 
    # down. If diagonals is set to True, the diagonals are also added.
    directions = [[ 0,  1],
                  [ 1,  0],
                  [ 0, -1],
                  [-1,  0]]
    if diagonals==True:
        directions.extend([[ 1,  1],
                           [-1,  1],
                           [ 1, -1],
                           [-1, -1]])
    
    # Create an array containting all of the obstacles as ones, and free space 
    # as zeros. If show_obstacle_grid is set to True, it will be plotted. 
    obstacle_grid = get_obstacle_grid()

    if show_obstacle_grid==True:
        plt.imshow(obstacle_grid)
        plt.show()
    
    # Get the width and height of the obstacle grid.
    HEIGHT, WIDTH = obstacle_grid.shape
    
    # Set start and end position
    start = start
    obstacle_grid[end[0]][end[1]] = 2
        
    
    
    # Create start-goal gradient
    wave = generate_wave(obstacle_grid, HEIGHT, WIDTH, directions)
    
    # Add gradient around obstacle if obstacle_gradient = True
    if obstacle_gradient == True:
        wave = add_obstacle_gradient(wave, HEIGHT, WIDTH, directions, value_increase=1)
        
    if show_wave == True:
        plt.imshow(wave)
        plt.show()

    
    # Generate path
    array, snake = generate_path(wave, start, HEIGHT, WIDTH, directions)
    
    for pos in snake:
        array[pos[0], pos[1]] = np.inf

    return snake, array

def tests_for_guus(start=[0, 0], end=[9, 14], diagonals=False, 
              show_obstacle_grid=False, show_wave=False, 
              obstacle_gradient=False):
    # By default, the algorithm checks in 4 directions: left, right, up, and 
    # down. If diagonals is set to True, the diagonals are also added.
    directions = [[ 0,  1],
                  [ 1,  0],
                  [ 0, -1],
                  [-1,  0]]
    if diagonals==True:
        directions.extend([[ 1,  1],
                           [-1,  1],
                           [ 1, -1],
                           [-1, -1]])
    
    # Create an array containting all of the obstacles as ones, and free space 
    # as zeros. If show_obstacle_grid is set to True, it will be plotted. 
    obstacle_grid = simple_obstacle_grid()

    if show_obstacle_grid==True:
        plt.imshow(obstacle_grid)
        plt.show()
    
    print(obstacle_grid)
    
    # Get the width and height of the obstacle grid.
    HEIGHT, WIDTH = obstacle_grid.shape
    
    # Set start and end position
    start = start
    obstacle_grid[end[0]][end[1]] = 2
        
    
    
    # Create start-goal gradient
    wave = generate_wave(obstacle_grid, HEIGHT, WIDTH, directions)
    
    # Add gradient around obstacle if obstacle_gradient = True
    if obstacle_gradient == True:
        wave = add_obstacle_gradient(wave, HEIGHT, WIDTH, directions, value_increase=1)
        
    if show_wave == True:
        plt.imshow(wave)
        plt.show()

    print(wave)
    # Generate path
    array, snake = generate_path(wave, start, HEIGHT, WIDTH, directions)
    
    for pos in snake:
        array[pos[0], pos[1]] = np.inf

    return snake, array


def main():
    # snake, array = tests_for_guus(diagonals=True, show_obstacle_grid=False,
    #                               show_wave=False, obstacle_gradient=True)
    # print(array)

    
    snake, array = get_snake(diagonals=True, show_obstacle_grid=False,
                              show_wave=False, obstacle_gradient=True)
    
    plt.imshow(array)
    plt.show()


if __name__ == '__main__':
    print(__file__ + " start!!")
    main()
    print(__file__ + " Done!!")