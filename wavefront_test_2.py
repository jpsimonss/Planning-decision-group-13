import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image

def simple_obstacle_grid(GRIDSIZE=0, threshold=0):
    obstacle_grid = np.zeros((10, 15))
    obstacle_grid[4:8, 6:10] = 1
    return obstacle_grid

def get_obstacle_grid(GRIDSIZE=3, threshold=.01):
    # Open image and convert to numpy array
        ### SELECT ONE ###
    # image = Image.open('supermarket3.jpeg')             # Real supermarket
    # image = Image.open('supermarkets/straight.jpg')     # Test with straight shelves
    # image = Image.open('supermarkets/rotated.jpg')      # Test with rotated shelves
    # image = Image.open('supermarkets/circles.jpg')      # Test with oval shapes
    image = Image.open('supermarkets/convex.jpg')      # Test with convex star shapes
    # image = Image.open('supermarkets/doolhof.jpg')      # Test with maze

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

def make_configuration_space(obstacle_grid, directions, size=1):
    HEIGHT, WIDTH = np.shape(obstacle_grid)
    
    new_grid = np.zeros((HEIGHT, WIDTH))
    configuration_space = np.copy(obstacle_grid)
    
    while size != 0:
        for row in range(HEIGHT):
            for col in range(WIDTH):
                if configuration_space[row, col] == 1:
                # Loop over all possible directions
                    for direction in directions:
                        # Check if not out of bounds and not an obstacle
                        if (row + direction[0] >= 0 and 
                            row + direction[0] <= HEIGHT - 1 and
                            col + direction[1] >= 0 and 
                            col + direction[1] <= WIDTH - 1 and
                            configuration_space[row + direction[0], 
                                                col + direction[1]] != 1):
                            # Add to the value
                            new_grid[row + direction[0], 
                                     col + direction[1]] += 1
        
        new_grid[new_grid!=0] = 1
        configuration_space += new_grid
        size -= 1
    return configuration_space

def random_start_end(configuration_space):
    i = 0
    while i < 2:
        randomRow = np.random.randint(configuration_space.shape[0], size=1)
        randomColumn = np.random.randint(configuration_space.shape[1], size=1)
        if configuration_space[randomRow, randomColumn] == 0:
            if i == 0:
                start = [randomRow[0], randomColumn[0]]
                i = i+1
            elif i == 1:
                end = [randomRow[0], randomColumn[0]]
                i = i+1
    return start, end

# def get_obstacle_gradient(obstacle_grid, directions, size=1, value_increase=1):
#     # Loop over all cells and check if the object is an obstacle (value = 1)    
#     HEIGHT, WIDTH = np.shape(obstacle_grid)
    
#     new_grid = np.zeros((HEIGHT, WIDTH))
#     obstacle_gradient = np.copy(obstacle_grid)
    
#     while size != 0:
#         for row in range(HEIGHT):
#             for col in range(WIDTH):
#                 if obstacle_gradient[row, col] != 0:
#                 # Loop over all possible directions
#                     for direction in directions:
#                         # Check if not out of bounds and not an obstacle
#                         if (row + direction[0] >= 0 and 
#                             row + direction[0] <= HEIGHT - 1 and
#                             col + direction[1] >= 0 and 
#                             col + direction[1] <= WIDTH - 1 and
#                             obstacle_gradient[row + direction[0], 
#                                               col + direction[1]] != 1):
#                             # Add to the value
#                             new_grid[row + direction[0], 
#                                      col + direction[1]] += 1
        
#         new_grid[new_grid!=0] = value_increase
#         obstacle_gradient += new_grid
#         size -= 1
    
#     obstacle_gradient -= obstacle_grid
#     return obstacle_gradient

def generate_path(array, start, end, directions):
    HEIGHT, WIDTH = np.shape(array)
    
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
            # in snake and not the end position
            if (row + direction[0] >= 0 and 
                row + direction[0] <= HEIGHT - 1 and
                col + direction[1] >= 0 and 
                col + direction[1] <= WIDTH - 1 and 
                array[row + direction[0], col + direction[1]] != 1 and
                [row + direction[0], col + direction[1]] not in snake and
                [row, col] != end):
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
    
    local_min = False
    if snake[-1] != end:
        local_min = True
    return array, snake, local_min
        

def get_snake(diagonals=True, 
              show_obstacle_grid=False, show_wave=False, 
              show_configuration_space = False, configuration_size=2):
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

    configuration_space = make_configuration_space(np.copy(obstacle_grid), directions, size=configuration_size)
    
    # Set start and end position
    start, end = random_start_end(configuration_space)
    configuration_space[end[0]][end[1]] = 2

    if show_configuration_space==True:
        plt.imshow(configuration_space)
        plt.show()

    # Create start-goal gradient
    wave = generate_wave(np.copy(configuration_space), HEIGHT, WIDTH, directions)
    
    # # Add gradient around obstacle if obstacle_gradient = True
    # if obstacle_gradient == True:
    #     new_wave += get_obstacle_gradient(obstacle_grid, directions, size=obstacle_gradient_size, value_increase = obstacle_gradient_value_increase)
        
    if show_wave == True:
        plt.imshow(wave)
        plt.show()

    # Generate path
    array, snake, local_min = generate_path(np.copy(wave), start, end, directions)

    for pos in snake:
        obstacle_grid[pos[0], pos[1]] = np.inf

    return snake, obstacle_grid, configuration_space


def main():
    # snake, array = tests_for_guus(diagonals=True, show_obstacle_grid=False,
    #                               show_wave=False, obstacle_gradient=True)
    # print(array)

    
    snake, array, configuration_space = get_snake(diagonals=True, 
              show_obstacle_grid=False, show_wave=False, 
              show_configuration_space=False, configuration_size=1)
    
    plt.imshow(array)
    plt.show()


if __name__ == '__main__':
    print(__file__ + " start!!")
    main()
    print(__file__ + " Done!!")