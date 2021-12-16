import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image



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


def generate_wave(array, HEIGHT, WIDTH):
    RUN = True
    # Perform wave algorithm
    while RUN:
        # Loop over all cells and perform wave
        RUN = False
        for row in range(HEIGHT):
            for col in range(WIDTH):
                if array[row, col] == 0:
    
                    # Test up direction
                    if row != 0:
                        if array[row - 1, col] > 1:
                            array[row, col] = array[row - 1, col] + 1
                            RUN = True
                       
                    # Test down direction
                    if row != HEIGHT - 1:
                            if array[row + 1, col] > 1:
                                array[row, col] = array[row + 1, col] + 1
                                RUN = True
                                
                    # Test left direction
                    if col != 0:
                        if array[row, col - 1] > 1:
                            array[row, col] = array[row, col - 1] + 1
                            RUN = True
                       
                    # Test right direction
                    if col != WIDTH - 1:
                            if array[row, col + 1] > 1:
                                array[row, col] = array[row, col + 1] + 1
                                RUN = True    

    return array


def generate_path(array, start, HEIGHT, WIDTH):
    # Find optimal solution starting from start position:
    lowest_value = math.inf
    next_head = start
    snake = []
    
    RUN = True
    # Look around snake head for next lowest number
    while RUN:
        RUN = False
        # Loop over all directions
        
        snake.append(next_head)
        row, col = snake[-1]

    
        # Test up direction
        if row != 0:
            if array[row - 1, col] != 1:
                if (array[row - 1, col] < lowest_value  
                    and array[row - 1, col] < array[row, col]):
                    lowest_value = array[row - 1, col]
                    next_head = [row - 1, col]
                    RUN = True
        
        # Test down direction
        if row != HEIGHT - 1:
            if array[row + 1, col] != 1:
                if (array[row + 1, col] < lowest_value  
                    and array[row + 1, col] < array[row, col]):
                    lowest_value = array[row + 1, col]
                    next_head = [row + 1, col]
                    RUN = True
                
        # Test left direction
        if col != 0:
            if array[row - 1, col] != 1:
                if (array[row, col - 1] < lowest_value  
                    and array[row, col - 1] < array[row, col]):
                    lowest_value = array[row - 1, col]
                    next_head = [row, col - 1]
                    RUN = True
        
        # Test right direction
        if col != WIDTH - 1:
            if array[row, col + 1] != 1:
                if (array[row, col + 1] < lowest_value 
                    and array[row, col + 1] < array[row, col]):
                    lowest_value = array[row, col + 1]
                    next_head = [row, col + 1]
                    RUN = True
        
    return array, snake
        

def get_snake(start=[24, 17], end=[85, 117], diagonals=False):
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
    # as zeros
    array = get_obstacle_grid()

    # plt.imshow(array)
    # plt.show()
    
    # Create empty array
    HEIGHT, WIDTH = array.shape
    
    # Set start and end position
    start = [24, 17]
    array[85, 117] = 2
    
    array = generate_wave(array, HEIGHT, WIDTH)
    
    array, snake = generate_path(array, start, HEIGHT, WIDTH)
    
    for pos in snake:
        array[pos[0], pos[1]] = np.inf

    return snake, array


def main():
    snake, array = get_snake(diagonals=True)
    print(snake)
    print(array)
    
    # plt.imshow(array)
    # plt.show()


if __name__ == '__main__':
    print(__file__ + " start!!")
    main()
    print(__file__ + " Done!!")