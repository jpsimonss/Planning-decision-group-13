## Get multiple max differences

import numpy as np
from mpc_steer_speed_us import multiple_max_diff
from wavefront_test_2 import get_snake
import time

def main():
    
    iterations = 17
    max_dif_normal = np.zeros(iterations)
    
    for i in range(iterations):
        print("iteration ", i, "started")
        max_diff = multiple_max_diff()
        max_dif_normal[i] = max_diff

    print(max_dif_normal)

def main2():
    
    iterations = 100
    elapsed_time = np.zeros(iterations)
    
    for i in range(iterations):
        print("iteration ", i, "started")
        start = time.time()
        snake, array, configuration_space = get_snake(configuration_size=2)
        end = time.time()
        elapsed_time[i] = end-start

    print(elapsed_time)

    print("mean =", np.mean(elapsed_time))
    print("std =", np.std(elapsed_time))




if __name__ == '__main__':
    # main()
    main2()





