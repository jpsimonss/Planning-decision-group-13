## Get multiple max differences

import numpy as np
from mpc_steer_speed_us import multiple_max_diff

def main():
    
    iterations = 100
    max_dif_normal = np.zeros(iterations)
    
    for i in range(iterations):
        print("iteration ", i, "started")
        max_diff = multiple_max_diff()
        max_dif_normal[i] = max_diff

    print(max_dif_normal)


if __name__ == '__main__':
    main()
