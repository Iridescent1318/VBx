import numpy as np
import argparse
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Randomly select C samples from 0 to N-1 without putting back, where 0<C<=N')
    parser.add_argument('-n', help='Total number N')
    parser.add_argument('-c', help='Select number C')
    args = parser.parse_args()

    result_list = random.sample(range(int(args.n)), int(args.c))
    output_str = ''
    for i in range(len(result_list)):
        output_str = output_str + f'{result_list[i]}'
        if i < len(result_list) - 1:
            output_str = output_str + ' '
    print(output_str)
