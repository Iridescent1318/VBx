import argparse
import os

import pandas as pd

def cal_prec_recall(table):
    all_sum = table.sum().values.tolist()
    return [all_sum[0] / (all_sum[0] + all_sum[1]),
        all_sum[0] / (all_sum[0] + all_sum[3])]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--thrs-dir', type=str, default='thrs')

    args = parser.parse_args()

    files = [f for f in os.listdir(args.thrs_dir) if \
        os.path.isfile(os.path.join(args.thrs_dir, f))]

    pr_dict = {}
    for fi in files:
        file_part = fi.split('_')
        table = pd.read_csv(
            os.path.join(args.thrs_dir, fi),
            sep=' ',
            header=None
        )
        if f'{file_part[0]}_{file_part[1]}' not in pr_dict:
            pr_dict[f'{file_part[0]}_{file_part[1]}'] = [cal_prec_recall(table)]
        else:
            pr_dict[f'{file_part[0]}_{file_part[1]}'].append(cal_prec_recall(table))
    
    for k, v in pr_dict.items():
        v.sort(key=lambda x: x[1])
        print(f'{k}: {v}')
        print('\n')
