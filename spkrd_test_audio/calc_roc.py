import argparse
import os

from matplotlib import pyplot
import pandas as pd

def cal_prec_recall(table):
    all_sum = table.sum().values.tolist()
    return [all_sum[0] / (all_sum[0] + all_sum[1]),
        all_sum[0] / (all_sum[0] + all_sum[3])]


def draw_eer_curve(plt, table, legend):
    precision = [1 - val[0] for val in table]
    recall = [1 - val[1] for val in table]
    plt.plot(recall, precision, label=legend)


def calc_prec_recall(table, n=100, fi=None):
    max_ll, min_ll = table[:][3].quantile(0.10), table[:][3].quantile(0.9)
    step = (max_ll - min_ll) / (n + 1)
    reg_set = set(table[:][2])
    prec_recall = [[0., 0., 0.] for _ in range(n)]
    for i in range(1, n + 1):
        if fi:
            print(f'Processing {fi}: {i} / {n}')
        cur_thrs = min_ll + i * step
        tp = fp = tn = fn = 0
        for index, ins in table.iterrows():
            if ins[1] in reg_set:
                if ins[1] == ins[2]:
                    # correctly predicted
                    if ins[3] >= cur_thrs:
                        tp += 1
                    # missed
                    else:
                        fn += 1
                # prediction in regset but wrong
                else:
                    fn += 1
            else:
                # predict nothing if not in the given set
                if ins[3] < cur_thrs:
                    tn += 1
                else:
                    fp += 1
        prec_recall[i - 1] = [tp / (tp + fp), tp / (tp + fn), cur_thrs]
    return prec_recall       


def interval_stat(table):
    quartiles = pd.cut(table[:][3], 10)
    counts = pd.value_counts(quartiles)
    print(counts)


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
        prec_recall = calc_prec_recall(table, 100, fi)
        prec_recall = pd.DataFrame(prec_recall, columns=['precision', 'recall', 'thrs'])
        prec_recall.to_csv(f'pr/{fi}', index=0, sep=' ')
