import argparse
import os
import json

import numpy as np

def merge(intervals):
    '''
    Given time intervals of speech, merge the intervals which overlap with
    others.

    Args:
        intervals: List (or array-like) of intervals to merge. Each interval
        should be a list (or array-like) of starting and ending time

    Returns:
        merge_list: List of merged intervals
        is_overlap: Boolean list with the same length of the old interval list.
            If the i-th interval overlaps with its former interval, then
            is_overlap[i] = True.
    '''
    n = len(intervals)
    intervals.sort(key=lambda x: x[0])

    merge_list = []
    is_overlap = [False] * n
    for i in range(n):
        if not merge_list or merge_list[-1][1] < intervals[i][0]:
            merge_list.append(intervals[i])
        else:
            is_overlap[i] = True
            merge_list[-1][1] = max(merge_list[-1][1], intervals[i][1])

    return merge_list, is_overlap


def read_rttm(file_name: str, input_dir: str):
    '''
    Read .rttm file named file_name in input_dir.
    '''
    rttm_content = np.loadtxt(
        os.path.join(input_dir, file_name + '.rttm'), dtype=np.str)
    if not rttm_content.shape[0]:
        raise Exception("Empty file")
    rttm_content[:, 3:5] = rttm_content[:, 3:5].astype(float)
    return rttm_content


def get_rttm_metadata(rttm_content):
    '''
    To be continued
    NOT GONNA USE THIS NOW
    '''
    rttm_duration = sum(rttm_content[-1, 3:5].astype(float))
    rttm_spkr_name = dict()
    for index, rttm_row in enumerate(rttm_content):
        if rttm_row[7] not in rttm_spkr_name:
            rttm_spkr_name[rttm_row[7]] = [(
                index, (rttm_row[3].astype(float), 
                rttm_row[3].astype(float) + rttm_row[4].astype(float))
            )]
        else:
            rttm_spkr_name[rttm_row[7]].append((
                index, (rttm_row[3].astype(float), 
                rttm_row[3].astype(float) + rttm_row[4].astype(float))
            ))
    rttm_metadata = dict()
    rttm_metadata['duration'] = rttm_duration
    rttm_metadata['speakers'] = rttm_spkr_name
    return rttm_metadata


def select_register_segments(rttm_content, start, end):
    '''
    To be continued
    '''
    time_dur = end - start
    rttm_start = rttm_content[0, 3].astype(float)
    rttm_duration = sum(rttm_content[-1, 3:5].astype(float))
    start = max(start, rttm_start)
    end = min(start + time_dur, rttm_duration)
    assert 0 <= start < end, 'Start is greater than end'
    rttm_reg_seg = []
    for index, rttm_row in enumerate(rttm_content):
        if rttm_row[3].astype(float) < end and \
            rttm_row[4].astype(float) + rttm_row[3].astype(float) > start:
            rttm_reg_seg.append((rttm_row[3:5].tolist(), rttm_row[7]))
        if float(rttm_reg_seg[-1][0][0]) + float(rttm_reg_seg[-1][0][1]) > end:
            rttm_reg_seg[-1][0][1] = f'{end - float(rttm_reg_seg[-1][0][0]): .2f}'
    return rttm_reg_seg


def output_register_segments(file_name, rttm_reg_seg, output_dir):
    '''
    To be continued
    '''
    with open(os.path.join(output_dir, file_name + '.regseg'), 'w') as f_regseg:
        for regseg in rttm_reg_seg:
            output_str = f'{regseg[0][0]} {regseg[0][1]} {regseg[1]}'
            f_regseg.write(output_str + '\n')


def check_overlap(rttm_content, need_time_proc):
    '''
    Check if the rttm contents has overlap.

    Args:
        rttm_content: rttm contents got by read_rttm()
        need_time_proc: if true, the durations of speech will be substituted
            with end time, i.e., if the interval [start_time, duration] is
            given and this arg is set true, then it will return [start_time,
            start_time + duration]

    Returns:
        new_rttm_content_time: list of merged intervals
        rttm_content_overlap: Boolean list with the same length of the old
            interval list. If the i-th interval overlaps with its former
            interval, then is_overlap[i] = True.
    '''
    start_read = 0
    for i, r_content in enumerate(rttm_content):
        if r_content[0] == '<NA>':
            continue
        start_read = i
        break
    rttm_content_time = rttm_content[i:, 3:5].astype(float)
    start_offset = rttm_content_time[0][0]
    for i, r_content_time in enumerate(rttm_content_time):
        r_content_time[0] = r_content_time[0] - start_offset
        if need_time_proc:
            r_content_time[1] = \
            r_content_time[0] + r_content_time[1]
    new_rttm_content_time, rttm_content_overlap = merge(
        rttm_content_time.tolist())
    return new_rttm_content_time, rttm_content_overlap


def output_overlaps(file_name, rttm_content, rttm_content_overlap, output_dir):
    '''
    Create .overlap files, which add an column based on .rttm files to show
    whether some interval overlaps with its former interval

    Args:
        file_name: name of .rttm file
        rttm_content: rttm contents got by read_rttm()
        rttm_content_overlap: Boolean list with the same length of the old
            interval list. If the i-th interval overlaps with its former
            interval, then is_overlap[i] = True.
        output_dir: output directory of .overlap file
    '''
    with open(os.path.join(output_dir, file_name + '.overlap'), 'w') as f_overlap:
        for i, r_content in enumerate(rttm_content):
            output_str = ''
            for c in r_content:
                output_str = output_str + f'{c} '
            output_str = output_str + 'OVERLAPPED' if \
                rttm_content_overlap[i] else output_str + 'NO_OVERLAP'
            f_overlap.write(output_str + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Transform .rttm files into needed .lab files')
    parser.add_argument(
        '-i', '--in-rttm-dir', help='Directory of .rttm files to process')
    parser.add_argument(
        '-o', '--out-dir', help='Output directory of .overlap files')
    parser.add_argument(
        '-a', '--action', help='Actions: mark \
        overlappings (overlap) or select registered segments (regseg)')
    parser.add_argument(
        '-s', '--start', type=float, help='register start', default=0)
    parser.add_argument(
        '-e', '--end', type=float, help='register end', default=15.0)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    file_list = os.listdir(args.in_rttm_dir)
    name_list = []

    for f in file_list:
        if os.path.splitext(f)[1] == '.rttm':
            name_list.append(os.path.splitext(f)[0])
    if args.action == 'overlap':
        for fname in name_list:
            content = read_rttm(fname, args.in_rttm_dir)
            content_time, content_overlap = check_overlap(
                content, need_time_proc=True)
            output_overlaps(
                fname, content, content_overlap, args.out_dir)
    elif args.action == 'metadata':
        for fname in name_list:
            with open (os.path.join(args.out_dir, fname + '.meta'), 'w') as f_metadata:
                content = read_rttm(fname, args.in_rttm_dir)
                metadata = get_rttm_metadata(content)
                f_metadata.write(json.dumps(metadata) + '\n')
    elif args.action == 'regseg':
        for fname in name_list:
            content = read_rttm(fname, args.in_rttm_dir)
            reg_seg = select_register_segments(content, args.start, args.end)
            output_register_segments(fname, reg_seg, args.out_dir)
