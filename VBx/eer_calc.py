import h5py
import kaldi_io
import argparse
import os
import itertools
import numpy as np
from scipy.linalg import eigh

from diarization_lib import read_xvector_timing_dict
from diarization_lib import l2_norm
from kaldi_utils import read_plda


def cross_interval_len(inv1, inv2):
    return max(0, min(inv1[1], inv2[1]) - max(inv1[0], inv2[0]))


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


def normal_pdf_log(x, mean, diag_cov):
    '''
    Compute the log value of a normal distribution given mean and diagonal covariance matrix

    Args: 
        x: log value of the PDF to be computed at x
        mean: mean of the normal distribution
        diag_cov: diagonal covariance matrix of the normal distribution

    Returns:
        the log value of a normal distribution given mean and diagonal covariance matrix
    '''
    dim = x.shape[0]
    diag_cov_det = np.abs(np.prod(diag_cov, axis=0))
    return - dim / 2 * np.log(2 * np.pi) - 1 / 2 * np.log(diag_cov_det) - np.dot((x - mean) / diag_cov, x - mean)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-name', required=True, type=str)
    parser.add_argument(
        '-i', '--in-rttm-dir', help='Directory of .rttm files to process', type=str)
    parser.add_argument('--rttm-file', required=True, type=str)
    parser.add_argument('--xvec-ark-file', required=True, type=str,
                        help='Kaldi ark file with x-vectors from one or more input recordings. '
                             'Attention: all x-vectors from one recording must be in one ark file')
    parser.add_argument('--segments-file', required=True, type=str,
                        help='File with x-vector timing info (see diarization_lib.read_xvector_timing_dict)')
    parser.add_argument('--xvec-transform', required=True, type=str,
                        help='path to x-vector transformation h5 file')
    parser.add_argument('--plda-file', required=True, type=str,
                        help='File with PLDA model in Kaldi format used for AHC and VB-HMM x-vector clustering')
    parser.add_argument('--lda-dim', required=True, type=int,
                        help='For VB-HMM, x-vectors are reduced to this dimensionality using LDA')
    parser.add_argument('--reg-seg-file', required=False, type=str, default=None,
                        help='File with registered segments, including start, end time and speaker label')
    parser.add_argument('--output-file', required=True, type=str, help='output file')

    args = parser.parse_args()

    # read segment files with x-vector timing information
    segs_dict_old = read_xvector_timing_dict(args.segments_file)
    segs_dict = segs_dict_old[args.file_name]
    seg_to_duration_dict = dict()
    for seg, dur in zip(segs_dict[0], segs_dict[1]):
        seg_to_duration_dict[seg] = dur

    kaldi_plda = read_plda(args.plda_file)
    plda_mu, plda_tr, plda_psi = kaldi_plda
    W = np.linalg.inv(plda_tr.T.dot(plda_tr))
    B = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
    acvar, wccn = eigh(B, W)
    plda_psi = acvar[::-1]
    plda_tr = wccn.T[::-1]

    # read registered segments from .regseg file
    if args.reg_seg_file:
        reg_segs = np.loadtxt(args.reg_seg_file, dtype=str)
        if len(reg_segs.shape) == 1:
            reg_segs = reg_segs.reshape((1, reg_segs.shape[0]))
        if reg_segs.size:
            for i, rs in enumerate(reg_segs):
                rs[1] = rs[1].astype(float) + rs[0].astype(float)
    else:
        print("Info: registered segments are not given")

    # Open ark file with x-vectors and in each iteration of the following for-loop
    # read a batch of x-vectors corresponding to one recording
    arkit = kaldi_io.read_vec_flt_ark(args.xvec_ark_file)
    # group xvectors in ark by recording name
    recit = itertools.groupby(arkit, lambda e: e[0].rsplit('_', 1)[0])
    for file_name, segs in recit:
        print(file_name)
        seg_names, xvecs = zip(*segs)
        x = np.array(xvecs)
        test_start_pos = 0
        reg_label_set = set()

        # prepare registered segments
        if args.reg_seg_file and reg_segs.size:
            reg_label = [" "] * x.shape[0]
            for i, reg_and_seg in enumerate(zip(reg_label, seg_names)):
                cur_label_dur = seg_to_duration_dict[reg_and_seg[1]]
                select_label_dur = []
                for given_reg in reg_segs:
                    cross_dur = cross_interval_len(
                        [given_reg[0].astype(float), given_reg[1].astype(float)],
                        [cur_label_dur[0], cur_label_dur[1]]
                    )
                    select_label_dur.append((given_reg[2], cross_dur))
                select_label_dur.sort(key=lambda x: -x[1])
                if select_label_dur[0][1] >= 1e-2:
                    # overlap
                    if select_label_dur[1][1] + select_label_dur[0][1] > 1.44:
                        if abs(select_label_dur[1][1] - select_label_dur[0][1]) < 0.72:
                            reg_label[i] = "#"
                        else:
                            reg_label[i] = select_label_dur[0][0]
                    else:
                        if select_label_dur[0][1] > 0.72:
                            reg_label[i] = select_label_dur[0][0]
                        else:
                            reg_label[i] = "#"
                if reg_label[i] != " " and reg_label[i] != "#":
                    if reg_label[i] not in reg_label_set:
                        reg_label_set.add(reg_label[i])
            for i in range(len(reg_label) - 1, 0, -1):
                if reg_label[i - 1] != " ":
                    test_start_pos = i
                    break
            
        else:
            reg_label = None

        if args.rttm_file:
            rttm_content = read_rttm(args.file_name, args.in_rttm_dir)
            for i, rc in enumerate(rttm_content):
                rc[4] = f'{rc[3].astype(float) + rc[4].astype(float): .2f}'
            rttm_content = rttm_content[:, [3, 4, 7]]
            sys_label = [" "] * x.shape[0]
            for i, reg_and_seg in enumerate(zip(sys_label, seg_names)):
                cur_label_dur = seg_to_duration_dict[reg_and_seg[1]]
                select_label_dur = []
                for given_reg in rttm_content:
                    cross_dur = cross_interval_len(
                        [given_reg[0].astype(float), given_reg[1].astype(float)],
                        [cur_label_dur[0], cur_label_dur[1]]
                    )
                    select_label_dur.append((given_reg[2], cross_dur))
                select_label_dur.sort(key=lambda x: -x[1])
                if select_label_dur[0][1] >= 1e-2:
                    # overlap
                    if select_label_dur[1][1] + select_label_dur[0][1] > 1.44:
                        if abs(select_label_dur[1][1] - select_label_dur[0][1]) < 0.72:
                            sys_label[i] = "#"
                        else:
                            sys_label[i] = select_label_dur[0][0]
                    else:
                        if select_label_dur[0][1] > 0.72:
                            sys_label[i] = select_label_dur[0][0]
                        else:
                            sys_label[i] = "#"
            sys_label = sys_label[test_start_pos:]

        with h5py.File(args.xvec_transform, 'r') as f:
            mean1 = np.array(f['mean1'])
            mean2 = np.array(f['mean2'])
            lda = np.array(f['lda'])
            x = l2_norm(lda.T.dot((l2_norm(x - mean1)).transpose()).transpose() - mean2)

        fea = (x - plda_mu).dot(plda_tr.T)[:, :args.lda_dim]

        if reg_label is not None:
            D = fea.shape[1]  # feature dimensionality
            assert fea.shape[0] == len(reg_label), 'Error: segment shape is not equal to labels'
            registered_frames = dict()
            for i, frame in enumerate(fea):
                if not (reg_label[i] == " " or reg_label[i] == "#"):
                    if reg_label[i] not in registered_frames:
                        registered_frames[reg_label[i]] = [frame]
                    else:
                        registered_frames[reg_label[i]].append(frame)
            reg_frames_mean = dict()
            reg_frames_num = dict()
            reg_frames_plda_mean = dict()
            reg_frames_plda_cov = dict()
            for key, val in registered_frames.items():
                reg_frames_mean[key] = np.sum(val, axis=0) / len(val)
                reg_frames_num[key] = len(val)
                n = len(val)
                reg_frames_plda_mean[key] = n * plda_psi / (n * plda_psi + np.ones(D)) * reg_frames_mean[key]
                reg_frames_plda_cov[key] = np.ones(D) + plda_psi / (n * plda_psi + np.ones(D))

            reg_label = reg_label[test_start_pos:]
            max_ll = [0.0] * len(reg_label)
            
            for i, frame in enumerate(fea[test_start_pos:]):
                plda_sim = dict()
                for key, val in reg_frames_mean.items():
                    plda_sim[key] = normal_pdf_log(frame, reg_frames_plda_mean[key], reg_frames_plda_cov[key])
                reg_label[i], max_ll[i] = max(plda_sim.items(), key=lambda e: e[1])[0], \
                                        max(plda_sim.items(), key=lambda e: e[1])[1]

        with open(args.output_file, 'a') as f:
            for p in zip(sys_label, reg_label, max_ll):
                # if p[1] == "#" or p[1] == " ":
                #     continue
                # if p[1] in reg_label_set:
                #     # y_pred == y_true
                #     if p[0] == p[1]:
                #         tp += 1
                #     # y_true in reg_labels but y_pred != y_true
                #     else:
                #         fn += 1
                # else:
                #     # y_true not in reg_labels, neither is y_pred
                #     # y_pred can only be ' ' or '#'
                #     if p[0] not in reg_label_set:
                #         tn += 1
                #     # y_true not in reg_labels, but y_pred is in
                #     else:
                #         fp += 1
                if p[0] == "#" or p[0] == " ":
                    continue
                output_str = f'{args.file_name} {p[0]} {p[1]} {p[2]: .2f}' if p[1] != " " else \
                    f'{args.file_name} {p[0]} [UNKNOWN] {p[2]: .2f}'
                f.write(output_str + '\n')
        