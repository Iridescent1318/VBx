#!/usr/bin/env python

# @Authors: Lukas Burget, Mireia Diez, Federico Landini, Jan Profant
# @Emails: burget@fit.vutbr.cz, mireia@fit.vutbr.cz, landini@fit.vutbr.cz, jan.profant@phonexia.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# The recipe consists in doing Agglomerative Hierachical Clustering on
# x-vectors in a first step. Then, Variational Bayes HMM over x-vectors
# is applied using the AHC output as args.initialization.
#
# A detailed analysis of this approach is presented in
# M. Diez, L. Burget, F. Landini, S. Wang, J. \v{C}ernock\'{y}
# Optimizing Bayesian HMM based x-vector clustering for the second DIHARD speech
# diarization challenge, ICASSP 2020
# A more thorough description and study of the VB-HMM with eigen-voice priors
# approach for diarization is presented in
# M. Diez, L. Burget, F. Landini, J. \v{C}ernock\'{y}
# Analysis of Speaker Diarization based on Bayesian HMM with Eigenvoice Priors,
# IEEE Transactions on Audio, Speech and Language Processing, 2019
# 
# TODO: Add new paper

import argparse
import os
import itertools

import h5py
import kaldi_io
import numpy as np
from scipy.special import softmax
from scipy.linalg import eigh

from diarization_lib import read_xvector_timing_dict, l2_norm, cos_similarity, twoGMMcalib_lin, AHC, \
    merge_adjacent_labels, mkdir_p
from kaldi_utils import read_plda
from VB_diarization import VB_diarization


def write_output(fp, out_labels, starts, ends):
    for label, seg_start, seg_end in zip(out_labels, starts, ends):
        fp.write(f'SPEAKER {file_name} 1 {seg_start:03f} {seg_end - seg_start:03f} '
                 f'<NA> <NA> {label + 1 if isinstance(label, int) else label} <NA> <NA>{os.linesep}')


def cross_interval_len(inv1, inv2):
    return max(0, min(inv1[1], inv2[1]) - max(inv1[0], inv2[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-name', required=True, type=str)
    parser.add_argument('--init', required=True, type=str, choices=['AHC', 'AHC+VB'],
                        help='AHC for using only AHC or AHC+VB for VB-HMM after AHC initilization', )
    parser.add_argument('--out-rttm-dir', required=True, type=str, help='Directory to store output rttm files')
    parser.add_argument('--xvec-ark-file', required=True, type=str,
                        help='Kaldi ark file with x-vectors from one or more input recordings. '
                             'Attention: all x-vectors from one recording must be in one ark file')
    parser.add_argument('--segments-file', required=True, type=str,
                        help='File with x-vector timing info (see diarization_lib.read_xvector_timing_dict)')
    parser.add_argument('--xvec-transform', required=True, type=str,
                        help='path to x-vector transformation h5 file')
    parser.add_argument('--plda-file', required=True, type=str,
                        help='File with PLDA model in Kaldi format used for AHC and VB-HMM x-vector clustering')
    parser.add_argument('--threshold', required=True, type=float, help='args.threshold (bias) used for AHC')
    parser.add_argument('--lda-dim', required=True, type=int,
                        help='For VB-HMM, x-vectors are reduced to this dimensionality using LDA')
    parser.add_argument('--Fa', required=True, type=float,
                        help='Parameter of VB-HMM (see VB_diarization.VB_diarization)')
    parser.add_argument('--Fb', required=True, type=float,
                        help='Parameter of VB-HMM (see VB_diarization.VB_diarization)')
    parser.add_argument('--loopP', required=True, type=float,
                        help='Parameter of VB-HMM (see VB_diarization.VB_diarization)')
    parser.add_argument('--init-smoothing', required=False, type=float, default=5.0,
                        help='AHC produces hard assignments of x-vetors to speakers. These are "smoothed" to soft '
                             'assignments as the args.initialization for VB-HMM. This parameter controls the amount of'
                             ' smoothing. Not so important, high value (e.g. 10) is OK  => keeping hard assigment')
    parser.add_argument('--output-2nd', required=False, type=bool, default=False,
                        help='Output also second most likely speaker of VB-HMM')
    parser.add_argument('--reg-seg-file', required=False, type=str, default=None,
                        help='File with registered segments, including start, end time and speaker label')
    parser.add_argument('--fusion-factor', required=False, type=float, default=0.0,
                        help='The hyperparameter controlling fusion of xvecs')
    parser.add_argument('--replace-label', required=False, type=bool, default=False)

    args = parser.parse_args()
    assert 0 <= args.loopP <= 1, f'Expecting loopP between 0 and 1, got {args.loopP} instead.'

    # segments file with x-vector timing information
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
        for i, rs in enumerate(reg_segs):
            rs[1] = rs[1].astype(float) + rs[0].astype(float)
    else:
        print("Info: registered segments are not given")

    # Open ark file with x-vectors and in each iteration of the following for-loop
    # read a batch of x-vectors corresponding to one recording
    arkit = kaldi_io.read_vec_flt_ark(args.xvec_ark_file)
    recit = itertools.groupby(arkit, lambda e: e[0].rsplit('_', 1)[0]) # group xvectors in ark by recording name
    for file_name, segs in recit:
        print(file_name)
        seg_names, xvecs = zip(*segs)
        x = np.array(xvecs)

        if args.reg_seg_file:
            reg_label = np.array([' '] * x.shape[0])
            for i, reg_and_seg in enumerate(zip(reg_label, seg_names)):
                cur_label_dur = seg_to_duration_dict[reg_and_seg[1]]
                max_dur = 0
                max_label = ' '
                for given_reg in reg_segs:
                    cross_dur = cross_interval_len(
                        [given_reg[0].astype(float), given_reg[1].astype(float)],
                        [cur_label_dur[0], cur_label_dur[1]]
                    )
                    if cross_dur > max_dur and cross_dur > 1e-3:
                        max_dur = cross_dur
                        max_label = given_reg[2]
                reg_label[i] = max_label
        else:
            reg_label = None

        with h5py.File(args.xvec_transform, 'r') as f:
            mean1 = np.array(f['mean1'])
            mean2 = np.array(f['mean2'])
            lda = np.array(f['lda'])
            x = l2_norm(lda.T.dot((l2_norm(x - mean1)).transpose()).transpose() - mean2)

        if args.init == 'AHC' or args.init.endswith('VB'):
            if args.init.startswith('AHC'):
                # Kaldi-like AHC of x-vectors (scr_mx is matrix of pairwise
                # similarities between all x-vectors)
                scr_mx = cos_similarity(x)
                # Figure out utterance specific args.threshold for AHC.
                thr, junk = twoGMMcalib_lin(scr_mx.ravel())
                # output "labels" is an integer vector of speaker (cluster) ids
                labels1st = AHC(scr_mx, thr + args.threshold)
            if args.init.endswith('VB'):
                # Smooth the hard labels obtained from AHC to soft assignments
                # of x-vectors to speakers
                qinit = np.zeros((len(labels1st), np.max(labels1st) + 1))
                qinit[range(len(labels1st)), labels1st] = 1.0
                qinit = softmax(qinit * args.init_smoothing, axis=1)
                fea = (x - plda_mu).dot(plda_tr.T)[:, :args.lda_dim]
                # Use VB-HMM for x-vector clustering. Instead of i-vector extractor model, we use PLDA
                # => GMM with only 1 component, V derived accross-class covariance,
                # and iE is inverse within-class covariance (i.e. identity)
                sm = np.zeros(args.lda_dim)
                siE = np.ones(args.lda_dim)
                sV = np.sqrt(plda_psi[:args.lda_dim])
                q, sp, L = VB_diarization(
                    fea, sm, np.diag(siE), np.diag(sV),
                    pi=None, gamma=qinit, maxSpeakers=qinit.shape[1],
                    maxIters=40, epsilon=1e-6, 
                    loopProb=args.loopP, Fa=args.Fa, Fb=args.Fb,
                    label=reg_label, fusionFactor=args.fusion_factor, 
                    plda_psi=plda_psi)

                labels1st = np.argsort(-q, axis=1)[:, 0]
                spk_to_clus_lab = dict()
                final_spk_to_clus_lab = dict()
                if reg_label is not None and args.replace_label:
                    for i, spk_with_clus_lab in enumerate(zip(reg_label, labels1st)):
                        if spk_with_clus_lab[0] == ' ':
                            continue
                        if spk_with_clus_lab[0] not in spk_to_clus_lab:
                            spk_to_clus_lab[spk_with_clus_lab[0]] = {spk_with_clus_lab[1]: 1}
                        else:
                            if spk_with_clus_lab[1] not in spk_to_clus_lab[spk_with_clus_lab[0]]:
                                spk_to_clus_lab[spk_with_clus_lab[0]][spk_with_clus_lab[1]] = 1
                            else:
                                spk_to_clus_lab[spk_with_clus_lab[0]][spk_with_clus_lab[1]] += 1
                    for key, value in spk_to_clus_lab.items():
                        max_spk = None
                        max_spk_num = -1
                        for k, v in spk_to_clus_lab[key].items():
                            if max_spk is None:
                                max_spk = k
                                max_spk_num = v
                            else:
                                if v > max_spk_num:
                                    max_spk = k
                                    max_spk_num = v
                        if max_spk not in final_spk_to_clus_lab:
                            final_spk_to_clus_lab[max_spk] = key
                        else:
                            raise Exception("Naming conflict during alignment of speakers to clustering labels")
                    labels1st_str = [''] * len(labels1st)
                    for i, l in enumerate(labels1st):
                        if labels1st[i] in final_spk_to_clus_lab:
                            labels1st_str[i] = final_spk_to_clus_lab[labels1st[i]]
                        else:
                            labels1st_str[i] = str(labels1st[i])
                    labels1st_str = np.array(labels1st_str)
                else:
                    labels1st_str = labels1st
                    
                if q.shape[1] > 1:
                    labels2nd = np.argsort(-q, axis=1)[:, 1]
        else:
            raise ValueError('Wrong option for args.initialization.')

        assert(np.all(segs_dict_old[file_name][0] == np.array(seg_names)))
        start, end = segs_dict_old[file_name][1].T

        starts, ends, out_labels = merge_adjacent_labels(start, end, labels1st_str)
        mkdir_p(args.out_rttm_dir)
        with open(os.path.join(args.out_rttm_dir, f'{file_name}.rttm'), 'w') as fp:
            write_output(fp, out_labels, starts, ends)

        if args.output_2nd and args.init.endswith('VB') and q.shape[1] > 1:
            starts, ends, out_labels2 = merge_adjacent_labels(start, end, labels2nd)
            output_rttm_dir = f'{args.out_rttm_dir}2nd'
            mkdir_p(output_rttm_dir)
            with open(os.path.join(output_rttm_dir, f'{file_name}.rttm'), 'w') as fp:
                write_output(fp, out_labels2, starts, ends)
