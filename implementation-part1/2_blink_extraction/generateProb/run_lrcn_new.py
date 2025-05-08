"""
Modified the original code, by adding post-process on final data, to match with the desired form of BPD dataset
by Muhammad Salihin Saealal - UMP (PhD Candidate)

In Ictu Oculi: original code by
Yuezun Li, Ming-ching Chang and Siwei Lyu
"""
# import tensorflow as tf
import argparse
import numpy as np
import sys
sys.path.append('..')
from solu_base import Solu
from blink_net import BlinkLRCN
from solver import Solver
import cv2
from py_utils import x_utils as ulib
import pandas as pd
import openpyxl
import csv
from pathlib import Path
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def main(input_vid_path, output_path, datagroup):#, out_dir):
    # Reset graph before processing each video
    tf.compat.v1.reset_default_graph()
    tf.test.is_gpu_available(cuda_only=False)

    solution = Solu(input_vid_path)

    net = BlinkLRCN(
        is_train=False
    )
    net.build()
    sess = tf.compat.v1.Session()
    # Init solver
    solver = Solver(sess=sess,
                    net=net,
                    mode='lrcn')
    solver.init()

    stride = 10
    batch_size = np.arange(0, solution.frame_num, stride)
    for i in batch_size:

        eye1_list, eye2_list = [], []
        eye1_index = []
        eye2_index = []

        for j in range(i, np.minimum(i + stride, solution.frame_num)):
            eye1, eye2 = solution.get_eye_by_fid(j)
            if eye1 is not None:
                eye1_index.append(j)
                eye1_list.append(cv2.resize(eye1, (net.img_size[0], net.img_size[1])))

            if eye2 is not None:
                eye2_index.append(j)
                eye2_list.append(cv2.resize(eye2, (net.img_size[0], net.img_size[1])))
        if len(eye1_list) == 0:
            # Create a default eye image with zeros
            default_eye = np.zeros((net.img_size[0], net.img_size[1], 3), dtype=np.int32)
            eye1_list.append(default_eye)
            eye1_index.append(i)  # Use the batch start frame as index
    
        if len(eye2_list) == 0:
            # Create a default eye image with zeros
            default_eye = np.zeros((net.img_size[0], net.img_size[1], 3), dtype=np.int32)
            eye2_list.append(default_eye)
            eye2_index.append(i)  # Use the batch start frame as index
        eye1_full = ulib.pad_to_max_len(eye1_list, net.max_time,
                                        pad=np.zeros(eye1_list[0].shape, dtype=np.int32))
        eye2_full = ulib.pad_to_max_len(eye2_list, net.max_time,
                                        pad=np.zeros(eye2_list[0].shape, dtype=np.int32))
        eye1_probs, = solver.test([eye1_full], [len(eye1_list)])
        eye2_probs, = solver.test([eye2_full], [len(eye2_list)])

        for j in range(i, np.minimum(i + stride, solution.frame_num)):
            if j in eye1_index:
                eye1_prob = eye1_probs[0][eye1_index.index(j), 1] #column 1 = probability
            else:
                eye1_prob = 0.5

            if j in eye2_index:
                eye2_prob = eye2_probs[0][eye2_index.index(j), 1]
            else:
                eye2_prob = 0.5

            solution.push_eye_prob(eye1_prob, eye2_prob)
            solution.plot_by_fid(j)
    print('ending session')
    sess.close()

    x_axis = np.arange(0, solution.frame_num/solution.fps, 0.02)
    eye1_descale = []
    eye2_descale = []

    prev = 0
    for i, x in enumerate(solution.total_eye1_prob):
        for j in x_axis:
            if j < i/solution.fps:
                if j >= prev:
                    eye1_descale.append(x)
            else:
                prev = j
                break
    prev = 0
    for i, x in enumerate(solution.total_eye2_prob):
        for j in x_axis:
            if j < i/solution.fps:
                if j >= prev:
                    eye2_descale.append(x)
            else:
                prev = j
                break

    with open(output_path / datagroup / f"{input_vid_path.stem}_righteye.csv", 'w', newline='') as f:
        write = csv.writer(f)
        write.writerow(eye1_descale)

    with open(output_path / datagroup / f"{input_vid_path.stem}_lefteye.csv", 'w', newline='') as f:
        write = csv.writer(f)
        write.writerow(eye2_descale)

        
    tf.compat.v1.reset_default_graph()
    #solution.gen_videos(out_dir, 'lrcn')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_vid_path', type=str)
    parser.add_argument('--out_dir', type=str)
    args = parser.parse_args()
    main(Path(args.input_vid_path), Path(args.output_path), args.datagroup)
