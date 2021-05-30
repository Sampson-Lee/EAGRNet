import time
import torch
import os, numpy as np
import os.path as osp
import shutil
import cv2

def generate_list(dir_name, mode):
    line_list = []
    for rootn, dirn, filen_list in os.walk(dir_name+'/{}/image'.format(mode)):
        for filen in filen_list:
            if filen.endswith('jpg'):
                print(rootn, dirn, filen)
                img_name = os.path.join(rootn, filen)
                lab_name = img_name.replace('image', 'seg').replace('jpg','png')
                line_list.append(img_name+' '+lab_name+'\n')

    f = open(dir_name+'/{}_list.txt'.format(mode), 'w')
    f.writelines(line_list)
    f.close()

def generate_edge(label_dir):
    for rootn, dirn, filen_list in os.walk(label_dir):
        for filen in filen_list:
            label = cv2.imread(osp.join(rootn, filen), cv2.IMREAD_GRAYSCALE)
            edge = np.zeros_like(label)
            for i in range(label.shape[0]):
                for j in range(label.shape[1]):
                    flag = 1
                    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        x = i + dx
                        y = j + dy
                        if 0 <= x < label.shape[0] and 0 <= y < label.shape[1]:
                            if label[i,j] != label[x,y]:
                                edge[i,j] = 255
            
            edge_dir = (osp.join(rootn, filen)).replace('seg', 'edge')
            if not os.path.exists(os.path.dirname(edge_dir)):
                os.makedirs(os.path.dirname(edge_dir))
            cv2.imwrite(edge_dir, edge)

def generate_edge_pt(meta_line):
    label_name = meta_line.strip().rstrip('\n').split(' ')[1]
    label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
    edge = np.zeros_like(label)
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            flag = 1
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                x = i + dx
                y = j + dy
                if 0 <= x < label.shape[0] and 0 <= y < label.shape[1]:
                    if label[i,j] != label[x,y]:
                        edge[i,j] = 255
    
    edge_name = label_name.replace('seg', 'edge')
    if not os.path.exists(os.path.dirname(edge_name)):
        os.makedirs(os.path.dirname(edge_name))
    cv2.imwrite(edge_name, edge)

def mp_work(mp_func, mp_num, mp_list):
    from multiprocessing import Pool
    pool = Pool(mp_num)
    mp_return = pool.map(mp_func, mp_list)
    pool.close()
    pool.join()
    return mp_return

def generate_edge_mp():
    from functools import partial
    meta_dir = '/home/xian/Documents/xinpeng/CVPR-PIC-DATA/train_list.txt'
    meta_lines = open(meta_dir, 'r').readlines()
    print(len(meta_lines))
    print(meta_lines)
    partia_func = partial(generate_edge_pt)
    mp_work(partia_func, 10, meta_lines)

generate_list('/home/xian/Documents/xinpeng/CVPR-PIC-DATA', 'test')
# generate_edge_mp()


