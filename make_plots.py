import pandas as pd
import re
import xml.etree.ElementTree as gfg
import torch
import numpy as np
import glob
from tqdm import tqdm

from collections import Counter, OrderedDict
import argparse
from itertools import cycle
from PIL import Image
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from pathlib import Path
import pickle

def parse_args():
    parser = argparse.ArgumentParser(
        description='handle DOTA dataset')
    parser.add_argument('--dir', default=DIOR_TRAIN, help='directory of the image to analyse')
    parser.add_argument('--img_dir', default=None, help='directory of the image to analyse')
    parser.add_argument('--class_to_find', default='all', help='find image with a given class')
    parser.add_argument('--path_all_files', default=None,#'/home/data/dota-1.5/train/labelTxt-v1.5/DOTA-v1.5_train/train.txt',
                         help='Where to store a txt file with all file name')
    parser.add_argument('--create_txt_by_classes', default=False, help='Define if ds is separated in classes ')
    parser.add_argument('--nshots', help="number of shots desired", default=[20])
    parser.add_argument('--few_shot_ann', choices=['none', 'random', 'easy','hard'], default='none')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # assign directory

    directory = args.dir

    # iterate over files in
    # that directory
    nbr_xnequal = 0
    nbr_xequal = 0
    nbr_ynequal = 0
    nbr_yequal = 0
    for file_id, filename in tqdm(enumerate(glob.iglob(f'{directory}/P*.txt'))):
        # print(filename)
        if 'plane.txt' in filename:
            continue
        with open(filename, 'r') as df:
            lines = df.readlines()
    # first u have to open  the file and seperate every line like below:

    # remove /n at the end of each line
        for index, line in enumerate(lines):
            lines[index] = line.strip()
            lines[index] = list(lines[index].split(" "))
            if 'P5' not in filename:
                lines[index].append(filename[len(directory)+1:-4])
                # x1 = x4 = xmin, x2 = x3 = xmax, y1 = y2 = ymin, y3 = y4 = ymax from OBB to HBB

        if 'P5' not in filename:
            lines = lines[2:]
    # name : filename[len(directory)+1:-4]
    # creating a dataframe(consider u want to convert your data to 2 columns)
        if not file_id:
            df_result = pd.DataFrame(lines, columns=['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'category', 'difficult','file'])
        else:
            df_temp = pd.DataFrame(lines, columns=['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'category', 'difficult','file'])
            df_result = pd.concat([df_result, df_temp], ignore_index=True, axis=0)

        

    # calculate counts
    del df_temp
    c = Counter(df_result['category'])
    df_classes = pd.DataFrame.from_dict(c, orient='index').reset_index().rename(columns={'index': 'class', 0: 'number'})
    df_classes = df_classes.sort_values('number', ascending=False)

    if args.path_all_files is not None:
        np.savetxt(args.path_all_files, np.array([df_result['file'].unique()]).T, fmt='%s')

    #print(df_classes.to_string(index=False))
    if args.class_to_find is not None:
        if args.class_to_find == 'all':
            for class_to_find in list(ALL_CLASSES_SPLIT1):
                df_of_class = get_images_with_class(class_to_find, df_result)
                file_of_class = df_of_class['file'].unique()
                print(len(file_of_class), "images contain the class", class_to_find)
        else:
            for class_to_find in args.class_to_find:
                df_of_class = get_images_with_class(class_to_find,df_result)
                file_of_class = df_of_class['file'].unique()
                print(file_of_class, "contain the class", class_to_find)

    if args.create_txt_by_classes or args.few_shot_ann:
        df_cat_file = df_result[['category','file']]
        for curent_class_name in df_cat_file['category'].unique():
            #print(curent_class_name)
            file_of_current_class = df_cat_file.loc[df_cat_file['category'] == curent_class_name]
            if args.create_txt_by_classes:
                np.savetxt(directory+'/'+curent_class_name+'.txt', np.array([file_of_current_class['file'].unique()]).T, fmt='%s')
            if args.few_shot_ann:
                for shot in [20]:#[1,5,10,100]:
                    data = np.array([file_of_current_class['file'].unique()]).T
                    #print(data)
                    files_names = list(map(lambda n: 'images/'+n+'.txt', data))
                    files_names = np.stack(files_names, axis=0 )
                    file_cycle = cycle(files_names)
                    files_names = [next(file_cycle) for i in range(shot)]
                    np.savetxt('/home/data/dota-1.5/few_shot_ann/benchmark_'+str(shot)+'shot/box_'+str(shot)+'shot_'+curent_class_name+'_train.txt',
                               files_names, fmt='%s')


def do_plot_fast():
    import matplotlib.gridspec as grd
    
    baseline_FPN_RPN_FT = {'baseballfield':[0.794, 0.881, 0.901],
                            'airplane':[0.138, 0.531, 0.730],
                            'tenniscourt':[0.497, 0.631, 0.623],
                            'trainstation':[0.007, 0.022, 0.067],
                            'windmill':[0.003, 0.017, 0.026],
                            'mean':[0.2878, 0.4164, 0.4694]}
    SAM = {'baseballfield':[0.730, 0.780, 0.810],
                            'airplane':[0.530, 0.660, 0.670],
                            'tenniscourt':[0.490, 0.650, 0.700],
                            'trainstation':[0.025, 0.035, 0.058],
                            'windmill':[0.140, 0.260, 0.305],
                            'mean':[0.3830, 0.4730, 0.5090]}
    ours = {'baseballfield':[0.868, 0.896, 0.909],
                        'airplane':[0.551, 0.765, 0.837],
                        'tenniscourt':[0.593, 0.704, 0.609],
                        'trainstation':[0.056, 0.153, 0.221],
                        'windmill':[0.007, 0.095, 0.119],
                        'mean':[0.4150, 0.5226, 0.5390]}                            
    print(baseline_FPN_RPN_FT.keys())

if __name__ == '__main__':
    do_plot_fast()