import pandas as pd
import re
import xml.etree.ElementTree as gfg
import torch

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
from PIL import Image
import os.path



import copy
import inspect
import math
import warnings

import cv2
import mmcv
import numpy as np
from numpy import random
from mmdet.core import BitmapMasks, PolygonMasks, find_inside_bboxes
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.utils import log_img_scale


ALL_CLASSES_SPLIT1 = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter',
               'container-crane','airport', 'helipad','tower-crane','crane-truck','mobile-crane')

BASE_CLASSES_SPLIT1 = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter')

NOVEL_CLASSES_SPLIT2 = ['airplane', 'windmill', 'baseballfield', 'tenniscourt', 'trainstation']
ALL_CLASSES_SPLIT1_DIOR = ('golffield', 'Expressway-toll-station' ,'trainstation' ,'chimney',
                            'storagetank', 'ship', 'harbor' ,'airplane',
                            'tenniscourt', 'dam' ,'basketballcourt', 'Expressway-Service-area' ,
                            'stadium', 'airport', 'baseballfield', 'bridge' ,
                            'windmill', 'overpass','groundtrackfield' , 'vehicle')

ALL_CLASSES_SPLIT1_DIOR_D = {0:'golffield', 1:'Expressway-toll-station' ,2:'trainstation' ,3:'chimney',
                            4:'storagetank', 5:'ship', 6:'harbor' ,7:'airplane',
                            8:'tenniscourt', 9:'dam' ,10:'basketballcourt', 11:'Expressway-Service-area' ,
                            12:'stadium', 13:'airport', 14:'baseballfield', 15:'bridge' ,
                            16:'windmill', 18:'overpass',19:'groundtrackfield' , 20:'vehicle'}

ALL_CLASSES_SPLIT1_DIOR_L = ['golffield', 'Expressway-toll-station' ,'trainstation' ,'chimney',
                            'storagetank', 'ship', 'harbor' ,'airplane',
                            'tenniscourt', 'dam' ,'basketballcourt', 'Expressway-Service-area' ,
                            'stadium', 'airport', 'baseballfield', 'bridge' ,
                            'windmill', 'overpass','groundtrackfield' , 'vehicle']
ALL_CLASSES_SPLIT1_DIOR_clean = ['Golffield', 'Expressway Toll Station' ,'Trainstation' ,'Chimney',
                            'Storage Tank', 'Ship', 'Harbor' ,'Airplane',
                            'Tennis court', 'Dam' ,'Basketball court', 'Expressway Service Area' ,
                            'Stadium', 'Airport', 'Baseball Field', 'Bridge' ,
                            'Windmill', 'Overpass','Ground Trackfield' , 'Vehicle']



DIOR_VAL = '/home/data/dior/ImageSets/Main/val.txt'
DIOR_TRAIN = '/home/data/dior/ImageSets/Main/train.txt'
DIOR_TEST = '/home/data/dior/ImageSets/Main/test.txt'
DIOR_PATH = '/home/data/dior/ImageSets/Main/'
DIOR_XML = '/home/data/dior/Annotations/'

DIOR_FSANN = '/home/data/dior/few_shot_ann/'

def parse_args():
    parser = argparse.ArgumentParser(
        description='handle DIOR dataset few shot annotations creation')
    parser.add_argument('--dir', default=DIOR_TRAIN, help='directory of the image to analyse')
    parser.add_argument('--save_dir', default=DIOR_FSANN, help='directory of the image to analyse')
    parser.add_argument('--img_dir', default=None, help='directory of the image to analyse')
    parser.add_argument('--class_to_find', default=None, help='find image with a given class')
    parser.add_argument('--path_all_files', default=None,
                         help='Where to store a txt file with all file name')
    parser.add_argument('--create_txt_by_classes', default=False, help='Define if ds is separated in classes ')
    parser.add_argument('--nshots', help="number of shots desired", default=[10, 20, 100])
    parser.add_argument('--few_shot_ann', choices=['none', 'random', 'easy','hard'], default='hard')
    args = parser.parse_args()
    return args


def get_images_with_class(class_name, df):
    """
    Output the dataframe composed of a single class
    """
    return df[(df == class_name).any(axis=1)]

def create_few_shot_ann(directory=None, version_ann='v3', nbr_shots=None, save=True):
    """
    Create the few shot annotations for the DIOR dataset.

    directory: indicate the train.txt file with all images of the train set. If set to None, uses the directory provided with parser
    Version_ann: specifies which annotation version is being produced, i.e. to maintain multiple diverse annotation panels
    nbr_shots: list of the number of shots to create annotations for. If set to None, uses the list from parser
    save: Boolean to indicate whether the annotations are saved or not
    """

        
    args = parse_args()
    print(directory)
    # assign directory
    if directory is None:
        directory = args.dir

    if nbr_shots is None:
        nbr_shots = args.nshots
    print(nbr_shots)
    
    with open(directory, 'r') as df:
            lines = df.readlines()
        # first open the file and seperate every line like below:
        # remove /n at the end of each line
    for index, line in enumerate(lines):
        lines[index] = line.strip()
    

    filel = []
    for file_id, file_name in tqdm(enumerate(lines)):

        xml_path = DIOR_XML + file_name + '.xml'
        if os.path.isfile(xml_path):
            current_file_classes = []
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                current_file_classes.append([name, file_name])

            if not file_id:
                df_result = pd.DataFrame(current_file_classes, columns=['category', 'file'])
            else:
                df_temp = pd.DataFrame(current_file_classes, columns=['category', 'file'])
                df_result = pd.concat([df_result, df_temp], ignore_index=True, axis=0)
    # calculate counts
    del df_temp
    if args.few_shot_ann != 'none':
        # category, nbr inst, file
        few_ann = []

    if args.class_to_find is not None or args.few_shot_ann != 'none':
        if args.class_to_find == 'all' or args.few_shot_ann != 'none':
            for class_to_find in list(ALL_CLASSES_SPLIT1_DIOR):
                
                nbr_instance = []
                df_of_class = get_images_with_class(class_to_find, df_result)
                file_of_class = df_of_class['file'].unique()
                for file_current in file_of_class:
                    a = df_result[(df_result['category'] == class_to_find) & (df_result['file'] == file_current)]
                    
                    nbr_instance.append((a.shape)[0])
                    if args.few_shot_ann != 'none': 
                        few_ann.append([class_to_find, (a.shape)[0], file_current])

    if args.few_shot_ann != 'none':
        df_inst_file = pd.DataFrame(few_ann, columns=['category', 'nbr_instances', 'file'])

        for shot in nbr_shots:
            print(shot)
            used_file = []
            for curent_class_name in df_inst_file['category'].unique():
                # Shuffle
                df_inst_file = df_inst_file.sample(frac=1)

                files_of_current_class = df_inst_file.loc[(df_inst_file['category'] == curent_class_name) & (~df_inst_file['file'].isin(used_file))]
                if args.few_shot_ann == 'random':
                    data = np.array([files_of_current_class['file'].unique()]).T

                # For novel classes
                # for easy it takes the images with most instances
                elif args.few_shot_ann == 'easy':
                    df_novel = files_of_current_class.nlargest(shot, ['nbr_instances'])

                    if isinstance(df_novel, str):
                        data = np.array([df_novel['file'].unique()]).T
                    else: 
                        data = np.array([df_novel['file']])

                # for hard it takes the images with least instances
                elif args.few_shot_ann == 'hard':
                    df_novel = files_of_current_class.nsmallest(shot, ['nbr_instances'])
                    
                    if isinstance(df_novel, str):
                        data = np.array([df_novel['file'].unique()]).T
                    else: 
                        data = np.array([df_novel['file']])

                files_names = np.stack(data, axis=0 )
                file_cycle = cycle(files_names)
                files_names = [next(file_cycle) for i in range(shot)]
                used_file += files_names
                if save:                    
                    Path(args.save_dir+version_ann).mkdir(parents=True, exist_ok=True)
                    save_path = args.save_dir+version_ann+args.few_shot_ann+'_benchmark_'+str(shot)+'shot'
                    Path(save_path).mkdir(parents=True, exist_ok=True)

                    if 'train' in directory:
                        with open(save_path+'/box_'+str(shot)+'shot_'+curent_class_name+'_train.txt', 'w') as f:
                            for element in files_names[0]:
                                f.write(element + "\n")
                        print(f'Training Few-Shot annotations have been saved to {save_path}')
                    elif 'val' in directory:
                        with open(save_path+'/box_'+str(shot)+'shot_'+curent_class_name+'_val.txt', 'w') as f:
                            for element in files_names[0]:
                                f.write(element + "\n")
                        print(f'Validation Few-Shot annotations have been saved to {save_path}')
                    else:
                        print('problem in directory selected. Not train nor validation')

def do_nice_graphs(input_path):
    """
    Compute diverse graphs such as size, area and aspect ratio distribution per class
    """
    import os.path
    args = parse_args()
    
    instances_per_class = {}
    are_per_class = {}
    h_per_class = {}
    w_per_class = {}
    ratio_per_class = {}
    for class_id in ALL_CLASSES_SPLIT1_DIOR_L:
        instances_per_class[class_id] = 0
        ratio_per_class[class_id] = []
        w_per_class[class_id] = []
        h_per_class[class_id] = []
        are_per_class[class_id] = []

    if type(input_path) is not list:
        input_path = [input_path]
    for current_path in input_path:
        with open(current_path, 'r') as df:
                lines = df.readlines()
            # first open the file and seperate every line like below:
            # remove /n at the end of each line
        for index, line in enumerate(lines):
            lines[index] = line.strip()
        
        for file_id, file_name in tqdm(enumerate(lines)):
            xml_path = DIOR_XML + file_name + '.xml'
            if os.path.isfile(xml_path):
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    instances_per_class[name] += 1
                    s = obj.find('bndbox')
                    if int(s.find('xmax').text)-int(s.find('xmin').text) > 0 and int(s.find('ymax').text)-int(s.find('ymin').text) > 0:
                        w_per_class[name].append(int(s.find('xmax').text)-int(s.find('xmin').text))
                        h_per_class[name].append(int(s.find('ymax').text)-int(s.find('ymin').text))
                        ratio_per_class[name].append((int(s.find('xmax').text)-int(s.find('xmin').text))/(int(s.find('ymax').text)-int(s.find('ymin').text)))
                        are_per_class[name].append((int(s.find('xmax').text)-int(s.find('xmin').text))*(int(s.find('ymax').text)-int(s.find('ymin').text)))

    sorted_instances_per_class = sorted(instances_per_class.items(), key=lambda x:x[1])
    instances_per_class = dict(sorted_instances_per_class)
    print(instances_per_class)


    clean_name = {}
    for id, save_name in enumerate(ALL_CLASSES_SPLIT1_DIOR_L):
        clean_name[save_name] = ALL_CLASSES_SPLIT1_DIOR_clean[id]
    # creating the bar plot
    

    # Horizontal Bar Plot
    plot = ['box_plot_ratio']
    if 'bar_plot' in plot:
        fig, ax = plt.subplots(figsize =(20, 9))
        name_plot,val_plot = [], []
        for key in instances_per_class.keys():
            name_plot.append(clean_name[key])
            val_plot.append(instances_per_class[key])
        ax.barh(name_plot, val_plot, height=0.2)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.invert_yaxis()
        ax.grid(visible = True, color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.8)
        ax.xaxis.set_tick_params(pad = 5)
        ax.yaxis.set_tick_params(pad = 2)
        for i in ax.patches:
            plt.text(i.get_width(), i.get_y()+.25,
                str(round((i.get_width()), 2)),
                fontsize = 10, fontweight ='bold',
                color ='grey')

        # Add Plot Title
        ax.set_xlabel("Number of instances")
        fig.savefig(f'inst_class.pdf', transparent=True)
    if 'box_plot_area' in plot:
        fig, ax = plt.subplots(figsize =(20, 9))
        name_plot,val_plot = [], []

        for key in are_per_class.keys():
            name_plot.append(clean_name[key])
            val_plot.append(are_per_class[key])

        ax.boxplot(val_plot, labels=name_plot)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        
        ax.grid(visible = True, color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.8)
            
        for i in ax.patches:
            plt.text(i.get_width(), i.get_y()+.5,
                str(round((i.get_width()), 2)),
                fontsize = 10, fontweight ='bold',
                color ='grey')

        # Add Plot Title
        ax.set_xlabel("Classes")
        fig.savefig(f'area_plot.png', transparent=True)
    
    if 'box_plot_w' in plot:
        fig, ax = plt.subplots(figsize =(40, 9))
        name_plot,val_plot = [], []

        for key in are_per_class.keys():
            name_plot.append(clean_name[key])
            val_plot.append(w_per_class[key])

        ax.boxplot(val_plot, labels=name_plot)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        
        ax.grid(visible = True, color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.8)
            
        for i in ax.patches:
            plt.text(i.get_width(), i.get_y()+.5,
                str(round((i.get_width()), 2)),
                fontsize = 10, fontweight ='bold',
                color ='grey')

        # Add Plot Title
        #ax.set_xlabel("Classes")
        fig.savefig(f'w_plot.png', transparent=True)
        fig.savefig(f'w_plot.pdf', transparent=True)
    if 'box_plot_wh' in plot:
        fig, ax = plt.subplots(figsize =(40, 9))
        name_plot,val_plot, wh_name = [], [], []

        for key in are_per_class.keys():
            name_plot.append(clean_name[key])
            name_plot.append(' ')
            wh_name.append(clean_name[key])
            val_plot.append(h_per_class[key])
            val_plot.append(w_per_class[key])

        ax.boxplot(val_plot, labels=name_plot)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        
        ax.grid(visible = True, color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.8)
            
        for i in ax.patches:
            plt.text(i.get_width(), i.get_y()+.5,
                str(round((i.get_width()), 2)),
                fontsize = 10, fontweight ='bold',
                color ='grey')
        
        # Add Plot Title
        #ax.set_xlabel("Classes")
        fig.savefig(f'wh_plot.png')#, transparent=True)
        fig.savefig(f'wh_plot.pdf', transparent=True)
    if 'box_plot_h' in plot:
        fig, ax = plt.subplots(figsize =(40, 9))
        name_plot,val_plot = [], []

        for key in are_per_class.keys():
            name_plot.append(clean_name[key])
            val_plot.append(h_per_class[key])

        ax.boxplot(val_plot, labels=name_plot)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        
        ax.grid(visible = True, color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.8)
            
        for i in ax.patches:
            plt.text(i.get_width(), i.get_y()+.5,
                str(round((i.get_width()), 2)),
                fontsize = 10, fontweight ='bold',
                color ='grey')

        # Add Plot Title
        #ax.set_xlabel("Classes")
        fig.savefig(f'h_plot.png', transparent=True)
        fig.savefig(f'h_plot.pdf', transparent=True)
    if 'box_plot_ratio' in plot:
        fig, ax = plt.subplots(figsize =(40, 9))
        name_plot,val_plot, wh_name = [], [], []

        for key in are_per_class.keys():
            name_plot.append(clean_name[key])
            val_plot.append(ratio_per_class[key])

        ax.boxplot(val_plot, labels=name_plot)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        
        ax.grid(visible = True, color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.8)
            
        for i in ax.patches:
            plt.text(i.get_width(), i.get_y()+.5,
                str(round((i.get_width()), 2)),
                fontsize = 10, fontweight ='bold',
                color ='grey')

        # Add Plot Title
        #ax.set_xlabel("Classes")
        fig.savefig(f'ratio_plot.png')#, transparent=True)
        fig.savefig(f'ratio_plot.pdf', transparent=True)

def check_bbox_per_file(version_ann=None):
    """
    Compute the number of instances of each class for the few shot annotation of base and novel classes
    """
    args = parse_args()
    # assign directory

    dict_cntr = {}
    for key in ALL_CLASSES_SPLIT1_DIOR_L:
        dict_cntr[key] = 0
    in_loop = False
    for shot in [10]:
        print(f'shot: {shot}')
        for comp in ['hard']:
            print(f'comp: {comp}')
            few_and_val = []
            new_train = []

            all_class_2 = []
            for cclass in ALL_CLASSES_SPLIT1_DIOR_L:
                if version_ann is None:
                    save_path_dior = DIOR_FSANN + comp + '_benchmark_' + str(shot) + 'shot/box_'+str(shot)+'shot_' + cclass + '_train.txt'
                else:
                    save_path_dior = DIOR_FSANN+version_ann + comp + '_benchmark_' + str(shot) + 'shot/box_'+str(shot)+'shot_' + cclass + '_train.txt'
                
                with open(save_path_dior, 'r') as df:
                    lines = df.readlines()
                    
                for index, line in enumerate(lines):
                    lines[index] = line.strip()

                for file_id, file_name in enumerate(lines):
                    xml_path = DIOR_XML + file_name + '.xml'
                    current_file_classes = []
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    names = []
                    
                    for obj in root.findall('object'):
                        name = obj.find('name').text   
                        dict_cntr[name] += 1
                print(f'after class {cclass}')
                for nclass in NOVEL_CLASSES_SPLIT2:
                    print(f'class {nclass} has {dict_cntr[nclass]} instances')

    print('Novel classes')
    for nclass in NOVEL_CLASSES_SPLIT2:
        print(f'class {nclass} has {dict_cntr[nclass]} instances')
    
    print('base classes')
    for nclass in ALL_CLASSES_SPLIT1_DIOR_L:
        if nclass not in NOVEL_CLASSES_SPLIT2:
            print(f'class {nclass} has {dict_cntr[nclass]} instances')


if __name__ == '__main__':
    

    create_few_shot_ann()
