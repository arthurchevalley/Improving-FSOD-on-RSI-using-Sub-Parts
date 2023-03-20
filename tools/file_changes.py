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
from PIL import Image
import os.path


# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import math
import warnings

import cv2
import mmcv
import numpy as np
from numpy import random
import torch
from mmdet.core import BitmapMasks, PolygonMasks, find_inside_bboxes
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.utils import log_img_scale

xView_classes = {
    11:'Fixed-wingAircraft', 12:'SmallAircraft',13:'CargoPlane',15:'Helicopter',17:'PassengerVehicle',
    18:'SmallCar',19:'Bus',20:'PickupTruck',21:'UtilityTruck',23:'Truck',24:'CargoTruck',25:'TruckTractorWBoxTrailer',
    26:'TruckTractor',27:'Trailer',28:'TruckTractorWFlatbedTrailer',29:'TruckTractorWLiquidTank',32:'crane-truck',
    33:'RailwayVehicle',34:'PassengerCar',35:'Cargo',36:'Flat-Car',37:'Tank-car',38:'Locomotive',40:'MaritimeVessel',
    41:'Motorboat',42:'Sailboat',44:'Tugboat',45:'Barge',47:'FishingVessel',49:'Ferry',50:'Yacht',51:'ContainerShip',
    52:'OilTanker',53:'EngineeringVehicle',54:'tower-crane',55:'container-crane',56:'ReachStacker',57:'StraddleCarrier',59:'mobile-crane',
    60:'DumpTruck',61:'HaulTruck',62:'Scraper-Tractor',63:'frontloader-bulldozer',64:'excavator',65:'cement-mixer',66:'GroundGrader',
    71:'hut-tent',72:'Shed',73:'Building',74:'AircraftHangar',75:'notDefined',76:'DamagedBuilding',77:'Facility',79:'construction-site',
    82:'notDefined',83:'VehicleLot',84:'helipad',86:'StorageTank',89:'shipping-container-lot',91:'ShippingContainer',93:'Pylon', 94:'Tower'
    }
ALL_CLASSES_xview = ('Fixed-wingAircraft', 'SmallAircraft','CargoPlane','Helicopter','PassengerVehicle',
    'SmallCar','Bus','PickupTruck','UtilityTruck','Truck','CargoTruck','TruckTractorWBoxTrailer',
    'TruckTractor','Trailer','TruckTractorWFlatbedTrailer','TruckTractorWLiquidTank','crane-truck',
    'RailwayVehicle','PassengerCar','Cargo','Flat-Car','Tank-car','Locomotive','MaritimeVessel',
    'Motorboat','Sailboat','Tugboat','Barge','FishingVessel','Ferry','Yacht','ContainerShip',
    'OilTanker','EngineeringVehicle','tower-crane','container-crane','ReachStacker','StraddleCarrier','mobile-crane',
    'DumpTruck','HaulTruck','Scraper-Tractor','frontloader-bulldozer','excavator','cement-mixer','GroundGrader',
    'hut-tent','Shed','Building','AircraftHangar','notDefined','DamagedBuilding','Facility','construction-site',
    'notDefined','VehicleLot','helipad','StorageTank','shipping-container-lot','ShippingContainer','Pylon', 'Tower')
truck_split = [20,21,23,24,25,26,28,29]

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
XVIEW_CLASSES_SPLIT1= ('tower-crane','crane-truck','mobile-crane')
DOTA_HBB_TRAIN = '/home/data/dota-1.5/train/labelTxt-v1.5/DOTA-v1.5_train_hbb'
DOTA_HBB_VAL = '/home/data/dota-1.5/val/labelTxt-v1.5/DOTA-v1.5_val_hbb'
DIOR_VAL = '/home/archeval/mmdetection/CATNet/mmdetection/data/dior/ImageSets/Main/val.txt'
DIOR_TRAIN = '/home/archeval/mmdetection/CATNet/mmdetection/data/dior/ImageSets/Main/train.txt'
DIOR_TEST = '/home/archeval/mmdetection/CATNet/mmdetection/data/dior/ImageSets/Main/test.txt'
DIOR_PATH = '/home/archeval/mmdetection/CATNet/mmdetection/data/dior/ImageSets/Main/'
DIOR_XML = '/home/archeval/mmdetection/CATNet/mmdetection/data/dior/Annotations/'

DIOR_XML_BIG = '/home/archeval/mmdetection/CATNet/mmdetection/data/dior2/Annotations/big/'
DIOR_VAL_BIG = '/home/archeval/mmdetection/CATNet/mmdetection/data/dior2/ImageSets/Main/big/val.txt'
DIOR_TRAIN_BIG = '/home/archeval/mmdetection/CATNet/mmdetection/data/dior2/ImageSets/Main/big/train.txt'
DIOR_FSANN_BIG = '/home/archeval/mmdetection/CATNet/mmdetection/data/dior2/few_shot_ann/'

DIOR_FSANN = '/home/archeval/mmdetection/CATNet/mmdetection/data/dior/few_shot_ann/'
xview_base = '/home/data/xView/'
xview_train = xview_base + 'ImageSets/Main/train.txt'
xview_val = xview_base + 'ImageSets/Main/val.txt'
xview_train_tile = xview_base + 'ImageSets/Main/tiles/train.txt'
xview_val_tile = xview_base + 'ImageSets/Main/tiles/val.txt'
xview_xml = xview_base + 'Annotations/'
xview_xml_tile = xview_base + 'Annotations/tiles/'
xview_xml_lower_tile = xview_base + 'Annotations/tiles_lower_overlap/'
xview_img = xview_base + 'train_images/'

def parse_args():
    parser = argparse.ArgumentParser(
        description='handle DOTA dataset')
    parser.add_argument('--dir', default=xview_train_tile, help='directory of the image to analyse')
    parser.add_argument('--img_dir', default=None, help='directory of the image to analyse')
    parser.add_argument('--class_to_find', default=None, help='find image with a given class')
    parser.add_argument('--path_all_files', default=None,#'/home/data/dota-1.5/train/labelTxt-v1.5/DOTA-v1.5_train/train.txt',
                         help='Where to store a txt file with all file name')
    parser.add_argument('--create_txt_by_classes', default=False, help='Define if ds is separated in classes ')
    parser.add_argument('--nshots', help="number of shots desired", default=[10, 20, 100])
    parser.add_argument('--few_shot_ann', choices=['none', 'random', 'easy','hard'], default='hard')
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

def create_few_shot_ann():
        
    args = parse_args()
    # assign directory
    
    directory = args.dir
    nbr_shots = args.nshots
    print(nbr_shots[0])
    
    with open(directory, 'r') as df:
            lines = df.readlines()
        # first open the file and seperate every line like below:
        # remove /n at the end of each line
    for index, line in enumerate(lines):
        lines[index] = line.strip()
    

    filel = []
    for file_id, file_name in tqdm(enumerate(lines)):

        xml_path = DIOR_XML + file_name + '.xml'
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

                    #print(f'image {file_current}.xml contains {(a.shape)[0]} time the class {class_to_find}')
                #if args.class_to_find is not None:
                   # print(f'{len(file_of_class)} images contain the class {class_to_find} for a total of {sum(nbr_instance)} instances')
                    #print(f'mean instances {sum(nbr_instance)/len(nbr_instance)}')
                   # print(f'There is at most {max(nbr_instance)} instance for image {nbr_instance.index(max(nbr_instance))} and at least {min(nbr_instance)} instances')
                    #print(f'unique instances {len(list(np.unique(np.array(nbr_instance))))}')
                   # d = dict(Counter(nbr_instance))

                    #print(f'Repartition of number instances per image {OrderedDict(sorted(d.items()))}')
        
    if args.few_shot_ann != 'none':
        df_inst_file = pd.DataFrame(few_ann, columns=['category', 'nbr_instances', 'file'])

        for shot in nbr_shots:
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
                    #print(df_novel)
                    #print("======")
                    if isinstance(df_novel, str):
                        data = np.array([df_novel['file'].unique()]).T
                    else: 
                        data = np.array([df_novel['file']])

                files_names = np.stack(data, axis=0 )
                file_cycle = cycle(files_names)
                files_names = [next(file_cycle) for i in range(shot)]
                used_file += files_names
                #creating a new directory 
                save_path = '/home/archeval/mmdetection/CATNet/mmdetection/data/dior/few_shot_ann/'+args.few_shot_ann+'_benchmark_'+str(shot)+'shot'
                Path(save_path).mkdir(parents=True, exist_ok=True)
                #np.savetxt(save_path+'/box_'+str(shot)+'shot_'+curent_class_name+'_train.txt', files_names, fmt='%s')
                with open(save_path+'/box_'+str(shot)+'shot_'+curent_class_name+'_train.txt', 'w') as f:
                    for element in files_names[0]:
                        f.write(element + "\n")

def check_ann():
        
    args = parse_args()
    # assign directory
    bizr = []
    
    directory = args.dir

    with open(directory, 'r') as df:
            lines = df.readlines()
        # first u have to open  the file and seperate every line like below:

        # remove /n at the end of each line
    for index, line in enumerate(lines):
        lines[index] = line.strip()
    

    filel = []
    for file_id, file_name in tqdm(enumerate(lines)):

        xml_path = DIOR_XML + file_name + '.xml'
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

                    #print(f'image {file_current}.xml contains {(a.shape)[0]} time the class {class_to_find}')
                if args.class_to_find is not None:
                    print(f'{len(file_of_class)} images contain the class {class_to_find} for a total of {sum(nbr_instance)} instances')
                    print(f'There is at most {max(nbr_instance)} instance for image {nbr_instance.index(max(nbr_instance))} and at least {min(nbr_instance)} instances')
                    print(f'mean instances {sum(nbr_instance)/len(nbr_instance)}')
                    print(f'unique instances {len(list(np.unique(np.array(nbr_instance))))}')
                    d = dict(Counter(nbr_instance))

                    print(f'Repartition of number instances per image {OrderedDict(sorted(d.items()))}')
        else:       
            for class_to_find in args.class_to_find:
                df_of_class = get_images_with_class(class_to_find,df_result)
                file_of_class = df_of_class['file'].unique()
                print(file_of_class, "contain the class", class_to_find)
    if args.few_shot_ann != 'none':
        df_inst_file = pd.DataFrame(few_ann, columns=['category', 'nbr_instances', 'file'])

        for shot in [10, 100]:
            used_file = []
            for curent_class_name in df_inst_file['category'].unique():

                # Shuffle
                df_inst_file = df_inst_file.sample(frac=1)

                files_of_current_class = df_inst_file.loc[(df_inst_file['category'] == curent_class_name) & (~df_inst_file['file'].isin(used_file))]
                if args.few_shot_ann == 'random' or class_to_find != 'vehicle':
                    data = np.array([files_of_current_class['file'].unique()]).T

                # For novel classes
                # for easy it takes the images with most instances
                elif args.few_shot_ann == 'easy' and class_to_find == 'vehicle':
                    df_novel = files_of_current_class.nlargest(shot, ['nbr_instances'])

                    if isinstance(df_novel, str):
                        data = np.array([df_novel['file'].unique()]).T
                    else: 
                        data = np.array([df_novel['file']])

                # for hard it takes the images with least instances
                elif args.few_shot_ann == 'hard' and class_to_find == 'vehicle':
                    df_novel = files_of_current_class.nsmallest(shot, ['nbr_instances'])
                    #print(df_novel)
                    #print("======")
                    if isinstance(df_novel, str):
                        data = np.array([df_novel['file'].unique()]).T
                    else: 
                        data = np.array([df_novel['file']])

                files_names = np.stack(data, axis=0 )
                file_cycle = cycle(files_names)
                if class_to_find != 'vehicle' and shot > 100:
                    files_names = [next(file_cycle) for i in range(100)]
                else:
                    files_names = [next(file_cycle) for i in range(shot)]
                used_file += files_names
                #creating a new directory 
                save_path = '/home/archeval/mmdetection/CATNet/mmdetection/data/dior/few_shot_ann/'+args.few_shot_ann+'_benchmark_'+str(shot)+'shot'
                Path(save_path).mkdir(parents=True, exist_ok=True)
                #np.savetxt(save_path+'/box_'+str(shot)+'shot_'+curent_class_name+'_train.txt', files_names, fmt='%s')
                with open(save_path+'/box_'+str(shot)+'shot_'+curent_class_name+'_train.txt', 'w') as f:
                    for element in files_names[0]:
                        f.write(element + "\n")

def check_ann_img():
        
    args = parse_args()
    # assign directory
    bizr = []
    inst_number = []
    directory = args.dir

    
    shots = [5,10,20, 100]
    for comp in ['hard']:
        print(f'complexity: {comp}')
        for shot in shots:
            filel = []
            nbr_inst =[]
            fann_path = '/home/archeval/mmdetection/CATNet/mmdetection/data/dior/few_shot_ann/'+comp+'_benchmark_'+str(shot)+'shot/'
            for classes in list(ALL_CLASSES_SPLIT1_DIOR):
                tmp_filel = []
                tmp_nbr_inst = []
                directory = fann_path+'box_'+str(shot)+'shot_'+classes+'_train.txt'

                with open(directory, 'r') as df:
                    lines = df.readlines()
                # first u have to open  the file and seperate every line like below:

                # remove /n at the end of each line
                for index, line in enumerate(lines):
                    lines[index] = line.strip()
                for file_id, file_name in enumerate(lines):

                    xml_path = DIOR_XML + file_name + '.xml'
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
                if args.class_to_find is not None or args.few_shot_ann != 'none':
                    if args.class_to_find == 'all' or args.few_shot_ann != 'none':
                        files = df_result['file'].unique()
                        for _, file_current in enumerate(files):
                        #for class_to_find in list(ALL_CLASSES_SPLIT1_DIOR):
                            nbr_classes = []
                            #df_of_class = get_images_with_class(class_to_find, df_result)
                            #file_of_class = df_of_class['file'].unique()

                            #for class_to_find in list(ALL_CLASSES_SPLIT1_DIOR):
                            #for file_current in file_of_class:
                            a = df_result[(df_result['category'] == classes) & (df_result['file'] == file_current)]
                            
                            
                            #nbr_instance.append((a.shape)[0])
                            if (a.shape)[0] != 0:
                                nbr_classes.append((classes, (a.shape)[0]))
                            tmp_filel.append([classes, (a.shape)[0], file_current])
                            tmp_nbr_inst.append((a.shape)[0])
                            #print(f'image {file_current}.xml contains {(a.shape)[0]} time the class {class_to_find}')
                            if args.class_to_find is not None and False:
                                print(f'{file_current} contain {len(nbr_classes)} classes.')
                                print(f'Those are {nbr_classes[:][0]}')
                                print(f'mean instances {sum(nbr_instance)/len(nbr_instance)}')
                                print(f'unique instances {len(list(np.unique(np.array(nbr_instance))))}')
                else:   
                    print('fumadf')
                    files = df_result['file'].unique()
                    only_one = []
                    for _, file_current in enumerate(files):
                    #for class_to_find in list(ALL_CLASSES_SPLIT1_DIOR):
                        nbr_classes = []
                        #df_of_class = get_images_with_class(class_to_find, df_result)
                        #file_of_class = df_of_class['file'].unique()

                        for class_to_find in args.class_to_find:
                        #for file_current in file_of_class:
                            a = df_result[(df_result['category'] == class_to_find) & (df_result['file'] == file_current)]
                            #nbr_instance.append((a.shape)[0])
                            if (a.shape)[0] != 0:
                                nbr_classes.append((class_to_find, (a.shape)[0]))

                            #print(f'image {file_current}.xml contains {(a.shape)[0]} time the class {class_to_find}')
                        #if args.class_to_find in nbr_classes:
                        #    nbr_classes = nbr_classes.pop(args.class_to_find)
                        #    if len(nbr_classes)==0:
                        #        only_one.append(file_current)
                    if args.class_to_find is not None:
                        print(f'Images {only_one} only contains the class {args.class_to_find} class')
                filel.append(tmp_filel)
                nbr_inst.append(tmp_nbr_inst)
            bizr.append(filel)
            inst_number.append(torch.tensor(nbr_inst))
        
        cls_lis = ['golffield', 'Expressway-toll-station', 'trainstation', 'chimney',
                                'storagetank', 'ship', 'harbor', 'airplane', 
                                'groundtrackfield', 'tenniscourt', 'dam', 'basketballcourt',
                                'Expressway-Service-area', 'stadium', 'airport', 
                                'baseballfield', 'bridge', 'windmill', 'overpass']
        for i in range(len(inst_number)):
            print(f'for {shots[i]} shot the shape is {inst_number[i].shape} and the number of instances is/are {inst_number[i].unique()}') 
            for j in range(inst_number[i].unique().shape[0]):
                ids = (torch.eq(inst_number[i], inst_number[i].unique()[j]))
                print(f'there is {inst_number[i].unique()[j]} instances for {int(ids.float().sum())} images')
            if inst_number[i].max() > 1:
                ids = (inst_number[i]==torch.max(inst_number[i])).nonzero()
                un_id = ids[:,0]
                un_id = un_id.unique()
                print(f'classes that have multiple instances:')
                for k in un_id:
                    print(f'{cls_lis[k]} has {int(torch.eq(ids,k).float().sum())} times {inst_number[i].max()} instances')

def check_ann_img2():
        
    args = parse_args()
    # assign directory
    bizr = []
    
    directory = args.dir

    
    
    for shot in [5,10,20,50,100]:
        print(f'shot: {shot}')
        filel = []

        fann_path = '/home/archeval/mmdetection/CATNet/mmdetection/data/dior/few_shot_ann/hard_benchmark_'+str(shot)+'shot/'
        for classes in (list(ALL_CLASSES_SPLIT1_DIOR)):
            directory = fann_path+'box_'+str(shot)+'shot_'+classes+'_train.txt'

        with open(directory, 'r') as df:
            lines = df.readlines()
        # first u have to open  the file and seperate every line like below:

        # remove /n at the end of each line
        for index, line in enumerate(lines):
            lines[index] = line.strip()
        for file_id, file_name in tqdm(enumerate(lines)):

            xml_path = DIOR_XML + file_name + '.xml'
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

                files = df_result['file'].unique()
                for _, file_current in tqdm(enumerate(files)):
                #for class_to_find in list(ALL_CLASSES_SPLIT1_DIOR):
                    nbr_classes = []
                    #df_of_class = get_images_with_class(class_to_find, df_result)
                    #file_of_class = df_of_class['file'].unique()

                    for class_to_find in list(ALL_CLASSES_SPLIT1_DIOR):
                    #for file_current in file_of_class:
                        a = df_result[(df_result['category'] == class_to_find) & (df_result['file'] == file_current)]
                        print(a)
                        print(dict().shape)
                        #nbr_instance.append((a.shape)[0])
                        if (a.shape)[0] != 0:
                            nbr_classes.append((class_to_find, (a.shape)[0]))

                        #print(f'image {file_current}.xml contains {(a.shape)[0]} time the class {class_to_find}')
                    if args.class_to_find is not None:
                        print(f'{file_current} contain {len(nbr_classes)} classes.')
                        print(f'Those are {nbr_classes[:][0]}')
                        print(f'mean instances {sum(nbr_instance)/len(nbr_instance)}')
                        print(f'unique instances {len(list(np.unique(np.array(nbr_instance))))}')
            else:   
                files = df_result['file'].unique()
                only_one = []
                for _, file_current in tqdm(enumerate(files)):
                #for class_to_find in list(ALL_CLASSES_SPLIT1_DIOR):
                    nbr_classes = []
                    #df_of_class = get_images_with_class(class_to_find, df_result)
                    #file_of_class = df_of_class['file'].unique()

                    for class_to_find in args.class_to_find:
                    #for file_current in file_of_class:
                        a = df_result[(df_result['category'] == class_to_find) & (df_result['file'] == file_current)]
                        #nbr_instance.append((a.shape)[0])
                        if (a.shape)[0] != 0:
                            nbr_classes.append((class_to_find, (a.shape)[0]))

                        #print(f'image {file_current}.xml contains {(a.shape)[0]} time the class {class_to_find}')
                    #if args.class_to_find in nbr_classes:
                    #    nbr_classes = nbr_classes.pop(args.class_to_find)
                    #    if len(nbr_classes)==0:
                    #        only_one.append(file_current)
                if args.class_to_find is not None:
                    print(f'Images {only_one} only contains the class {args.class_to_find} class')

def check_ann_img_faster():
    

    args = parse_args()
    # assign directory
    bizr = []
    
    directory = args.dir

    with open(directory, 'r') as df:
            lines = df.readlines()
        # first u have to open  the file and seperate every line like below:

        # remove /n at the end of each line
    for index, line in enumerate(lines):
        lines[index] = line.strip()
    

    filel = []
    only_one = []
    #df = df.append({'Name' : 'Anna', 'Scores' : 97, 'Questions' : 2200}, 
    #            ignore_index = True)

    #df_instances_per_img = pd.DataFrame(columns=['class', 'instances', 'file'])
    
    for file_id, file_name in tqdm(enumerate(lines)):
        xml_path = DIOR_XML + file_name + '.xml'
        current_file_classes = []
        tree = ET.parse(xml_path)
        root = tree.getroot()
        names = []
        
        
        for obj in root.findall('object'):
            name = obj.find('name').text            
            current_file_classes.append([name, file_name])
            names.append(name)
        unique_obj = list(set(names))
        rep = Counter(names)
        for id_obj, u_obj in enumerate(unique_obj):
            amount = rep[u_obj]
            if not file_id and not id_obj:
                ll = [u_obj, str(amount), file_name]
                ll = np.array(ll).reshape(-1,len(ll))
                df_instances_per_img = pd.DataFrame(ll, columns=['class', 'instances', 'file'])
            else:
                ll = [u_obj, str(amount), file_name]

                ll = np.array(ll).reshape(-1,len(ll))
                df_temp = pd.DataFrame(ll, columns=['class', 'instances', 'file'])                
                df_instances_per_img = pd.concat([df_instances_per_img, df_temp], ignore_index=True, axis=0)

            #df_instances_per_img = df_instances_per_img.append({'class':u_obj, 'instances':amount, 'file':file_name}, ignore_index = True)
        #for i in args.class_to_find:
        #    if i in names and len(names) < 2:
        #        only_one.append(file_name)
    print("part 2")
    for class_to_find in ['trainstation', 'windmill', 'ship']:
        b = df_instances_per_img[(df_instances_per_img['class'] == class_to_find)]  
        #print(b)
        b = b.sort_values(by=['instances'])
        c = b['instances']
        print(c)
        bbb = (b.shape)[0]
        print(bbb)
        fig, ax = plt.subplots(figsize=(20,5))
        c.hist(bins = bbb, ax = ax)
        ax.tick_params(axis='x', labelrotation = 45)
        fig.savefig(f'hist_{class_to_find}.png')
        
        
        #b.savefig("hist_test.png")
        print(class_to_find)
    
def get_images_with_class(class_name, df):
    return df[(df == class_name).any(axis=1)]

def check_image_without_fewshots():
        
    args = parse_args()
    # assign directory
    bizr = []
    
    directory = args.dir

    with open(directory, 'r') as df:
            lines = df.readlines()
        # first u have to open  the file and seperate every line like below:

        # remove /n at the end of each line
    for index, line in enumerate(lines):
        lines[index] = line.strip()
    
    filel = []
    only_one = []
    few_and_val = []
    new_train = []
    for shot in [5, 10, 20, 50, 100]:
        print(f'shot: {shot}')
        for comp in ['hard']:
            print(f'comp: {comp}')
            few_and_val = []
            new_train = []
            save_path = '/home/archeval/mmdetection/CATNet/mmdetection/data/dior/few_shot_ann/'+comp+'_benchmark_'+str(shot)+'shot/'
            for classes in (list(ALL_CLASSES_SPLIT1_DIOR)):
                save_path_tmp = save_path+'box_'+str(shot)+'shot_'+classes+'_train.txt'
                with open(save_path_tmp, 'r') as df:
                    lines_few = df.readlines()
                for index, line_few in enumerate(lines_few):
                    lines_few[index] = line_few.strip()
                for file_id, file_name in tqdm(enumerate(lines_few)):
                    if file_name in lines:
                        few_and_val.append(file_name)
            nn = lines.copy()
            for ii in few_and_val:
                if ii in nn:
                    nn.remove(ii)
            save_path_dior = DIOR_PATH + 'train_' + str(shot) + 'shots.txt'
           # np.savetxt(save_path+'/box_'+str(shot)+'shot_'+curent_class_name+'_train.txt', files_names, fmt='%s')
            with open(save_path_dior, 'w') as f:
                for element in nn:
                    f.write(element + "\n")
    #print(f'Images {len(few_and_val)} are in few shot and val set..')

def check_stuff_faster():
    
    args = parse_args()
    # assign directory
    bizr = []
    
    
    val_n_train = []
    val_not_train = []
    stuff = True
    if not stuff:
        directory = args.dir
        with open(directory, 'r') as df:
                lines = df.readlines()
            # first u have to open  the file and seperate every line like below:

            # remove /n at the end of each line
        for index, line in enumerate(lines):
            lines[index] = line.strip()

        with open(DIOR_VAL, 'r') as df:
                lines_val = df.readlines()
            # first u have to open  the file and seperate every line like below:

            # remove /n at the end of each line

        for index, line in enumerate(lines_val):
            lines_val[index] = line.strip()

        for val_file in lines_val:
            if val_file in lines:
                val_n_train.append(val_file)
            elif val_file not in lines:
                val_not_train.append(val_file)
            else:
                print('coucou')
        print(f'Nbr in both {len(val_n_train)}')
        print(f'Nbr only in val {len(val_not_train)}')

   
    if stuff:
        filel = []
        only_one = []
        few_and_val = []
        new_train = []

        base_name = [DIOR_VAL, DIOR_TEST, DIOR_TRAIN]
        for compare_name in base_name:
            with open(compare_name, 'r') as df:
                    lines_comp = df.readlines()
                # first u have to open  the file and seperate every line like below:

                # remove /n at the end of each line

            for index, line in enumerate(lines_comp):
                lines_comp[index] = line.strip()
            for shot in [10, 100]:
                print(f'shot: {shot}')
                for comp in ['hard']:
                    print(f'comp: {comp}')
                    few_and_val = []
                    new_train = []
                    #save_path_dior = DIOR_PATH + 'train_' + str(shot) + 'shots.txt'
                    all_class_2 = []
                    for cclass in ALL_CLASSES_SPLIT1_DIOR_L:
                        save_path_dior = DIOR_FSANN + comp + '_benchmark_' + str(shot) + 'shot/box_'+str(shot)+'shot_' + cclass + '_train.txt'
                        with open(save_path_dior, 'r') as df:
                            lines_2 = df.readlines()
                        for index, line_2 in enumerate(lines_2):
                            lines_2[index] = line_2.strip()
                        all_class_2 += lines_2
                    print(len(all_class_2))
                    for file_id, file_name in (enumerate(all_class_2)):
                        if file_name in lines_comp:
                            few_and_val.append(file_name)
                        else:
                            new_train.append(file_name)
                    #print(f'new training list length: {len(lines_2)}')
                    #print(f'old training list length: {len(lines)}')
                    print(f'Number of shots: {shot}')
                    print(f'max id is: {max([int(i)  for i in all_class_2])}')
                    print(f'there is {len(few_and_val)} few shot ann that are both in few shot and in {compare_name}')
                    print(f'there is {len(new_train)} few shot ann that are not in few shot and in {compare_name}')
                    #print(f'For {shot} shots there is {len(few_and_val)} image that are in {save_path_dior} and {directory}')
                    #print(f'For {shot} shots there is {len(new_train)} image that are in {directory} but not in {save_path_dior}')

def do_plot_fast():
    import matplotlib.gridspec as grd
    n_mAP_FT = [0, 0.252, 0.3699, 0.4107]
    b_mAP_FT = [0.744, 0.7165, 0.3791, 0.3607]

    n_mAP_TFA = [0, 0.134,0.296]
    b_mAP_TFA = [0.745, 0.705,0.682]
    shot_TFA = [1, 10, 100]
    shot_FT = [1, 10, 100, 150]
    ticks = [1,2,3,4]
    fig, ax = plt.subplots(figsize=(20,5))
    ax.plot(ticks[:-1], n_mAP_TFA, label='Novel Classes', color='red', marker='v')
    ax.plot(ticks[:-1], b_mAP_TFA, label='Base Classes', color='blue', marker='d')
    ax.set_xticks(ticks[:-1])
    ax.set_xticklabels(shot_TFA)
    ax.legend(loc='lower right')
    ax.set_xlabel('Number of shots')
    ax.set_ylabel('mAP')
    ax.set_title("TFA finetuning")
    ax.legend()
    ax.grid()
    #ax.tick_params(axis='x', labelrotation = 45)
    fig.savefig(f'TFA_shot.png')

    fig, ax = plt.subplots(figsize=(20,5))
    ax.plot(ticks, n_mAP_FT, label='Novel Classes', color='red', marker='v')
    ax.plot(ticks, b_mAP_FT, label='Base Classes', color='blue', marker='d')
    ax.set_xticks(ticks)
    ax.set_xticklabels(shot_FT)
    ax.legend(loc='lower right')
    ax.grid()
    #ax.tick_params(axis='x', labelrotation = 45)
    ax.set_xlabel('Number of shots')
    ax.set_ylabel('mAP')
    ax.set_title("FT finetuning")
    fig.savefig(f'FT_shot.png')

def check_val_test_train():
        
    args = parse_args()
    # assign directory
    bizr = []
    plot_type = []
    label_type = []
    plot_list = []
    for type_path in ['train', 'val', 'test']:#, 'VAL', 'TEST']:
        path = DIOR_PATH + type_path + '.txt'
        with open(path, 'r') as df:
            lines = df.readlines()
        for index, line in enumerate(lines):
            lines[index] = line.strip()

        for file_id, file_name in tqdm(enumerate(lines)):
            xml_path = DIOR_XML + file_name + '.xml'
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                bnd_box = obj.find('bndbox')
                xmin = int(float(bnd_box.find('xmin').text))
                ymin = int(float(bnd_box.find('ymin').text))
                xmax = int(float(bnd_box.find('xmax').text))
                ymax = int(float(bnd_box.find('ymax').text))
                size = (xmax - xmin) * (ymax - ymin)
                size = max((xmax - xmin),(ymax - ymin))
                if not file_id:
                    ll = [name, size, 1, file_name]
                    ll = np.array(ll).reshape(-1,len(ll))
                    df = pd.DataFrame(ll, columns=['class', 'size', 'instances', 'file'])
                else:
                    ll = [name, size, 1, file_name]

                    ll = np.array(ll).reshape(-1,len(ll))
                    df_temp = pd.DataFrame(ll, columns=['class', 'size', 'instances', 'file'])                
                    df = pd.concat([df, df_temp], ignore_index=True, axis=0)
        all_inst = []
            
        for class_to_find in ALL_CLASSES_SPLIT1_DIOR:
            b = df[(df['class'] == class_to_find)]  
            b_list = b['size'].tolist()
            b_list = [int(i) for i in b_list]
            plot_list.append(b_list)
        #    c = b['file'].unique().tolist()
        #    print(b)
        #    inst_perimg = []
        #    for file_now in c:
        #        d = b.loc[b['file'] == file_now]
        #        inst_perimg.append(d.shape[0])
        #    all_inst.append(Counter(inst_perimg))
        label_t = [i+'_'+type_path for i in ALL_CLASSES_SPLIT1_DIOR_L]
        label_type = label_type + label_t
        plot_type.append(plot_list)
    
    #print(all_inst)
    #for class_inst in all_inst:

    #    nbr_inst = OrderedDict(sorted(dict(class_inst).items()))
    #    print(nbr_inst)
        

    tmp = []
    tmp_lbl = []
    for i in range(20):
        tmp = tmp + [plot_list[i]] + [plot_list[i+20]] + [plot_list[i+40]]
        tmp_lbl = tmp_lbl + [label_type[i]] + [label_type[i+20]] + [label_type[i+40]]
    fig, ax = plt.subplots(figsize=(20,30))
    ax.boxplot(tmp, whis= [0, 100], vert=False)
    ax.grid()
    ax.set_yticklabels(tmp_lbl)
    fig.savefig(f'plots/4.png')

def check_bbox_nbr():
    
    args = parse_args()
    # assign directory
    bizr = []

    in_loop = False
    for shot in [10]:
        print(f'shot: {shot}')
        for comp in ['hard']:
            print(f'comp: {comp}')
            few_and_val = []
            new_train = []
            #save_path_dior = DIOR_PATH + 'train_' + str(shot) + 'shots.txt'
            all_class_2 = []
            for cclass in ALL_CLASSES_SPLIT1_DIOR_L:
                save_path_dior = DIOR_FSANN + comp + '_benchmark_' + str(shot) + 'shot/box_'+str(shot)+'shot_' + cclass + '_train.txt'
                with open(save_path_dior, 'r') as df:
                    lines = df.readlines()
                    
                for index, line in enumerate(lines):
                    lines[index] = line.strip()

                for file_id, file_name in tqdm(enumerate(lines)):
                    xml_path = DIOR_XML + file_name + '.xml'
                    current_file_classes = []
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    names = []
                    
                    for obj in root.findall('object'):
                        name = obj.find('name').text            
                        current_file_classes.append([name, file_name])
                        names.append(name)

                    unique_obj = list(set(names))
                    rep = Counter(names)
                    total_instance_in_image = len(names)
                    for id_obj, u_obj in enumerate(unique_obj):
                        amount = rep[u_obj]
                        if not in_loop:
                            ll = [u_obj, str(amount), file_name]
                            ll = np.array(ll).reshape(-1,len(ll))
                            df_instances_per_img = pd.DataFrame(ll, columns=['class', 'instances', 'file'])
                            ll = [total_instance_in_image, file_name]
                            ll = np.array(ll).reshape(-1,len(ll))
                            df_all_instances_per_img = pd.DataFrame(ll, columns=['all_instancess', 'file'])
                            in_loop = True
                        else:
                            ll = [u_obj, str(amount), file_name]

                            ll = np.array(ll).reshape(-1,len(ll))
                            df_temp = pd.DataFrame(ll, columns=['class', 'instances', 'file'])                
                            df_instances_per_img = pd.concat([df_instances_per_img, df_temp], ignore_index=True, axis=0)
                            ll = [total_instance_in_image, file_name]
                            ll = np.array(ll).reshape(-1,len(ll))

                            df_temp = pd.DataFrame(ll, columns=['all_instancess', 'file'])       
                            df_all_instances_per_img = pd.concat([df_all_instances_per_img, df_temp], ignore_index=True, axis=0)
    

    b = df_all_instances_per_img['all_instancess']  
    #print(b)
    b = b.sort_values()
    uni_id = b.shape[0]
    bl = df_all_instances_per_img['all_instancess'].str.split(',',expand=True).astype(int).values.tolist()
    print(bl)
    inst_per_id = []
    unique_id_l = []
    for id,i in enumerate(b.unique()):
        unique_id_l.append(int(i))
        inst_per_id.append(0)
        for j in range(uni_id):
            if bl[j][0] == int(i):
                inst_per_id[id] += 1


    fig, ax = plt.subplots(figsize=(25,5))
    #plt.scatter(unique_id_l, inst_per_id)
    ax.stem(unique_id_l, inst_per_id)
    ax.tick_params(axis='x', labelrotation = 90)
    plt.xticks(np.arange(min(unique_id_l), max(unique_id_l)+1, 1.0))
    fig.savefig(f'hist_all_inst.png')

    print(df_all_instances_per_img)
    
def pickled_dict():
    read = True
    if not read:
        data = {1: 0, 2:1, 3:2, 4:3, 5:4}
        with open('data.p', 'wb') as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('init_queue.p', 'rb') as fp:
            data = pickle.load(fp)
    
    queue_base_res, queue_base_trg, queue_novel_res, queue_novel_trg, queue_aug_res, queue_aug_trg = [], [], [], [], [], []
    for key in data.keys():
        queue_base_res.append(data[key]['base_results'])
        queue_base_trg.append(data[key]['base_trg'])
        queue_novel_res.append(data[key]['novel_results'])
        queue_novel_trg.append(data[key]['novel_trg'])
        queue_aug_res.append(data[key]['aug_results'])
        queue_aug_trg.append(data[key]['aug_trg'])

    queue_base_res = queue_base_res + queue_novel_res + queue_aug_res
    queue_base_res = torch.stack(queue_base_res)

    queue_base_trg = queue_base_trg + queue_novel_trg + queue_aug_trg
    queue_base_trg = torch.stack(queue_base_trg)
    #queue_novel_res = torch.stack(queue_novel_res)
    #queue_novel_trg = torch.stack(queue_novel_trg)
    #queue_aug_res = torch.stack(queue_aug_res)
    #queue_aug_trg = torch.stack(queue_aug_trg)

    print(f'shapes {queue_base_res.shape} trg {queue_base_trg.shape}')
    #print(f'shapes {queue_novel_res.shape} trg {queue_novel_trg.shape}')
    #print(f'shapes {queue_aug_res.shape} trg {queue_aug_trg.shape}')
    a = queue_base_res, queue_base_trg#, queue_novel_res, queue_novel_trg, queue_aug_res, queue_aug_trg
    return a

def check_xview_train(file_save_path, file_save_path_val):
    import os.path
    args = parse_args()

    
    # assign directory
    x_path = xview_base + 'xView_train.geojson'
    directory = args.dir

    import json
    with open(x_path) as f:
        data = json.load(f)

    ordered_lis = []
    l_dict = []
    classes_dict = {}
    classes_file_nbr = {}
    class_cntr = {}
    truck_cntr = {}
    truck_files = {}
    class_files = {}
    for class_key in xView_classes.keys():

        class_files[class_key] = []
        classes_dict[class_key] = 0
        if class_key in truck_split:
            continue
        classes_file_nbr[class_key] = 0
        class_cntr[class_key] = 0
    for class_key in truck_split:
        truck_files[class_key] = []
        truck_cntr[class_key] = 0
    no_truck = []
    #if False:
    for nbr_img in range(len(data['features'])): 
        dict_cur = {}
        current_class = data['features'][nbr_img]['properties']['type_id'] 
        img_path = xview_img + data['features'][nbr_img]['properties']['image_id']
        
        if not os.path.isfile(img_path):
            continue
        dict_cur['class_id'] = current_class
        if current_class in truck_split:
            truck_cntr[current_class] += 1
        
        classes_dict[current_class] += 1
        current_file = data['features'][nbr_img]['properties']['image_id']
        if current_class in truck_split:
            if current_file not in truck_files[current_class]:
                truck_files[current_class].append(current_file)
        if current_file not in class_files[current_class]:
                class_files[current_class].append(current_file)
                no_truck.append(current_file[:-4])
        dict_cur['img_id'] = current_file 
        l_dict.append(dict_cur)
        ordered_lis.append(int(data['features'][nbr_img]['properties']['image_id'][:-4]))
    
    sorted_id = sorted(range(len(ordered_lis)), key=ordered_lis.__getitem__)
    train,val = [],[]
    ordr_tensor = torch.tensor(ordered_lis)
    yy = ordr_tensor.unique()
    
    if False:
        rnd_id = [i for i in range(yy.shape[0])]
        rnd_val = np.random.choice(rnd_id, int(0.2*len(rnd_id)))
        for rnd_id in range(yy.shape[0]):
            if rnd_id in rnd_val:
                val.append(yy[rnd_id])
            else:
                train.append(yy[rnd_id])
    train,val = [],[]
    rnd_id = [i for i in range(yy.shape[0])]
    rnd_val = np.random.choice(rnd_id, int(0.2*len(rnd_id)))
    for rnd_id in range(yy.shape[0]):
        if rnd_id in rnd_val:
            val.append(rnd_id)
        else:
            train.append(rnd_id)

    ordered_lis = []
    l_dict = []
    cntr_val = {}
    cntr_train = {}
    for class_key in xView_classes.keys():
        cntr_train[class_key] = 0
        cntr_val[class_key] = 0
    if False:  
        for nbr_img in tqdm(range(len(data['features']))):
            current_class = data['features'][nbr_img]['properties']['type_id'] 

            current_file = int(data['features'][nbr_img]['properties']['image_id'][:-4])
            if current_file in val:
                cntr_val[current_class] += 1
            else:
                cntr_train[current_class] += 1
                
        print(cntr_val, cntr_train)
        
    with open(file_save_path, 'w') as f_train:
        for save_train in train:
            if type(save_train) is int:
                f_train.write("%s\n" % save_train)
            else:
                f_train.write("%s\n" % save_train.item())
    with open(file_save_path_val, 'w') as f_val:
        for save_val in val:
            if type(save_val) is int:
                f_val.write("%s\n" % save_val)
            else:
                f_val.write("%s\n" % save_val.item())
                    
def check_xview():
    import os.path


    args = parse_args()
    # assign directory
    x_path = xview_base + 'xView_train.geojson'
    directory = args.dir

    import json
    with open(x_path) as f:
        data = json.load(f)

    ordered_lis = []
    l_dict = []
    for nbr_img in range(len(data['features'])):
        dict_cur = {}
        dict_cur['bbox_coord'] = data['features'][nbr_img]['properties']['bounds_imcoords']

        dict_cur['class_id'] = data['features'][nbr_img]['properties']['type_id'] 
        dict_cur['img_id'] = data['features'][nbr_img]['properties']['image_id']
        l_dict.append(dict_cur)
        ordered_lis.append(int(data['features'][nbr_img]['properties']['image_id'][:-4]))
    sorted_id = sorted(range(len(ordered_lis)), key=ordered_lis.__getitem__)


    sorted_dict = []
    for sorting_id in sorted_id:
        sorted_dict.append(l_dict[sorting_id])
    
    sorted_tensor = torch.tensor(sorted_id).unique()
    i_step = 0
    i_same_img = 0
    done = False
    skipped = 0
    for i_xml in range(sorted_tensor.shape[0]):
        
        save_path = xview_xml + str(i_xml-skipped) + '.xml'#'testing_xview/' + str(i_xml) + '.xml'
        data = ET.Element('annotation')
        # Adding a subtag named `Opening`
        # inside our root tag
        file_xml = ET.SubElement(data, 'filename')
        file_xml.text = sorted_dict[i_step+i_xml]['img_id']

        
        img_path = xview_img + sorted_dict[i_step+i_xml]['img_id']
        
        if os.path.isfile(img_path):
            im = Image.open(img_path)
            
            w, h = im.size

            src = ET.SubElement(data, 'source')
            src_db = ET.SubElement(src, 'database')
            src_db.text = 'xview'
            size = ET.SubElement(data, 'size')
            size_w = ET.SubElement(size, 'width')
            size_h = ET.SubElement(size, 'height')
            size_d = ET.SubElement(size, 'depth')
            size_w.text = str(w)
            size_h.text = str(h)
            size_d.text = "3"

            seg = ET.SubElement(data, 'segmented')
            seg.text='0'
            old_id = sorted_dict[i_step+i_xml]['img_id']
            
            
            i_same_img = 0
            while sorted_dict[i_same_img+i_step+i_xml]['img_id'] == old_id:
                id_object = i_same_img+i_step+i_xml
                #print(f'image id {id_object}')
                obj_xml = ET.SubElement(data, 'object')
                obj_name = ET.SubElement(obj_xml, 'name')
                obj_name.text = xView_classes[sorted_dict[id_object]['class_id']]
                obj_pose = ET.SubElement(obj_xml, 'pose')
                obj_pose.text = "Unspecified"
                obj_bndbbox = ET.SubElement(obj_xml, 'bndbox')
                obj_xmin = ET.SubElement(obj_bndbbox, 'xmin')
                coord = sorted_dict[id_object]['bbox_coord']
                conv_coord = []
                for id_coord in range(len(coord)):
                    if coord[id_coord] == ',':
                        conv_coord.append(id_coord)
                obj_xmin.text = sorted_dict[id_object]['bbox_coord'][:conv_coord[0]]
                obj_ymin = ET.SubElement(obj_bndbbox, 'ymin')
                obj_ymin.text = sorted_dict[id_object]['bbox_coord'][conv_coord[0]+1:conv_coord[1]]
                obj_xmax = ET.SubElement(obj_bndbbox, 'xmax')
                obj_xmax.text = sorted_dict[id_object]['bbox_coord'][conv_coord[1]+1:conv_coord[2]]
                obj_ymax = ET.SubElement(obj_bndbbox, 'ymax')
                obj_ymax.text = sorted_dict[id_object]['bbox_coord'][conv_coord[2]+1:]


                old_id = sorted_dict[id_object]['img_id']
                i_same_img += 1
                if len(sorted_dict) == i_same_img+i_step+i_xml:
                    done = True
                    break
            
            i_step += (i_same_img-1)
            
            # Converting the xml data to byte object,
            # for allowing flushing data to file
            # stream
            b_xml = ET.tostring(data)
            
            # Opening a file under the name `items2.xml`,
            # with operation mode `wb` (write + binary)
            #print(save_path)
            with open(save_path, "wb") as f:
                f.write(b_xml)
            if done:
                break
        else:
            skipped += 1

def create_few_shot_ann_xview():
    import os.path
        
    args = parse_args()
    # assign directory
    
    directory = args.dir
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

        xml_path = xview_xml + file_name + '.xml'
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
            for class_to_find in list(ALL_CLASSES_xview):
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
                print(curent_class_name)
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
                    #print(df_novel)
                    #print("======")
                    if isinstance(df_novel, str):
                        data = np.array([df_novel['file'].unique()]).T
                    else: 
                        data = np.array([df_novel['file']])

                files_names = np.stack(data, axis=0 )
                file_cycle = cycle(files_names)
                files_names = [next(file_cycle) for i in range(shot)]
                used_file += files_names
                #creating a new directory 
                save_path = xview_base+'few_shot_ann/'+args.few_shot_ann+'_benchmark_'+str(shot)+'shot'
                Path(save_path).mkdir(parents=True, exist_ok=True)
                #np.savetxt(save_path+'/box_'+str(shot)+'shot_'+curent_class_name+'_train.txt', files_names, fmt='%s')
                with open(save_path+'/box_'+str(shot)+'shot_'+curent_class_name+'_train.txt', 'w') as f:
                    for element in files_names[0]:
                        f.write(element + "\n")

def reform_dior(path,file_save_path=None, file_save_path_val=None, val_ratio=.2):
    
    #directory = path
    nbr_id = 0
    train,val = [],[]
    rnd_id = []
    rnd_id_name = []
    print(path)
    for directory in path:
        print(directory)
        with open(directory, 'r') as df:
            lines = df.readlines()
            # first u have to open  the file and seperate every line like below:

            # remove /n at the end of each line
        for index, line in enumerate(lines):
            lines[index] = line.strip()

        filel = []
        wh_files = []
        xml_cntr = 0
        nbr_new_file = 0
        not_changed = 0
        for file_id, file_name in tqdm(enumerate(lines)):
            
            xml_path = DIOR_XML + file_name + '.xml'
            tree = ET.parse(xml_path)
            root = tree.getroot()
            save_name = root.find('filename').text
            size_init = root.find('size')
            
            data = ET.Element('annotation')
            # Adding a subtag named `Opening`
            # inside our root tag
            file_xml = ET.SubElement(data, 'filename')
            file_xml.text = save_name

            src = ET.SubElement(data, 'source')
            src_db = ET.SubElement(src, 'database')
            src_db.text = 'DIOR'
            size = ET.SubElement(data, 'size')
            size_w = ET.SubElement(size, 'width')
            size_h = ET.SubElement(size, 'height')
            size_d = ET.SubElement(size, 'depth')
            size_w.text = size_init.find('width').text
            size_h.text = size_init.find('height').text
            size_d.text = "3"

            seg = ET.SubElement(data, 'segmented')
            seg.text='0'
            changed_any = False
            for obj in root.findall('object'):
                bnd_box = obj.find('bndbox')
                xmin = int(float(bnd_box.find('xmin').text))
                ymin = int(float(bnd_box.find('ymin').text))
                xmax = int(float(bnd_box.find('xmax').text))
                ymax = int(float(bnd_box.find('ymax').text))
                
                    
                obj_xml = ET.SubElement(data, 'object')
                obj_name = ET.SubElement(obj_xml, 'name')
                tmp_name  = obj.find('name').text
                obj_name.text = tmp_name
                obj_pose = ET.SubElement(obj_xml, 'pose')
                obj_pose.text = "Unspecified"
                obj_bndbbox = ET.SubElement(obj_xml, 'bndbox')
                obj_xmin = ET.SubElement(obj_bndbbox, 'xmin')                        
                obj_xmin.text = str(xmin)
                obj_ymin = ET.SubElement(obj_bndbbox, 'ymin')
                obj_ymin.text = str(ymin)
                obj_xmax = ET.SubElement(obj_bndbbox, 'xmax')
                obj_xmax.text = str(xmax)
                obj_ymax = ET.SubElement(obj_bndbbox, 'ymax')
                obj_ymax.text = str(ymax)

            b_xml = ET.tostring(data)
            
            # Opening a file under the name `items2.xml`,
            # with operation mode `wb` (write + binary)
            #print(save_path)
            xml_save_path = DIOR_XML_BIG + save_name[:-4] + '.xml'
            rnd_id_name.append(save_name[:-4])
            rnd_id.append(nbr_id)
            with open(xml_save_path, "wb") as f:
                f.write(b_xml)
            nbr_id += 1
            
    if file_save_path is not None and file_save_path_val is not None:
        rnd_val = np.random.choice(rnd_id, int(val_ratio*len(rnd_id)))
        for append_rnd_id in range(len(rnd_id)):
            if append_rnd_id in rnd_val:
                val.append(rnd_id_name[append_rnd_id])
            else:
                train.append(rnd_id_name[append_rnd_id])
        with open(file_save_path, 'w') as f_train:
            for save_train in train:
                if type(save_train) is int:
                    f_train.write("%s\n" % save_train)
                else:
                    f_train.write("%s\n" % save_train)
        with open(file_save_path_val, 'w') as f_val:
            for save_val in val:
                if type(save_val) is int:
                    f_val.write("%s\n" % save_val)
                else:
                    f_val.write("%s\n" % save_val)
        
def create_tiles_few_shot_ann_dior(directory=None, version_ann='v2', nbr_shots=[10],big=False, save=True):
    import os.path
        
    args = parse_args()
    print(directory)
    # assign directory
    if directory is None:
        directory = args.dir

    nbr_shots = nbr_shots
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
                    #print(df_novel)
                    #print("======")
                    if isinstance(df_novel, str):
                        data = np.array([df_novel['file'].unique()]).T
                    else: 
                        data = np.array([df_novel['file']])

                files_names = np.stack(data, axis=0 )
                file_cycle = cycle(files_names)
                files_names = [next(file_cycle) for i in range(shot)]
                used_file += files_names
                if save:
                #creating a new directory 
                    if big:
                        save_path = DIOR_FSANN_BIG+version_ann+args.few_shot_ann+'_benchmark_'+str(shot)+'shot'
                    else:
                        Path(DIOR_FSANN+version_ann).mkdir(parents=True, exist_ok=True)
                        save_path = DIOR_FSANN+version_ann+args.few_shot_ann+'_benchmark_'+str(shot)+'shot'
                    Path(save_path).mkdir(parents=True, exist_ok=True)
                    #np.savetxt(save_path+'/box_'+str(shot)+'shot_'+curent_class_name+'_train.txt', files_names, fmt='%s')
                    if 'train' in directory:
                        with open(save_path+'/box_'+str(shot)+'shot_'+curent_class_name+'_train.txt', 'w') as f:
                            for element in files_names[0]:
                                f.write(element + "\n")
                    elif 'val' in directory:
                        with open(save_path+'/box_'+str(shot)+'shot_'+curent_class_name+'_val.txt', 'w') as f:
                            for element in files_names[0]:
                                f.write(element + "\n")
                    else:
                        print('problem in directory selected. Not train nor validation')

def do_nice_graphs(input_path):

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

def check_xview_ann(dir):
    #args = parse_args()
    # assign directory
    
    directory = dir

    with open(directory, 'r') as df:
            lines = df.readlines()
        # first u have to open  the file and seperate every line like below:

        # remove /n at the end of each line
    for index, line in enumerate(lines):
        lines[index] = line.strip()

    filel = []
    wh_files = []
    for file_id, file_name in tqdm(enumerate(lines)):

        xml_path = xview_xml + file_name + '.xml'
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for s in root.findall('size'):
            wh_files.append([s.find('width').text,s.find('height').text])
    print(wh_files)
    
def tile_xview(path,img_scale=[800,800],overlap_ratio=.025,file_save_path=None, file_save_path_val=None, val_ratio=.2):
    directory = path

    with open(directory, 'r') as df:
            lines = df.readlines()
        # first u have to open  the file and seperate every line like below:

        # remove /n at the end of each line
    for index, line in enumerate(lines):
        lines[index] = line.strip()

    filel = []
    wh_files = []
    xml_cntr = 0
    nbr_new_file = 0
    not_changed = 0
    train,val = [],[]
    rnd_id = []
    for file_id, file_name in tqdm(enumerate(lines)):
        
        xml_path = xview_xml + file_name + '.xml'
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for s in root.findall('size'):
            save_name = root.find('filename').text

            w=int(s.find('width').text)
            h=int(s.find('height').text)
            img_path = xview_img + save_name

            if os.path.isfile(img_path):
                img = Image.open(img_path)
                img = np.array(img)
                assert img_scale[0] < w or img_scale[1] < h

               # N_w = int((w/(overlap_ratio*img_scale[0])-1)//3)
                #N_h = int((h/(overlap_ratio*img_scale[1])-1)//3)
                N_w = int((w-img_scale[0])//((1-overlap_ratio)*img_scale[0])+1)
                N_h = int((h-img_scale[0])//((1-overlap_ratio)*img_scale[0])+1)
                #print(w,h, N_w, N_h)
                # crop bboxes accordingly and clip to the image boundary
                

                for nh in range(1, N_h+1):
                    for nw in range(1, N_w+1):
                        nbr_new_file +=1
                        ymin_crop=int((nh-1)*(1-overlap_ratio)*img_scale[1])
                        xmin_crop=int((nw-1)*(1-overlap_ratio)*img_scale[0])
                        xmax_crop = int(xmin_crop+img_scale[1])
                        ymax_crop = int(ymin_crop+img_scale[0])


                        img_crop = img[ymin_crop:ymax_crop,xmin_crop:xmax_crop,:]                        
                        
                        save_crop_name = save_name[:-4] + '_' + str(nw) + '_'+str(nh)+'.tif'
                        save_path = xview_img + 'test_tile/'+ save_crop_name
                        img_crop = Image.fromarray(img_crop)
                        img_crop.save(save_path)
                
                    
                        data = ET.Element('annotation')
                        # Adding a subtag named `Opening`
                        # inside our root tag
                        file_xml = ET.SubElement(data, 'filename')
                        file_xml.text = save_crop_name

                        src = ET.SubElement(data, 'source')
                        src_db = ET.SubElement(src, 'database')
                        src_db.text = 'xview'
                        size = ET.SubElement(data, 'size')
                        size_w = ET.SubElement(size, 'width')
                        size_h = ET.SubElement(size, 'height')
                        size_d = ET.SubElement(size, 'depth')
                        size_w.text = str(img_scale[0])
                        size_h.text = str(img_scale[1])
                        size_d.text = "3"

                        seg = ET.SubElement(data, 'segmented')
                        seg.text='0'
                        changed_any = False
                        for obj in root.findall('object'):
                            bnd_box = obj.find('bndbox')
                            xmin_old = int(float(bnd_box.find('xmin').text))
                            ymin_old = int(float(bnd_box.find('ymin').text))
                            xmax_old = int(float(bnd_box.find('xmax').text))
                            ymax_old = int(float(bnd_box.find('ymax').text))
                            #print(xmin_old, ymin_old, xmax_old, ymax_old)
                            if xmin_old == xmax_old or ymin_old == ymax_old:
                                continue
                            changed_bbox = False
                            if (xmin_crop <= xmin_old < xmax_crop and (ymin_crop <= ymin_old < ymax_crop or ymin_crop < ymax_old <= ymax_crop)) or (xmin_crop < xmax_old <= xmax_crop and (ymin_crop < ymax_old <= ymax_crop or ymin_crop <= ymin_old < ymax_crop)):
                                xmin = np.clip(xmin_old, xmin_crop, xmax_crop-1)
                                ymin = np.clip(ymin_old, ymin_crop, ymax_crop-1)
                                xmax = np.clip(xmax_old, xmin_crop, xmax_crop-1)
                                ymax = np.clip(ymax_old, ymin_crop, ymax_crop-1)
                                if int(ymin) == int(ymax) or int(xmin) == int(xmax):
                                 #   print("======")
                                  #  print(xmin_old, ymin_old, xmax_old, ymax_old)
                                   # print(xmin_crop, ymin_crop, xmax_crop, ymax_crop)
                                    #print(xmin, ymin, xmax, ymax)
                                    pass
                                else:
                                    changed_bbox = True
                                
                            if changed_bbox:
                              #  print(f'x old {xmin_old, ymin_old, xmax_old, ymax_old}')
                               # print(f'xmin {xmin,ymin,xmax,ymax}')
                                #print(f'crop {xmin_crop, ymin_crop, xmax_crop, ymax_crop}')
                                
                                xmin = int(xmin-xmin_crop)#min((nw-1),1)*img_scale[0]*(1-overlap_ratio)-max((nw-2),0)*img_scale[0])
                                xmax = int(xmax-min((nw-1),1)*xmin_crop)#min((nw-1),1)*img_scale[0]*(1-overlap_ratio)-max((nw-2),0)*img_scale[0])
                                ymin = int(ymin-ymin_crop)#min((nh-1),1)*img_scale[1]*(1-overlap_ratio)-max((nh-2),0)*img_scale[1])
                                ymax = int(ymax-min((nh-1),1)*ymin_crop)#min((nh-1),1)*img_scale[1]*(1-overlap_ratio)-max((nh-2),0)*img_scale[1])
                                #print(f'xmin crop {xmin,ymin,xmax,ymax}')
                                
                                obj_xml = ET.SubElement(data, 'object')
                                obj_name = ET.SubElement(obj_xml, 'name')
                                tmp_name  = obj.find('name').text
                                obj_name.text = tmp_name
                                obj_pose = ET.SubElement(obj_xml, 'pose')
                                obj_pose.text = "Unspecified"
                                obj_bndbbox = ET.SubElement(obj_xml, 'bndbox')
                                obj_xmin = ET.SubElement(obj_bndbbox, 'xmin')                        
                                obj_xmin.text = str(xmin)
                                obj_ymin = ET.SubElement(obj_bndbbox, 'ymin')
                                obj_ymin.text = str(ymin)
                                obj_xmax = ET.SubElement(obj_bndbbox, 'xmax')
                                obj_xmax.text = str(xmax)
                                obj_ymax = ET.SubElement(obj_bndbbox, 'ymax')
                                obj_ymax.text = str(ymax)
                                changed_any = True

                                #img_crop = np.asarray(img_crop)
                                #img_c_tmp[ymin:ymax,xmin,2] = 255
                                #img_c_tmp = img_crop.copy()
                                #img_c_tmp[ymin:ymax,xmax,2] = 255
                                #img_c_tmp[ymax,xmin:xmax,2] = 255
                                #img_c_tmp[ymin,xmin:xmax,2] = 255

                            #print(xmin_old, ymin_old, xmax_old, ymax_old)
                            #print("====")

                            
                        #cv2.imwrite('plots/'+str(file_id)+'img_in_'+str(nw) + '_'+str(nh)+'.jpg', img_tmp)  
                        #cv2.imwrite('plots/'+str(file_id)+'img_crop_'+str(nw) + '_'+str(nh)+'.jpg', img_c_tmp)  
                        b_xml = ET.tostring(data)
                        if not changed_any:
                            not_changed += 1
                        if changed_any:
                        # Opening a file under the name `items2.xml`,
                        # with operation mode `wb` (write + binary)
                        #print(save_path)
                            xml_save_path = xview_xml_tile + str(xml_cntr) + '.xml'
                            rnd_id.append(xml_cntr)
                            with open(xml_save_path, "wb") as f:
                                f.write(b_xml)
                            xml_cntr += 1

    print(f'nbr tiles {nbr_new_file}')
    print(f'nbr tiles without object {not_changed}')
    print(f'nbr tiles with annotation {nbr_new_file-not_changed}')
    if file_save_path is not None and file_save_path_val is not None:
        rnd_val = np.random.choice(rnd_id, int(val_ratio*len(rnd_id)))
        for append_rnd_id in range(len(rnd_id)):
            if append_rnd_id in rnd_val:
                val.append(append_rnd_id)
            else:
                train.append(append_rnd_id)
        with open(file_save_path, 'w') as f_train:
            for save_train in train:
                if type(save_train) is int:
                    f_train.write("%s\n" % save_train)
                else:
                    f_train.write("%s\n" % save_train.item())
        with open(file_save_path_val, 'w') as f_val:
            for save_val in val:
                if type(save_val) is int:
                    f_val.write("%s\n" % save_val)
                else:
                    f_val.write("%s\n" % save_val.item())
    
def create_tiles_few_shot_ann_xview(nbr_shots=[10]):
    import os.path
        
    args = parse_args()
    # assign directory
    
    directory = args.dir
    nbr_shots = nbr_shots
    print(nbr_shots)
    
    with open(directory, 'r') as df:
            lines = df.readlines()
        # first open the file and seperate every line like below:
        # remove /n at the end of each line
    for index, line in enumerate(lines):
        lines[index] = line.strip()
    

    filel = []
    for file_id, file_name in tqdm(enumerate(lines)):

        xml_path = xview_xml_tile + file_name + '.xml'
        print(xml_path)
        if os.path.isfile(xml_path):
            current_file_classes = []
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                file_name_xml =root.find('filename').text
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
            for class_to_find in list(ALL_CLASSES_xview):
                
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
                print(curent_class_name)
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
                    #print(df_novel)
                    #print("======")
                    if isinstance(df_novel, str):
                        data = np.array([df_novel['file'].unique()]).T
                    else: 
                        data = np.array([df_novel['file']])

                files_names = np.stack(data, axis=0 )
                file_cycle = cycle(files_names)
                files_names = [next(file_cycle) for i in range(shot)]
                used_file += files_names
                #creating a new directory 
                save_path = xview_base+'few_shot_ann/tiles/'+args.few_shot_ann+'_benchmark_'+str(shot)+'shot'
                Path(save_path).mkdir(parents=True, exist_ok=True)
                #np.savetxt(save_path+'/box_'+str(shot)+'shot_'+curent_class_name+'_train.txt', files_names, fmt='%s')
                if 'train' in directory:
                    with open(save_path+'/box_'+str(shot)+'shot_'+curent_class_name+'_train.txt', 'w') as f:
                        for element in files_names[0]:
                            f.write(element + "\n")
                elif 'val' in directory:
                    with open(save_path+'/box_'+str(shot)+'shot_'+curent_class_name+'_val.txt', 'w') as f:
                        for element in files_names[0]:
                            f.write(element + "\n")
                else:
                    print('problem in directory selected. Not train nor validation')

def check_bbox_per_file(version_ann=None):
    
    args = parse_args()
    # assign directory
    bizr = []
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
            #save_path_dior = DIOR_PATH + 'train_' + str(shot) + 'shots.txt'
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

def check_xview_ann_tile(path_train, path_val):

    with open(path_train, 'r') as df:
            lines_train = df.readlines()
        # first u have to open  the file and seperate every line like below:
        # remove /n at the end of each line
    for index, line in enumerate(lines_train):
        lines_train[index] = line.strip()
    
    with open(path_val, 'r') as df:
            lines_val = df.readlines()
        # first u have to open  the file and seperate every line like below:
        # remove /n at the end of each line
    for index, line in enumerate(lines_val):
        lines_val[index] = line.strip()

    train,val = [],[]
    double_val_train = []
    for file_id, file_name in tqdm(enumerate(lines_train)):
        train.append(file_name)
    for file_id, file_name in tqdm(enumerate(lines_val)):
        val.append(file_name)

    for val_file in val:
        if val_file in train:
            double_val_train.append(val_file)
        
    print(double_val_train)
    print(f'there is {len(double_val_train)} file in val and train')
    if False:
        xml_path = xview_xml + file_name + '.xml'
        tree = ET.parse(xml_path)
        root = tree.getroot()
        s = root.find('size')
        save_name = root.find('filename').text

        w=int(s.find('width').text)
        h=int(s.find('height').text)
    
        data = ET.Element('annotation')
        # Adding a subtag named `Opening`
        # inside our root tag
        
        for obj in root.findall('object'):
            bnd_box = obj.find('bndbox')
            xmin_old = int(float(bnd_box.find('xmin').text))
            ymin_old = int(float(bnd_box.find('ymin').text))
            xmax_old = int(float(bnd_box.find('xmax').text))
            ymax_old = int(float(bnd_box.find('ymax').text))

def exp4_analysis(file):
    exp_result = pd.read_excel(file)

    no_bg = exp_result.set_index(['random base', 'background base', 'random ft'])

    for w_reg in [1,10]:
        high_weight = no_bg.loc[(no_bg['w reg'] == w_reg)] 
        high_weight_wbg = high_weight.loc[(high_weight['background ft'] == 1)]
        high_weight_nobg = high_weight.loc[(high_weight['background ft'] == 0)]
        comp = high_weight_wbg['novel map']*100 - high_weight_nobg['novel map']*100
        comp.name = 'with bg bigger than no bg for wreg equal to ' + str(w_reg)
        print(comp)
        print("======")
    no_bg = exp_result.set_index(['random base', 'background base', 'random ft', 'background ft'])
    low_weight = no_bg.loc[(no_bg['w reg'] == 1)] 
    high_weight = no_bg.loc[(no_bg['w reg'] == 10)] 
    comp = low_weight['novel map']*100 - high_weight['novel map']*100
    comp.name = 'with lower w vs higher w'
    #print(comp)
    print("======")

    no_rndbase = exp_result.set_index(['background base', 'random ft', 'w reg', 'background ft'])
    low_weight = no_rndbase.loc[(no_rndbase['random base'] == 1)] 
    high_weight = no_rndbase.loc[(no_rndbase['random base'] == 0)] 
    comp = low_weight['novel map']*100 - high_weight['novel map']*100
    comp.name = 'random base q - pc base q '
    #print(comp)
    print("======")
    no_bg = exp_result.set_index(['novel map'])
    no_bg = no_bg.sort_values('novel map',ascending=False)
    no_bg_dec = no_bg.loc[(no_bg['random base'] == 1)] 
    no_bg.name = 'sorted'
    print(no_bg_dec)
    print("======")

if __name__ == '__main__':
    #check_image_without_fewshots()
  #  check_stuff_faster()
    #check_val_test_train()

    #check_bbox_nbr()
    #check_ann_img()
    #create_few_shot_ann()
    #b = pickled_dict()
    #queue_base_res, queue_base_trg, queue_novel_res, queue_novel_trg, queue_aug_res, queue_aug_trg = b
    #queue_res, queue_trg = b
    #print(f'shapes {queue_res.shape} trg {queue_trg.shape}')
    #print(f'shapes {queue_novel_res.shape} trg {queue_novel_trg.shape}')
    #print(f'shapes {queue_aug_res.shape} trg {queue_aug_trg.shape}')
    #l = np.arange(queue_res.shape[0])
    #np.random.shuffle(l)
    #queue_trg_shuffle = queue_trg[l]

    #check_bbox_per_file()

    # ==============
    # xView dataset

    #check_xview()
    #check_xview_train(xview_train, xview_val)
    #create_few_shot_ann_xview()
    #
    #tile_xview(path=xview_train,img_scale=[800,800],overlap_ratio=.25,file_save_path=xview_train_tile, file_save_path_val=xview_val_tile, val_ratio=.2)
    #
    #create_tiles_few_shot_ann_xview(nbr_shots=[10, 20, 100])

    # ==============
    # BIG dior
    #reform_dior(path=[DIOR_TRAIN, DIOR_VAL, DIOR_TEST],file_save_path=DIOR_TRAIN_BIG, file_save_path_val=DIOR_VAL_BIG, val_ratio=.25)
    #create_tiles_few_shot_ann_dior(DIOR_TRAIN_BIG, nbr_shots=[5, 10, 20, 100])
    #check_bbox_per_file(version_ann=None)
    #create_tiles_few_shot_ann_dior(directory=DIOR_TRAIN, version_ann='v3/', nbr_shots=[10],big=False, save=False)
    #check_bbox_per_file(version_ann='v2/')
    #check_xview_ann_tile(xview_train_tile, xview_val_tile)

    exp4_analysis('exp5_res.xlsx')



    # VISUAL
    # do_nice_graphs([DIOR_TRAIN, DIOR_VAL])
