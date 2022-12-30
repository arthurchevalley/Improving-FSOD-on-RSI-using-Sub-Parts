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

xView_classes = {
    11:'Fixed-wingAircraft', 12:'SmallAircraft',13:'CargoPlane',15:'Helicopter',17:'PassengerVehicle',
    18:'SmallCar',19:'Bus',20:'PickupTruck',21:'UtilityTruck',23:'Truck',24:'CargoTruck',25:'TruckTractorWBoxTrailer',
    26:'TruckTractor',27:'Trailer',28:'TruckTractorWFlatbedTrailer',29:'TruckTractorWLiquidTank',32:'crane-truck',
    33:'RailwayVehicle',34:'PassengerCar',35:'Cargo',36:'Flat-Car',37:'Tank-car',38:'Locomotive',40:'MaritimeVessel',
    41:'Motorboat',42:'Sailboat',44:'Tugboat',45:'Barge',47:'FishingVessel',49:'Ferry',50:'Yacht',51:'ContainerShip',
    52:'OilTanker',53:'EngineeringVehicle',54:'tower-crane',55:'container-crane',56:'ReachStacker',57:'StraddleCarrier',59:'mobile-crane',
    60:'DumpTruck',61:'HaulTruck',62:'Scraper/Tractor',63:'frontloader/bulldozer',64:'excavator',65:'cement-mixer',66:'GroundGrader',
    71:'hut/tent',72:'Shed',73:'Building',74:'AircraftHangar',75:'notDefined',76:'DamagedBuilding',77:'Facility',79:'construction-site',
    82:'notDefined',83:'VehicleLot',84:'helipad',86:'StorageTank',89:'shipping-container-lot',91:'ShippingContainer',93:'Pylon', 94:'Tower'
    }



ALL_CLASSES_SPLIT1 = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter',
               'container-crane','airport', 'helipad','tower-crane','crane-truck','mobile-crane')

BASE_CLASSES_SPLIT1 = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter')

ALL_CLASSES_SPLIT1_DIOR = ('golffield', 'Expressway-toll-station' ,'trainstation' ,'chimney',
                            'storagetank', 'ship', 'harbor' ,'airplane',
                            'tenniscourt', 'dam' ,'basketballcourt', 'Expressway-Service-area' ,
                            'stadium', 'airport', 'baseballfield', 'bridge' ,
                            'windmill', 'overpass','groundtrackfield' , 'vehicle')

ALL_CLASSES_SPLIT1_DIOR_L = ['golffield', 'Expressway-toll-station' ,'trainstation' ,'chimney',
                            'storagetank', 'ship', 'harbor' ,'airplane',
                            'tenniscourt', 'dam' ,'basketballcourt', 'Expressway-Service-area' ,
                            'stadium', 'airport', 'baseballfield', 'bridge' ,
                            'windmill', 'overpass','groundtrackfield' , 'vehicle']

XVIEW_CLASSES_SPLIT1= ('tower-crane','crane-truck','mobile-crane')
DOTA_HBB_TRAIN = '/home/data/dota-1.5/train/labelTxt-v1.5/DOTA-v1.5_train_hbb'
DOTA_HBB_VAL = '/home/data/dota-1.5/val/labelTxt-v1.5/DOTA-v1.5_val_hbb'
DIOR_VAL = '/home/archeval/mmdetection/CATNet/mmdetection/data/dior/ImageSets/Main/val.txt'
DIOR_TRAIN = '/home/archeval/mmdetection/CATNet/mmdetection/data/dior/ImageSets/Main/train.txt'
DIOR_TEST = '/home/archeval/mmdetection/CATNet/mmdetection/data/dior/ImageSets/Main/test.txt'
DIOR_PATH = '/home/archeval/mmdetection/CATNet/mmdetection/data/dior/ImageSets/Main/'
DIOR_XML = '/home/archeval/mmdetection/CATNet/mmdetection/data/dior/Annotations/'
DIOR_FSANN = '/home/archeval/mmdetection/CATNet/mmdetection/data/dior/few_shot_ann/'
def parse_args():
    parser = argparse.ArgumentParser(
        description='handle DOTA dataset')
    parser.add_argument('--dir', default=DIOR_TRAIN, help='directory of the image to analyse')
    parser.add_argument('--img_dir', default=None, help='directory of the image to analyse')
    parser.add_argument('--class_to_find', default=['vehicle'], help='find image with a given class')
    parser.add_argument('--path_all_files', default=None,#'/home/data/dota-1.5/train/labelTxt-v1.5/DOTA-v1.5_train/train.txt',
                         help='Where to store a txt file with all file name')
    parser.add_argument('--create_txt_by_classes', default=False, help='Define if ds is separated in classes ')
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
                for shot in [1,5,10,100]:
                    data = np.array([file_of_current_class['file'].unique()]).T
                    #print(data)
                    files_names = list(map(lambda n: 'images/'+n+'.txt', data))
                    files_names = np.stack(files_names, axis=0 )
                    file_cycle = cycle(files_names)
                    files_names = [next(file_cycle) for i in range(shot)]
                    np.savetxt('/home/data/dota-1.5/few_shot_ann/benchmark_'+str(shot)+'shot/box_'+str(shot)+'shot_'+curent_class_name+'_train.txt',
                               files_names, fmt='%s')


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

            files = df_result['file'].unique()
            for _, file_current in tqdm(enumerate(files)):
            #for class_to_find in list(ALL_CLASSES_SPLIT1_DIOR):
                nbr_classes = []
                #df_of_class = get_images_with_class(class_to_find, df_result)
                #file_of_class = df_of_class['file'].unique()

                for class_to_find in list(ALL_CLASSES_SPLIT1_DIOR):
                #for file_current in file_of_class:
                    a = df_result[(df_result['category'] == class_to_find) & (df_result['file'] == file_current)]
                    #nbr_instance.append((a.shape)[0])
                    if (a.shape)[0] != 0:
                        nbr_classes.append((class_to_find, (a.shape)[0]))

                    #print(f'image {file_current}.xml contains {(a.shape)[0]} time the class {class_to_find}')
                if args.class_to_find is not None and False:
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
    for shot in [1, 5, 10, 50, 100]:
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
    for shot in [10,100]:
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
if __name__ == '__main__':
    #check_image_without_fewshots()
  #  check_stuff_faster()
    #check_val_test_train()

    #check_bbox_nbr()
    b = pickled_dict()
    #queue_base_res, queue_base_trg, queue_novel_res, queue_novel_trg, queue_aug_res, queue_aug_trg = b
    queue_res, queue_trg = b
    print(f'shapes {queue_res.shape} trg {queue_trg.shape}')
    #print(f'shapes {queue_novel_res.shape} trg {queue_novel_trg.shape}')
    #print(f'shapes {queue_aug_res.shape} trg {queue_aug_trg.shape}')
    l = np.arange(queue_res.shape[0])
    np.random.shuffle(l)
    queue_trg_shuffle = queue_trg[l]
    print(queue_trg_shuffle)
    print(queue_trg)
    print(l)