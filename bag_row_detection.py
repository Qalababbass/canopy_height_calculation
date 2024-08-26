'''
This script renames a bag file which contains the \'/ublox_gps/fix\' topic by putting the row in the front of the file name
Therefore it compares the gps coordinates of the given bag file with coordinates out of a yaml file.
The yaml file contains the gps coordinates per row and represents a ground truth for each row.

To run the script define the path from where to read the bag file in line#196 and define the path to save these bag files at line#169.
The script read the bag files recursively from the defined path at line#180 which means it will also read bag files from subdirectories of this path.
'''

import math
import os
import sys
import shutil
from glob import glob

import numpy as np
import rosbag
import yaml
from numpy.linalg import norm
from sklearn.linear_model import LinearRegression


def parse_bag_file(bag):
    '''
    Iterate through the given bagfile with the \'/ublox_gps/fix\' topic
    and saves the longitude and latitude if the gps coordinates in a list of lists
    '''

    coordinates = []
    try:
        for topic, msg, t in bag.read_messages(topics=['/ublox_gps/fix']):

            # degree to radian
            lat_rad = msg.latitude  * math.pi / 180
            lon_rad = msg.longitude * math.pi / 180

            coordinates.append([lat_rad, lon_rad])
    except Exception as e:
        print(e)
        sys.exit(0)

    return coordinates

def expand_groundtruth(coordinates, row):
    '''
    Expand the groundtruth:
    If a row already exists in the ground truth: only append coordinates which new
    Else: create a new entry for the new row
    '''

    try:
        # Read the data which is already in the yaml file
        with open('./groundtruth.yaml', 'r') as file:
            data = yaml.full_load(file)
            if data:
                if row in data:
                    for c in coordinates:
                        if not(c in data[row]):
                            data[row].append(c)
                else:
                    data[row] = coordinates
            else:
                data = {row : coordinates}
    except Exception as e:
        print(e)
        sys.exit(0)

    try:
        # write the updated data to the file
        with open('./groundtruth.yaml', 'w') as file:
            yaml.safe_dump(data, file)
    except Exception as e:
        print(e)
        sys.exit(0)

def get_groundtruth():
    '''
    Open the yaml file and return the data
    '''

    try:
        with open('./groundtruth.yaml', 'r') as file:
            data = yaml.full_load(file)

    except Exception as e:
        print(e)
        sys.exit(0)

    if data:
        return data
    else:
        print('There is no ground truth yet to find the fitting row.')
        sys.exit(0)

def list_to_np_array(list_):
    '''
    Converts a list to a np array
    '''

    np_list = []
    for l in list_:
        np_list.append(np.array(l))

    return np_list

def find_row_number(ground_truth, coordinates):
    '''
    Makes a linear regression of each row of the ground truth.
    Then the coordinates of the given bag file are compared to the linear function.
    If 90% of the points are in a distance to the row of 25cm, the wanted row is matched to the ground truth row
    Else: There is no matching row in the ground truth 
    '''

    for row in ground_truth:
        gt_lat = []
        gt_lon = []
        for c in ground_truth[row]:
            gt_lat.append(c[0])
            gt_lon.append(c[1])

        # Linear Regression: https://databraineo.com/statistik/lineare-regression-in-python/
        x = np.array(gt_lat)
        x = x.reshape(-1,1)
        y = np.array(gt_lon)

        model = LinearRegression()
        model.fit(x,y)

        # Two points of the linear regression to make a vector out of those points
        # first point: x = 0 -> y = model.intercept_
        # second point: y = 0 -> x = (-model.intercept_)/model.coef_
        p1 = np.array([0, model.intercept_])
        x2 = np.float64((-1 * model.intercept_)  / model.coef_)
        p2 = np.array([x2, 0])

        # distance: Point to vector
        distances = []
        for c in coordinates:
            dist = norm(np.cross(p2 - p1, p1 - c)) / norm(p2 - p1)

            # radian distance to meters (approximately)
            radian_to_meters = dist * 6371000
            distances.append(radian_to_meters)

        divergent_points = 0
        for d in distances:
            if d > 0.30:
                divergent_points += 1

        # if more than 15% of the coordinates are too far away, check next row, else return the row
        if (divergent_points / len(coordinates)) > 0.15:
            continue
        else:
            return row

    print('Couldn\'t find a matching row')
    #sys.exit(0)

def rename_file(bag_name, bag_path, row):
    '''
    Renames the given file by adding the row number to the name/path
    '''
    
    old_path = '{}{}'.format(bag_path, bag_name)
    #new_path = '{}{}{}{}'.format(bag_path,"haca", row, bag_name)
    new_path = '{}{}{}'.format("robot", row, bag_name)
    print(new_path)
    mount_file='save_dir/' # Path of Directory where rosbags are to be saved
    if os.path.isdir(mount_file)==False:
        print(f'The directory {mount_file} to save the bag files is not found')
        sys.exit(0)
    new_name=mount_file + new_path
    #print(new_name)
    try:
        shutil.copy(old_path,new_name)
        #os.rename(old_path, new_name)

    except Exception as e:
        print(e)
        sys.exit(0)

def remove_bags(path):
    try:
        for files in os.listdir(path):
            dir=os.path.join(path,files)
            if os.path.isdir(dir):
                shutil.rmtree(dir)
            else:
                pass
    except Exception as e:
        print(e)
    print('Bags Removed')

def main():
    read_path='rosbags_dir/' # Path of Directory containing all the rosbags in it.
    if os.path.isdir(read_path)==False:
        print(f'The defined path {read_path} to read the bag files is not found')
        sys.exit(0)
    files=glob(f'{read_path}**/*.bag',recursive=True)
    for file in files:
        path = file.split('/')
        file_name = path[-1]
        #print(file_name)
        file_type = file_name.split('.')[-1]
        if file_type == 'bag':
            bag_name = file_name  
            # puts the path to the bagfile together
            bag_path = ''
            for p in path:
                if p == path[-1]:
                    break
                bag_path += p
                bag_path += '/'
                    
            try:
                bag = rosbag.Bag(file)
            except rosbag.ROSBagException as e:
                print(e)
                sys.exit(0)
            except Exception as e:
                print(e)
                sys.exit(0)
        #else:
        #    help_msg()
        #    sys.exit(0)
        coordinates = parse_bag_file(bag)
        # no row given -> find a matching row
        ground_truth = get_groundtruth()
        np_coord = list_to_np_array(coordinates)
        found_row = find_row_number(ground_truth, np_coord)
        rename_file(bag_name, bag_path, found_row)
    #remove_bags(read_path)

if __name__ == '__main__':
    main()




