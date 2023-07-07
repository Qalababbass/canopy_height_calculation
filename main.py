'''
This script takes folder path for greenland data and complete path for reference measurement rosbag as inputs.
The script reads all the bags in the given folder path recursively.
It then divide the total timestamp(The total time of rosbag in seconds) of each greenland bag into 20 equal timevalue and then extract the data of lidar for each 20 timevalues
from greenland data bag . From the reference measurement bag it only extracts the data for 1 timevalue. From the lidar data
extracted, the script first convert these points into pointcloud object so that open3d operations can be performed on it. Then the pointcloud in front of machine part
is cropped from pointcloud so that beam_angle of lidar sensor can be put into consideration with these points. As there is no uniform pattern of points
in front of machine so I had to do put beam_angle manipulation manually for each row of points. From these points the pointcloud is cropped into an area of 1*1m^2 area
which is considered as ROI(region of extraction) full . From this ROI this pointcloud area is cropped into subsection of equal sub_pointcloud.


These points are then used to compute the average value of ROI full and sub_pcds for each 20 timestamp. The data points are used in 1 way
1: Data points with beam_angle into consideration

Calculations are performed on these average values for each computed timevalue.
The following are calculated:
Average Value (CSM(ROI full) - DTM(ROI full))
Average Value (CSM(Sub_pcd[i])- DTM(Sub_pcd[i]))
Maximum Value (CSM(Sub_pcd_array) - DTM(Sub_pcd_array))
Minimum Value (CSM(Sub_pcd_array) - DTM(Sub_pcd_array))
Maximum (CSM(Sub_pcd_array)) - Mininmum Value(DTM(Sub_pcd_array))

Currently, there is no pre-processing done on lidar data such as outlier removal, and will be inserted as needed.
'''


import rosbag
import numpy as np
from sensor_msgs import point_cloud2
import glob
import os
import sys
import open3d as o3d
import copy


def data_with_beamangle(pcd_points):
    
    beam_angle=0.519*np.pi/180
    lidar_angle=np.pi/4

    pcd_points[0:96,-1]=pcd_points[0:96,-1]*np.cos(31*beam_angle)
    pcd_points[96:193,-1]=pcd_points[96:193,-1]*np.cos(30*beam_angle)
    pcd_points[193:291,-1]=pcd_points[193:291,-1]*np.cos(29*beam_angle)
    pcd_points[291:391,-1]=pcd_points[291:391,-1]*np.cos(28*beam_angle)
    pcd_points[391:492,-1]=pcd_points[391:492,-1]*np.cos(27*beam_angle)
    pcd_points[492:594,-1]=pcd_points[492:594,-1]*np.cos(26*beam_angle)
    pcd_points[594:696,-1]=pcd_points[594:696,-1]*np.cos(25*beam_angle)
    pcd_points[696:799,-1]=pcd_points[696:799,-1]*np.cos(24*beam_angle)
    pcd_points[799:904,-1]=pcd_points[799:904,-1]*np.cos(23*beam_angle)
    pcd_points[904:1010,-1]=pcd_points[904:1010,-1]*np.cos(22*beam_angle)
    pcd_points[1010:1116,-1]=pcd_points[1010:1116,-1]*np.cos(21*beam_angle)
    pcd_points[1116:1224,-1]=pcd_points[1116:1224,-1]*np.cos(20*beam_angle)
    pcd_points[1224:1333,-1]=pcd_points[1224:1333,-1]*np.cos(19*beam_angle)
    pcd_points[1333:1444,-1]=pcd_points[1333:1444,-1]*np.cos(18*beam_angle)
    pcd_points[1444:1555,-1]=pcd_points[1444:1555,-1]*np.cos(17*beam_angle)
    pcd_points[1555:1668,-1]=pcd_points[1555:1668,-1]*np.cos(16*beam_angle)
    pcd_points[1668:1782,-1]=pcd_points[1668:1782,-1]*np.cos(15*beam_angle)
    pcd_points[1782:1896,-1]=pcd_points[1782:1896,-1]*np.cos(14*beam_angle)
    pcd_points[1896:2012,-1]=pcd_points[1896:2012,-1]*np.cos(13*beam_angle)
    pcd_points[2012:2128,-1]=pcd_points[2012:2128,-1]*np.cos(12*beam_angle)
    pcd_points[2128:2246,-1]=pcd_points[2128:2246,-1]*np.cos(11*beam_angle)
    pcd_points[2246:2365,-1]=pcd_points[2246:2365,-1]*np.cos(10*beam_angle)
    pcd_points[2365:2485,-1]=pcd_points[2365:2485,-1]*np.cos(9*beam_angle)
    pcd_points[2485:2605,-1]=pcd_points[2485:2605,-1]*np.cos(8*beam_angle)
    pcd_points[2605:2727,-1]=pcd_points[2605:2727,-1]*np.cos(7*beam_angle)
    pcd_points[2727:2849,-1]=pcd_points[2727:2849,-1]*np.cos(6*beam_angle)
    pcd_points[2849:2973,-1]=pcd_points[2849:2973,-1]*np.cos(5*beam_angle)
    pcd_points[2973:3099,-1]=pcd_points[2973:3099,-1]*np.cos(4*beam_angle)
    pcd_points[3099:3225,-1]=pcd_points[3099:3225,-1]*np.cos(3*beam_angle)
    pcd_points[3225:3353,-1]=pcd_points[3225:3353,-1]*np.cos(2*beam_angle)
    pcd_points[3353:3482,-1]=pcd_points[3353:3482,-1]*np.cos(beam_angle)
    pcd_points[3482:3611,-1]=pcd_points[3482:3611,-1]
    pcd_points[3611:3741,-1]=pcd_points[3611:3741,-1]*np.cos(-beam_angle)
    pcd_points[3741:3872,-1]=pcd_points[3741:3872,-1]*np.cos(-2*beam_angle)
    pcd_points[3872:4005,-1]=pcd_points[3872:4005,-1]*np.cos(-3*beam_angle)
    pcd_points[4005:4139,-1]=pcd_points[4005:4139,-1]*np.cos(-4*beam_angle)
    pcd_points[4139:4274,-1]=pcd_points[4139:4274,-1]*np.cos(-5*beam_angle)
    pcd_points[4274:4410,-1]=pcd_points[4274:4410,-1]*np.cos(-6*beam_angle)
    pcd_points[4410:4546,-1]=pcd_points[4410:4546,-1]*np.cos(-7*beam_angle)
    pcd_points[4546:4684,-1]=pcd_points[4546:4684,-1]*np.cos(-8*beam_angle)
    pcd_points[4684:4822,-1]=pcd_points[4684:4822,-1]*np.cos(-9*beam_angle)
    pcd_points[4822:4961,-1]=pcd_points[4822:4961,-1]*np.cos(-10*beam_angle)
    pcd_points[4962:5103,-1]=pcd_points[4962:5103,-1]*np.cos(-11*beam_angle)
    pcd_points[5103:5245,-1]=pcd_points[5103:5245,-1]*np.cos(-12*beam_angle)
    pcd_points[5245:5389,-1]=pcd_points[5245:5389,-1]*np.cos(-13*beam_angle)
    pcd_points[5389:5534,-1]=pcd_points[5389:5534,-1]*np.cos(-14*beam_angle)
    
    
    return pcd_points  # return pcd points in front of machine part x with beam angle

def pcd_to_data(pcd_roi_f,sub_pcds):
    
    # inputs are roi full as pcd object and array of sub pcds in roi full with each element of array as pcd object

    pcd_roi_f_xyz=np.asarray(pcd_roi_f.points)
    pcd_roi_f_z=np.abs(pcd_roi_f_xyz[:,-1])
    pcd_roi_f_z_mean=np.average(pcd_roi_f_z)
    sub_pcds_mean_array=np.array([])

    for i in range(0,len(sub_pcds)):
        sub_pcds_points=np.asarray(sub_pcds[i].points)
        sub_pcds_points_z=np.abs(sub_pcds_points[:,2])
        sub_pcds_z_mean=np.average(sub_pcds_points_z)
        sub_pcds_mean_array=np.append(sub_pcds_mean_array,sub_pcds_z_mean)


    return pcd_roi_f_z_mean,sub_pcds_mean_array  # return mean value of pcd_roi_full, array of mean value of sub_pcds



def point_cloud_operations(rosbag_points):

    # input is points obtained from pointcloud2 message in rosbag

    point_cloud=o3d.geometry.PointCloud()
    point_cloud.points=o3d.utility.Vector3dVector(rosbag_points)
    point_cloud_ccw=copy.deepcopy(point_cloud)
    R=point_cloud.get_rotation_matrix_from_xyz((0,np.pi/4,0))
    point_cloud_ccw.rotate(R, center=(0, 0, 0))

    #######################################################################################################
    # The below commented code is to crop the pcd in front of machine and include beam_angle with points

    #bbox_full=o3d.geometry.AxisAlignedBoundingBox(
    #min_bound=[2.3, -2.0, -10],
    #max_bound=[6.59, 2.0, 10]
    #)
    #pcd_crop_front=point_cloud_ccw.crop(bbox_full)

    #pcd_front_xyz=np.asarray(pcd_crop_front.points)
    #pcd_front_with_angle_points=data_with_beamangle(pcd_front_xyz)
    #pcd_front_with_angle=o3d.geometry.PointCloud()
    #pcd_front_with_angle.points=o3d.utility.Vector3dVector(pcd_front_with_angle_points)
    ##############################################################################################

    bbox_roi = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=[2.9, -0.5, -10],
    max_bound=[3.99, 0.5, 10]
    )

    pointcloud_roi_full = point_cloud_ccw.crop(bbox_roi)

    pcd_array=np.array([])


    x_min=2.9
    x_max=3.9
    y_min=-0.5
    y_max=0.5
    z_min=-10
    z_max=10

    for x in np.arange(x_min,x_max,0.2):
        for y in np.arange(y_min,y_max,0.2):    
            bbox=o3d.geometry.AxisAlignedBoundingBox(
                min_bound=[x,y,z_min],
                max_bound=[x+0.2,y+0.2,z_max]
            )
            pcd=pointcloud_roi_full.crop(bbox)
            pcd_array=np.append(pcd_array,pcd)


    return pointcloud_roi_full,pcd_array #return pcd roi full as pcd object, array of sub pcds in roi full with each element of array as pcd object




def calculations(csm_roi,csm_sub_pcds,dtm_roi,dtm_sub_pcds):
    # Inputs are average of csm_roi_full z values and array of average value of z in each sub_pcds, same for dtm
    

    average_roi_f=csm_roi-dtm_roi
    print('Average value of full ROI : {}'.format(average_roi_f))

    for i in range(0,len(csm_sub_pcds)):
        averge_sub_pcds=csm_sub_pcds[i]-dtm_sub_pcds[i]
        print('Average value of sub pcd {} : {} meter'.format(i,round(averge_sub_pcds,3)))

    

    maxim=np.max(csm_sub_pcds)-np.max(dtm_sub_pcds)
    mini=np.min(csm_sub_pcds)-np.min(dtm_sub_pcds)
    max_min=np.max(csm_sub_pcds)-np.min(dtm_sub_pcds)

    print('Maximum Value of csm_sub_pcds - Maximum Value of dtm_sub_pcds : {} meter'.format(round(maxim,3)))
    print('Minimum Value of csm_sub_pcds - Minimum Value of dtm_sub_pcds : {} meter'.format(round(mini,3)))
    print('Maximum value of csm_sub_pcds - Minimum Value of dtm_sub_pcds : {} meter'.format(round(max_min,3)))
    

    return

def greenland_csm_processing(dtm_roi_values,dtm_sub_pcds_values,bag_path):

    bag=rosbag.Bag(bag_path)
    bag_name= os.path.basename(bag_path)
    print("Starting with bag {}".format(bag_name))
    
    start_time = int(bag.get_start_time())
    end_time = int(bag.get_end_time())

    time_array=np.linspace(start_time,end_time,num=20,dtype=int)

    for time in time_array:
        ran=False
        for topic,msg,t in bag.read_messages(topics=['/ouster_cloud_os1_0/points']):
            if int(t.to_sec())==time and not ran:
                print(f"Time {time} : received mesage on topic {topic} at time {int(t.to_sec())}")
                points=np.array(list(point_cloud2.read_points(msg, field_names = ("x", "y", "z"), skip_nans=True)))
                
                csm_pcd_roi_full,csm_sub_pcds=point_cloud_operations(points)
                csm_roi_full_mean,csm_sub_pcds_mean=pcd_to_data(csm_pcd_roi_full,csm_sub_pcds)
                calculations(csm_roi_full_mean,csm_sub_pcds_mean,dtm_roi_values,dtm_sub_pcds_values)

                ran = True
    
    print('Bag Ended {}'.format(bag_name))
    print('#############################################################################################')
    return csm_roi_full_mean,csm_sub_pcds_mean   # return mean of roi_full and array of mean of sub_pcds

def greenland_dtm_processing(bag):
    
    
    start_time = int(bag.get_start_time())
    end_time = int(bag.get_end_time())

    ran=False
    dtm_values_array=np.array([])
    dtm_values_cos_array=np.array([])
    
    for topic,msg,t in bag.read_messages(topics=['/ouster_cloud_os1_0/points']):
        if t.secs==1670316852 and not ran:
            points=np.array(list(point_cloud2.read_points(msg, field_names = ("x", "y", "z"), skip_nans=True)))

            dtm_pcd_roi_full,dtm_sub_pcds=point_cloud_operations(points)
            dtm_roi_full_mean,dtm_sub_pcds_mean=pcd_to_data(dtm_pcd_roi_full,dtm_sub_pcds)

            ran=True
    
    return dtm_roi_full_mean,dtm_sub_pcds_mean



if __name__== '__main__':

    folder_path='/home/qalab/DFKI/greenland_height/krone_data/greenland_data'

    # greenland reference bag to given as input to dtm_bag
    dtm_bag=rosbag.Bag('/home/qalab/DFKI/greenland_height/krone_data/Ground_default_test/recording_2022-12-06-09-53-56_0.bag')
    dtm_roi_full_m,dtm_sub_pcds_m=greenland_dtm_processing(dtm_bag)

    for bag_files in glob.glob(f'{folder_path}/**/*.bag',recursive=True):
        print(bag_files)
        path=bag_files.split('/')
        file_name=path[-1]
        file_type=file_name.split('.')[-1]
        try:
            bag_check=rosbag.Bag(bag_files)
        except rosbag.ROSBagException as e:
            print(e)
            sys.exit(0)
        except Exception as e:
            print(e)
            sys.exit(0)
        csm_roi_full_m,csm_sub_pcds_m=greenland_csm_processing(dtm_roi_full_m,dtm_sub_pcds_m,bag_files)

