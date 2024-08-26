import os
import sys

#import rospy
#import geometry_msgs.msg

import rosbag

def help_msg():
    script_name = os.path.basename(__file__)
    print(
        'This script renames a bag file which contains the \'/rocas_metadata/rope_position\' topic by putting the last position in the front of the file name \n\n' +
        'Usage: \n' +
        '  python3 %s --bag <example.bag> \n' % script_name +
        '    -> to rename one specific bagfile \n' + 
        'or \n' +
        '  python3 %s --path <path> \n\n' % script_name +
        '<example.bag> : A bag file which contains the \'/rocas_metadata/rope_position\' topic \n' +
        '<path>        : Path to a directory with bagfiles'
    )

def parse_input():
    '''
    Parse the input
    by saving the bag file as a rospy bag file, its name and its path to rename it
    '''

    if len(sys.argv) != 3:
        help_msg()
        sys.exit(0)

    bag_path  = ''
    bag_names = []
    bag_files = []
    i = 0
    for arg in sys.argv:
        # parse the bag file
        if arg == '--bag':
            file = sys.argv[i + 1]
            path = file.split('/')
            file_name = path[-1]
            file_type = file_name.split('.')[-1]

            if file_type == 'bag':
                bag_names.append(file_name)
                
                # puts the path to the bagfile together
                bag_path = ''
                for p in path:
                    if p == path[-1]:
                        break
                    bag_path += p
                    bag_path += '/'

                try:
                    bag = rosbag.Bag(file)
                    bag_files.append(bag)
                except rosbag.ROSBagException as e:
                    print(e)
                    sys.exit(0)
                except Exception as e:
                    print(e)
                    sys.exit(0)
                i += 1
            
        elif arg == '--path':
            bag_path = sys.argv[i + 1]
            try:

                for root,d_name,f_name in os.walk(bag_path):
                    directory=os.path.split(root)
                    print('Starting in Directory {}'.format(root))
                    for files in f_name:
                        
                        if files.endswith('.bag'):
                            bag_names.append(os.path.join(directory[-1],files))
                            bag_file=os.path.join(root,files)
                            bag=rosbag.Bag(bag_file)
                            bag_files.append(bag)


#################################################### old code #######################################################
                # directory = os.fsencode(bag_path)

                # for file in os.listdir(directory):
                #     filename = os.fsdecode(file)
                #     if filename.endswith('.bag'):
                #         bag_names.append(filename)

                #         # create rosbagfiles
                #         bag_file = bag_path + filename
                #         bag = rosbag.Bag(bag_file)
                #         bag_files.append(bag)
###############################################################################################

                i += 1

            except:
                help_msg()
                sys.exit(0)

        else:
            i += 1

    return bag_names, bag_path, bag_files

def parse_bag_file(bag):
    '''
    Iterate through the given bagfile with the '/rosbag_position' topic
    and saves the position where the bagfile was created
    '''
    source_frame='odom'
    target_frame='base_link'
    position=''
    try:
        # the position does not change in the bag file -> break after finding the position
        for topic, msg, t in bag.read_messages(topics=['/tf']):
            #print(msg.transforms.transform.translation)
            for transform in msg.transforms:
                if transform.header.frame_id==source_frame and transform.child_frame_id==target_frame:
                    position = int(transform.transform.translation.x * 1000)
                    if position < 0:
                        position =str(abs(position))
                else:
                    #print('No transform available')
                    position=''

                #transform_stamped = geometry_msgs.msg.TransformStamped(
                #            header=transform.header,
                #            child_frame_id=transform.child_frame_id,
                #            transform=transform.transform
                #        )


    except Exception as e:
        print(e)
        sys.exit(0)

    return position

def rename_file(bag_name, bag_path, position):
    '''
    Renames the given file by adding the position to the name/path
    update
    '''

    name_parts = bag_name.split('_')
    bag_name_only=os.path.split(bag_name)
    first_letter=name_parts[0].split('/')
    if first_letter[-1] == 'rocas':
        print('File:', bag_name, 'is already renamed')
        return

    old_path = '{}{}'.format(bag_path, bag_name)
    if len(position)==0:
        position='#####'
    new_path = '{}{}{}{}_{}'.format(bag_path,bag_name_only[0] ,'/robot_', position, bag_name_only[-1])
    try:
        os.rename(old_path, new_path)
    except Exception as e:
        print(e)
        sys.exit(0)

def main():
    bag_names, bag_path, bag_files = parse_input()

    i = 0
    for bag in bag_files:
        position = parse_bag_file(bag)
        rename_file(bag_names[i], bag_path, position)
        i += 1

if __name__ == '__main__':
    main()
    print('Script completed successfully')
