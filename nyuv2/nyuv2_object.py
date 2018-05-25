''' Helper class and functions for loading KITTI objects

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
import nyuv2_util as utils

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3


class nyuv2_object(object):
    '''Load and parse object data into a usable format.'''
    
    def __init__(self, root_dir, split='training'):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':
            self.num_samples = 537 #TODO right number
        elif split == 'testing':
            self.num_samples = 654 #TODO right number
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'color')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.lidar_dir = os.path.join(self.split_dir, 'dmap_f')
        self.label_dir = os.path.join(self.split_dir, 'gt_label_19')
        self.box2d_dir = os.path.join(self.split_dir, 'gt_2Dsel_19')
        self.box3d_dir = os.path.join(self.split_dir, 'gt_3D_19')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        #assert(idx<self.num_samples) #number are not ordered, so udx can be greater than num samples
        img_filename = os.path.join(self.image_dir, '%d.jpg'%(idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx): 
        #assert(idx<self.num_samples) 
        lidar_filename = os.path.join(self.lidar_dir, '%d.mat'%(idx))
        return utils.load_velo_scan(lidar_filename)
        
    def get_depth(self, idx): 
        #assert(idx<self.num_samples) 
        lidar_filename = os.path.join(self.lidar_dir, '%d.mat'%(idx))
        return utils.load_depth(lidar_filename)

    def get_calibration(self, idx):
        #assert(idx<self.num_samples) 
        calib_filename = os.path.join(self.calib_dir, '%d.mat'%(idx))
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        #print(self.num_samples,self.split)
        #assert(idx<self.num_samples and self.split=='training') 
        assert(self.split=='training') 
        label_filename = os.path.join(self.label_dir, '%d.mat'%(idx))
        box2d_filename = os.path.join(self.box2d_dir, '%d.mat'%(idx))
        box3d_filename = os.path.join(self.box3d_dir, '%d.mat'%(idx))
        return utils.read_label(label_filename,box2d_filename, box3d_filename)
        
    def get_label_gt_objects(self, idx):
        
        label_filename = os.path.join("train/detection_results_v1/data/", '%d.txt'%(idx)) #TODO variable for path
       
        return utils.read_gt_label(label_filename)
        
    def get_depth_map(self, idx):
        pass

    def get_top_down(self, idx):
        pass
'''
class kitti_object_video(object):
    ###Load data for KITTI videos
    def __init__(self, img_dir, lidar_dir, calib_dir):
        self.calib = utils.Calibration(calib_dir, from_video=True)
        self.img_dir = img_dir
        self.lidar_dir = lidar_dir
        self.img_filenames = sorted([os.path.join(img_dir, filename) \
            for filename in os.listdir(img_dir)])
        self.lidar_filenames = sorted([os.path.join(lidar_dir, filename) \
            for filename in os.listdir(lidar_dir)])
        print(len(self.img_filenames))
        print(len(self.lidar_filenames))
        #assert(len(self.img_filenames) == len(self.lidar_filenames))
        self.num_samples = len(self.img_filenames)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        #assert(idx<self.num_samples) 
        img_filename = self.img_filenames[idx]
        return utils.load_image(img_filename)

    def get_lidar(self, idx): 
        #assert(idx<self.num_samples) 
        lidar_filename = self.lidar_filenames[idx]
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, unused):
        return self.calib
'''
def viz_kitti_video():
    video_path = os.path.join(ROOT_DIR, 'dataset/2011_09_26/')
    dataset = kitti_object_video(\
        os.path.join(video_path, '2011_09_26_drive_0023_sync/image_02/data'),
        os.path.join(video_path, '2011_09_26_drive_0023_sync/velodyne_points/data'),
        video_path)
    print(len(dataset))
    for i in range(len(dataset)):
        img = dataset.get_image(0)
        pc = dataset.get_lidar(0)
        Image.fromarray(img).show()
        draw_lidar(pc)
        raw_input()
        pc[:,0:3] = dataset.get_calibration().project_velo_to_rect(pc[:,0:3])
        draw_lidar(pc)
        raw_input()
    return

def depth_image_to_coords(dmap_f):
    w,h = dmap_f.shape

    xt = np.linspace(1, h,h )
    yt = np.linspace(1, w,w )
    x, y = np.meshgrid(xt, yt)

    x3 = x.reshape(-1);
    y3 = y.reshape(-1);
    z3 = dmap_f.reshape(-1);
    return np.stack([x3,y3,z3]).T

def show_image_with_boxes(img, objects, calib, show3d=True, color=(0,255,0), gt=True):
    ''' Show image with 2D bounding boxes '''
    img1 = np.copy(img) # for 2d bbox
    img2 = np.copy(img) # for 3d bbox
    label="GT"
    #obj = objects[0] #TODO togliere
    #objects = [obj] #TODO togliere
    for obj in objects:
        
        if(len(obj.type)>2): #results label are literals
            obj_type = labelName2classNum(obj.type)
        else:
            obj_type = int(obj.type) 

        label_color = class2color(obj_type) 
        cv2.rectangle(img1, (int(obj.xmin),int(obj.ymin)),
            (int(obj.xmax),int(obj.ymax)), label_color, 2)
        #s = '%s/%.3f' % (classes_names[i], scores[i])
        score = 1
        if hasattr(obj, 'score'):
            score = obj.score
        class_name = classNum2labelName(obj_type)
        s = '%s/%.3f' % (class_name, score)
        p1 = (int(obj.ymin)-5, int(obj.xmin))
        cv2.putText(img1, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.6, label_color, 1)
        if not gt:
            label = "Results"
        cv2.putText(img1, label, (10,30), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 1)
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P, calib.Rtilt) 
        img2 = utils.draw_projected_box3d(img2, box3d_pts_2d,label_color,cs_score=s, label=label) 
    Image.fromarray(img1).show()
    if show3d: 
        Image.fromarray(img2).show() 

def save_gt_and_results(img, objects1, objects2, calib, name="test",save=False ):
    img1 = np.copy(img) # for 2d bbox
    img2 = np.copy(img) # for 3d bbox
    img3 = np.copy(img) # for 2d bbox
    img4 = np.copy(img) # for 3d bbox
    
    label="GT"
    #GT
    for obj in objects1:
        
        if(len(obj.type)>2): #results label are literals
            obj_type = labelName2classNum(obj.type)
        else:
            obj_type = int(obj.type) 

        label_color = class2color(obj_type) 
        cv2.rectangle(img1, (int(obj.xmin),int(obj.ymin)),
            (int(obj.xmax),int(obj.ymax)), label_color, 2)

        score = 1
        if hasattr(obj, 'score'):
            score = obj.score
        class_name = classNum2labelName(obj_type)
        s = '%s/%.3f' % (class_name, score)
        p1 = (int(obj.ymin)-5, int(obj.xmin))
        cv2.putText(img1, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.6, label_color, 1)
        
        label = "GT"
        cv2.putText(img1, label, (10,30), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 1)
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P, calib.Rtilt) 
        if(box3d_pts_2d is None): #Why?
            continue
        img2 = utils.draw_projected_box3d(img2, box3d_pts_2d,label_color,cs_score=s, label=label) 
    
    for obj in objects2:
        
        if(len(obj.type)>2): #results label are literals
            obj_type = labelName2classNum(obj.type)
        else:
            obj_type = int(obj.type) 

        label_color = class2color(obj_type) 
        cv2.rectangle(img3, (int(obj.xmin),int(obj.ymin)),
            (int(obj.xmax),int(obj.ymax)), label_color, 2)

        score = 1
        if hasattr(obj, 'score'):
            score = obj.score
        class_name = classNum2labelName(obj_type)
        s = '%s/%.3f' % (class_name, score)
        p1 = (int(obj.ymin)-5, int(obj.xmin))
        cv2.putText(img3, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.6, label_color, 1)
        
        label = "Results"
        cv2.putText(img3, label, (10,30), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 1)
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P, calib.Rtilt) 
        if(box3d_pts_2d is None): #Why?
            continue
        img4 = utils.draw_projected_box3d(img4, box3d_pts_2d,label_color,cs_score=s, label=label) 
    
    
    
    
    
    
    width, height = (561,427)
    
    total_width = 2*width
    max_height = 2*height
    
    
    new_im = Image.new('RGB', (total_width, max_height))
   

    new_im.paste(Image.fromarray(img1), (0,0))
    new_im.paste(Image.fromarray(img2), (561,0))
    new_im.paste(Image.fromarray(img3), (0,427))
    new_im.paste(Image.fromarray(img4), (561,427))

    if save==True:
        new_im.save('results/'+name+'.jpg')
    else:
        new_im.show()
    
    
    
def class2color(i):
    colors = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    return colors[i]

def classNum2labelName(i):
    classes = ['background', 'bathtub', 'bed', 'bookshelf', 'box', 
               'chair', 'counter', 'desk', 'door', 'dresser', 
               'garbage_bin', 'lamp', 'monitor', 'night_stand', 
               'pillow', 'sink', 'sofa', 'table', 'television', 'toilet'];
    return classes[i]

def labelName2classNum(className):
    classes = ['background', 'bathtub', 'bed', 'bookshelf', 'box', 
               'chair', 'counter', 'desk', 'door', 'dresser', 
               'garbage_bin', 'lamp', 'monitor', 'night_stand', 
               'pillow', 'sink', 'sofa', 'table', 'television', 'toilet'];
    return classes.index(className)
    
    

def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
        (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)
    fov_inds = fov_inds & (pc_velo[:,0]>clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds,:]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def show_lidar_with_boxes(pc_velo, objects, calib,
                          img_fov=False, img_width=None, img_height=None): 
        #Show all LiDAR points.
        #Draw 3d box in LiDAR point cloud (in velo coord system)
    if 'mlab' not in sys.modules: import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    print(('All point num: ', pc_velo.shape[0]))
    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    if img_fov:
        pc_velo = get_lidar_in_image_fov(pc_velo, calib, 0, 0,
            img_width, img_height)
        print(('FOV point num: ', pc_velo.shape[0]))
    draw_lidar(pc_velo, fig=fig)
    
    for obj in objects:
        if obj.type=='DontCare':continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P,calib.Rtilt) 
        #box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        box3d_pts_3d_velo = box3d_pts_3d
        # Draw heading arrow
        ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
        #ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        ori3d_pts_3d_velo = ori3d_pts_3d
        x1,y1,z1 = ori3d_pts_3d_velo[0,:]
        x2,y2,z2 = ori3d_pts_3d_velo[1,:]
        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
        #mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5),
        #    tube_radius=None, line_width=1, figure=fig)
    
    mlab.show(1)

def show_lidar_on_image(pc_velo, img, calib, img_width, img_height):
    ''' Project LiDAR points to image '''
    #imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
    #    calib, 0, 0, img_width, img_height, True)
    #imgfov_pts_2d = pts_2d[fov_inds,:]
    #imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)
    imgfov_pts_2d = pc_velo
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:,:3]*255
    #print("imgfov_pts_2d.shape[0]",imgfov_pts_2d.shape)
    maxD = np.max(imgfov_pts_2d)
    minD = np.min(imgfov_pts_2d)
    for i in range(imgfov_pts_2d.shape[0]):
        for j in range(imgfov_pts_2d.shape[1]):
            depth = pc_velo[i,j]
            
            color = cmap[int(255*(depth-minD)/(maxD-minD)),:]
            #cv2.circle(img, (50,50),2,color=tuple(color), thickness=-1)
           
            cv2.circle(img, (j,i),
                2, color=tuple(color), thickness=-1)
    Image.fromarray(img).show() 
    return img

def dataset_viz():
    dataset = nyuv2_object(os.path.join(ROOT_DIR, 'dataset/NYUV2/object'))

    for data_idx in range(len(dataset)):
        # Load data from dataset
        objects = dataset.get_label_objects(data_idx)
        objects[0].print_object()
        img = dataset.get_image(data_idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_height, img_width, img_channel = img.shape
        print(('Image shape: ', img.shape))
        pc_velo = dataset.get_lidar(data_idx)[:,0:3]
        calib = dataset.get_calibration(data_idx)

        # Draw 2d and 3d boxes on image
        show_image_with_boxes(img, objects, calib, False)
        raw_input()
        # Show all LiDAR points. Draw 3d box in LiDAR point cloud
        show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height)
        raw_input()

if __name__=='__main__':
    import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d
    dataset_viz()
