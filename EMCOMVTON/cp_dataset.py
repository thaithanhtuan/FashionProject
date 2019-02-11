#coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

import os.path as osp
import numpy as np
import json
import cv2

#python trainnew.py --name gmm_train_new/top --clothtype=top --stage GMM
# --workers 4 --save_count 5000 --shuffle

class CPDataset(data.Dataset):
    """Dataset for CP-VTON.
    """
    def __init__(self, opt):
        super(CPDataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode # train or test or self-defined
        self.stage = opt.stage # GMM or TOM
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.clothtype = opt.clothtype
        self.withbg = opt.withbg
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([  \
                #transforms.RandomAffine(degrees=5,translate=(0.1,0.1)),  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        # load data list
        im_names = []
        c_names = []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names

    def name(self):
        return "CPDataset"

    def __getitem__(self, index):
        #withbg = False
        c_name = self.c_names[index]
        im_name = self.im_names[index]

        # cloth image & cloth mask
        if self.stage == 'GMM':
            c = Image.open(osp.join(self.data_path, 'cloth', self.clothtype , c_name))

            cm = Image.open(osp.join(self.data_path, 'cloth-mask', self.clothtype , c_name.split(".")[0]+".png"))

            #after getting c and mask, apply random affine tranform - Tuan
            import math
            angle = (np.random.random_integers(-30,30))/180.0*math.pi
            translation = (np.random.random_integers(-10,10), np.random.random_integers(-10,10))
            scale = (np.random.uniform(1/1.2,1.2),np.random.uniform(1/1.2,1.2))
            a = math.cos(angle)/scale[0]
            b = math.sin(angle)/scale[0]
            nx,ny = x,y = (c.size[0]/2, c.size[1]/2)
            nx = nx + translation[0]
            ny = ny + translation[1]
            ct = nx - x*a - y *b
            d = -math.sin(angle)/scale[1]
            e = math.cos(angle)/scale[1]
            f = ny - x *d -y *e
            c2 = c.convert('RGBA')
            rot = c2.transform(c.size, Image.AFFINE, (a,b,ct,d,e,f), resample = Image.BILINEAR)
            fff = Image.new('RGBA', rot.size, (255,)*4)
            c2 = Image.composite(rot, fff, rot)
            c = c2.convert(c.mode)

            cm = cm.transform(cm.size, Image.AFFINE, (a,b,ct,d,e,f), resample = Image.NEAREST)

            #Done transform
            cbottom = ''
            cmbottom = ''

        else:
            c = Image.open(osp.join(self.data_path, 'warp-cloth', "top" , c_name))
            cm = Image.open(osp.join(self.data_path, 'warp-mask', "top" , c_name.split(".")[0]+".png"))
            cbottom = Image.open(osp.join(self.data_path, 'warp-cloth', "bottom" , c_name))
            cmbottom = Image.open(osp.join(self.data_path, 'warp-mask', 'bottom' , c_name.split(".")[0]+".png"))

            cbottom = cbottom.resize((self.fine_width, self.fine_height), Image.BILINEAR)  # Tuan
            cmbottom = cmbottom.resize((self.fine_width, self.fine_height), Image.BILINEAR)  # Tuan
            cbottom = self.transform(cbottom)  # [-1,1]
            cm_array = np.array(cmbottom)
            cm_array = (cm_array >= 128).astype(np.float32)
            cmbottom = torch.from_numpy(cm_array) # [0,1]
            cmbottom.unsqueeze_(0)

        c = c.resize((self.fine_width, self.fine_height), Image.BILINEAR)  # Tuan
        cm = cm.resize((self.fine_width, self.fine_height), Image.BILINEAR)  # Tuan
        c = self.transform(c)  # [-1,1]
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array) # [0,1]
        cm.unsqueeze_(0)

        # person image
        if(self.withbg == True):
            im = Image.open(osp.join(self.data_path, 'imagewithbg', im_name))
        else:
            im = Image.open(osp.join(self.data_path, 'imagenobg', im_name))
        im = im.resize((self.fine_width, self.fine_height), Image.BILINEAR)#Tuan

        im = self.transform(im) # [-1,1]

        # load parsing image
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(osp.join(self.data_path, 'image-parse', parse_name))
        im_parse = im_parse.resize((self.fine_width, self.fine_height), Image.BILINEAR)#Tuan
        """
        im_parse = np.asarray(im_parse)
        plt.imshow(im_parse * 16, cmap='gray')
        plt.show()
        """

        parse_array = np.array(im_parse)

        #print(np.unique(parse_array))

        parse_shape = (parse_array > 0).astype(np.float32)
        parse_head = (parse_array == 1).astype(np.float32) + \
                (parse_array == 2).astype(np.float32) + \
                (parse_array == 3).astype(np.float32) + \
                (parse_array == 11).astype(np.float32)
        if(self.clothtype == "top"):
            parse_cloth = (parse_array == 4).astype(np.float32) + \
                          (parse_array == 7).astype(np.float32)
        else:
            parse_cloth = (parse_array == 5).astype(np.float32) + \
                          (parse_array == 6).astype(np.float32)
       
        # shape downsample
        parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8))
        parse_shape = parse_shape.resize((self.fine_width//16, self.fine_height//16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        shape = self.transform(parse_shape) # [-1,1]
        phead = torch.from_numpy(parse_head) # [0,1]
        pcm = torch.from_numpy(parse_cloth) # [0,1]

        # upper cloth
        #print("im:", im.shape, ", pcm:", pcm.shape)
        im_c = im * pcm + (1 - pcm) # [-1,1], fill 1 for other parts
        im_h = im * phead - (1 - phead) # [-1,1], fill 0 for other parts

        # load pose points
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, 'pose', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1,3))
            #print(":::", pose_data)
            #pose_data = pose_data*[192.0/400.0,256.0/600.0,1]
            #print(":", pose_data)

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i,0]
            pointy = pose_data[i,1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = self.transform(one_map)
            pose_map[i] = one_map[0]

        # just for visualization
        im_pose = self.transform(im_pose)
        #print("shape:", shape.shape, ", im_h:", im_h.shape, ", pose_map:", pose_map.shape)

        # cloth-agnostic representation

        agnostic = torch.cat([shape, im_h, pose_map], 0) 

        #print("jjjjjjjjjjjjjjjjjjjjjjjjjjj")

        if self.stage == 'GMM':
            im_g = Image.open('grid.png')
            im_g = self.transform(im_g)
        else:
            im_g = ''

        result = {
            'c_name':   c_name,     # for visualization
            'im_name':  im_name,    # for visualization or ground truth
            'cloth':    c,          # for input
            'cloth_mask':     cm,   # for input
            'cloth_bottom':    cbottom,          # for input
            'cloth_mask_bottom':     cmbottom,   # for input
            'image':    im,         # for visualization
            'agnostic': agnostic,   # for input
            'parse_cloth': im_c,    # for ground truth
            'shape': shape,         # for visualization
            'head': im_h,           # for visualization
            'pose_image': im_pose,  # for visualization
            'grid_image': im_g,     # for visualization
            }

        return result

    def __len__(self):
        return len(self.im_names)

class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()

        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=1)
    
    opt = parser.parse_args()
    dataset = CPDataset(opt)
    data_loader = CPDataLoader(opt, dataset)

    print('Size of the dataset: %05d, dataloader: %04d' \
            % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()

    from IPython import embed; embed()

