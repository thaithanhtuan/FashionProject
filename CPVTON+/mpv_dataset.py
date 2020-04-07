# coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os.path as osp
import numpy as np
import json
from scipy.sparse import load_npz
import os

class CPDataset(data.Dataset):
    """Dataset for CP-VTON+.
    """

    def __init__(self, opt):
        super(CPDataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode  # train or test or self-defined
        self.stage = opt.stage  # GMM or TOM
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.data_path = osp.join(opt.dataroot, opt.datamode)


        # load data list
        im_namess = []
        im_namest = []
        c_names = []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name1, im_name2, c_name, is_train = line.strip().split()
                # first pair
                if (is_train == "train"):
                    im_namess.append(im_name1)
                    im_namest.append(im_name2)
                    c_names.append(c_name)

        self.im_namess = im_namess
        self.im_namest = im_namest
        self.c_names = c_names
        self.transform = transforms.Compose([ \
            transforms.ToTensor(), \
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.transformmask = transforms.Compose([ \
            transforms.ToTensor(), \
            transforms.Normalize((0.5,), (0.5,))])
        # get same clothes in diff test pairs
        """new_c_names = []
        for each in im_names:
            new_c_names.append(each.split("_")[0] + "_1.jpg")
        self.c_names = new_c_names"""

    def name(self):
        return "CPDataset"



    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_names = self.im_namess[index]
        im_namet = self.im_namest[index]



        name = os.path.splitext(os.path.basename(c_name))[0] + "_to_" + os.path.splitext(os.path.basename(im_names))[0] + "_to_" + os.path.splitext(os.path.basename(im_namet))[0]
        data_sparse = load_npz("./data/results/warp/"+name+".npz")
        parse_array = data_sparse.tocoo()
        parse_array.data = parse_array.data
        parse_array = parse_array.todense()

        # person image
        im = Image.open("./data/results/warp/images/"+name+"_fakes_decoded.png")
        im = self.transform(im)  # [-1,1]

        parse_shape = (parse_array == 0).astype(np.float32) + \
                      (parse_array == 3).astype(np.float32) + \
                      (parse_array == 5).astype(np.float32) + \
                      (parse_array == 6).astype(np.float32) + \
                      (parse_array == 7).astype(np.float32) + \
                      (parse_array == 8).astype(np.float32) + \
                      (parse_array == 9).astype(np.float32) + \
                      (parse_array == 10).astype(np.float32) + \
                      (parse_array == 11).astype(np.float32) + \
                      (parse_array == 12).astype(np.float32) + \
                      (parse_array == 14).astype(np.float32) + \
                      (parse_array == 15).astype(np.float32) + \
                      (parse_array == 16).astype(np.float32) + \
                      (parse_array == 17).astype(np.float32) + \
                      (parse_array == 18).astype(np.float32)

        parse_cloth = (parse_array == 5).astype(np.float32) + \
                      (parse_array == 6).astype(np.float32) + \
                      (parse_array == 7).astype(np.float32)

        parse_shape = Image.fromarray((parse_shape * 255).astype(np.uint8))
        parse_shape = parse_shape.resize((self.fine_width // 16, self.fine_height // 16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)
        shape = self.transformmask(parse_shape)  # [-1,1]
        parse_cloth = Image.fromarray((parse_cloth * 255).astype(np.uint8))
        cloth_mask_gt = self.transformmask(parse_cloth)  # [-1,1]


        c = Image.open("./data/MPV/all/" + c_name)
        c = self.transform(c)  # [-1,1]
        cm = Image.open("./data/MPV/all/" + c_name.split(".jpg")[0]+"_mask.jpg")
        cm = self.transformmask(cm)  # [-1,1]

        im_g = Image.open('grid.png')
        im_g = self.transform(im_g)
        return {
            "cloth_mask": cm,  # for input
            "cloth_mask_gt": cloth_mask_gt,
            "shape": shape,
            "c_name": c_name,
            "im_names": im_names,
            "im_namet": im_namet,
            'grid_image': im_g,  # for visualization
            'image': im,  # for visualization
            'c': c,  # for visualization
        }

    def __len__(self):
        return len(self.im_namess)

    def remove_extension(self, fname):
        return os.path.splitext(fname)[0]

class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()


        if opt.shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=(
                train_sampler is None),
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
    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--shuffle", action='store_true',
                        help='shuffle input data')
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=1)

    opt = parser.parse_args()
    dataset = CPDataset(opt)
    data_loader = CPDataLoader(opt, dataset)

    print('Size of the dataset: %05d, dataloader: %04d'
          % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()

    from IPython import embed
    embed()
