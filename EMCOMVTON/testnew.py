#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import GMM, UnetGenerator, load_checkpoint

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images, save_images
#TEst pairs: model image <--> cloth image


#python testnew.py --name gmm_train_test_new/top --clothtype=top --stage GMM --workers 4 --datamode test --data_list train_pairs.txt --checkpoint checkpoints/gmm_train_new/top/step_040000.pth


#python testnew.py --name gmm_train_test_new/bottom --clothtype=bottom --stage GMM --workers 4
# --datamode test --data_list train_pairs.txt --checkpoint checkpoints/gmm_train_new/bottom/step_040000.pth

#Test TOM nobg
#python testnew.py --name TOM_Test_Nobg --stage TOM --workers 4 --datamode test --data_list test_pairs.txt --checkpoint checkpoints/tom_train_new/step_180000.pth


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--clothtype", default = "top")
    parser.add_argument("--withbg", default = False)
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for test')
    parser.add_argument("--display_count", type=int, default = 1)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

def test_gmm(opt, test_loader, model, board):
    model.cuda()
    model.eval()

    base_name = os.path.basename(opt.checkpoint)
    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth', opt.clothtype)
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)
    warp_mask_dir = os.path.join(save_dir, 'warp-mask', opt.clothtype)
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)

    affine_cloth_dir = os.path.join(save_dir, 'affine-warp-cloth', opt.clothtype)
    if not os.path.exists(affine_cloth_dir):
        os.makedirs(affine_cloth_dir)
    affine_mask_dir = os.path.join(save_dir, 'affine-warp-mask', opt.clothtype)
    if not os.path.exists(affine_mask_dir):
        os.makedirs(affine_mask_dir)

    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        
        c_names = inputs['c_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c =  inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()
            
        grid, theta = model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

        visuals = [ [im_h, shape, im_pose], 
                   [c, warped_cloth, im_c], 
                   [warped_grid, (warped_cloth*0.9+im*0.1), im]]
        
        save_images(warped_cloth, c_names, warp_cloth_dir,isCloth = True)
        save_images(warped_mask*2-1, c_names, warp_mask_dir, isCloth = False)
        save_images(c, c_names, affine_cloth_dir, isCloth = True)
        save_images(cm*2-1, c_names, affine_mask_dir, isCloth = False)

        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)
        


def test_tom(opt, test_loader, model, board):
    model.cuda()
    model.eval()
    
    base_name = os.path.basename(opt.checkpoint)
    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try_on_dir = os.path.join(save_dir, 'try-on')
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)
    render_dir = os.path.join(save_dir, 'render_dir')
    if not os.path.exists(render_dir):
        os.makedirs(render_dir)
    alpha_top_dir = os.path.join(save_dir, 'alpha_top_dir')
    if not os.path.exists(alpha_top_dir):
        os.makedirs(alpha_top_dir)
    alpha_bottom_dir = os.path.join(save_dir, 'alpha_bottom_dir')
    if not os.path.exists(alpha_bottom_dir):
        os.makedirs(alpha_bottom_dir)
    print('Dataset size: %05d!' % (len(test_loader.dataset)), flush=True)
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        
        im_names = inputs['im_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        cbottom = inputs['cloth_bottom'].cuda()
        cmbottom = inputs['cloth_mask_bottom'].cuda()

        outputs = model(torch.cat([agnostic, c, cbottom], 1))
        p_rendered, m_composite = torch.split(outputs, 3,1)
        m_composite, m_composite_bottom = torch.split(m_composite, 1, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        m_composite_bottom = F.sigmoid(m_composite_bottom)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite - m_composite_bottom) + cbottom * m_composite_bottom
        #print("m_composite:", m_composite.shape, ", max: ", m_composite.max, ", min: ", m_composite.min)
        visuals = [[im_h, shape, im_pose],
                   [c, cm * 2 - 1, m_composite * 2 - 1],
                   [cbottom, cmbottom * 2 - 1, m_composite_bottom * 2 - 1],
                   [p_rendered, p_tryon, im]]

        save_images(p_tryon, im_names , try_on_dir, isCloth=True)
        save_images(p_rendered, im_names , render_dir, isCloth=True)
        save_images(m_composite * 2 - 1, im_names , alpha_top_dir, isCloth=True)
        save_images(m_composite_bottom * 2 - 1, im_names , alpha_bottom_dir, isCloth=True)
        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t), flush=True)


def main():
    opt = get_opt()
    print(opt)
    print("Start to test stage: %s, named: %s!" % (opt.stage, opt.name))
   
    # create dataset 
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))
   
    # create model & train
    if opt.stage == 'GMM':
        model = GMM(opt)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_gmm(opt, train_loader, model, board)
    elif opt.stage == 'TOM':
        model = UnetGenerator(28, 5, 6, ngf=64, norm_layer=nn.InstanceNorm2d)#Tuan change input dimention from 25,4 to 28,5
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_tom(opt, train_loader, model, board)
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)
  
    print('Finished test %s, named: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
