# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import LAGMM, UnetGenerator, load_checkpoint

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images, save_images


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GMM")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)

    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for test')
    parser.add_argument("--display_count", type=int, default=1)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt


def test_gmm(opt, test_loader, model, board):
    model.cuda()
    model.eval()

    base_name = os.path.basename(opt.checkpoint)
    name = opt.name
    save_dir = os.path.join(opt.result_dir, name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)
    warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)
    warp_grid_dir = os.path.join(save_dir, 'warp-grid')
    if not os.path.exists(warp_grid_dir):
        os.makedirs(warp_grid_dir)
    Ipersp_dir = os.path.join(save_dir, 'warp-cloth_affine')
    if not os.path.exists(Ipersp_dir):
        os.makedirs(Ipersp_dir)
    warped_mask_affine_dir = os.path.join(save_dir, 'warped_mask_affine')
    if not os.path.exists(warped_mask_affine_dir):
        os.makedirs(warped_mask_affine_dir)
    warped_grid_affine_dir = os.path.join(save_dir, 'warped_grid_affine')
    if not os.path.exists(warped_grid_affine_dir):
        os.makedirs(warped_grid_affine_dir)
    overlayed_affine_dir = os.path.join(save_dir, 'overlayed_affine')
    if not os.path.exists(overlayed_affine_dir):
        os.makedirs(overlayed_affine_dir)
    overlayed_TPS_dir = os.path.join(save_dir, 'overlayed_TPS')
    if not os.path.exists(overlayed_TPS_dir):
        os.makedirs(overlayed_TPS_dir)

    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        im_names = inputs['im_name']
        c_names = inputs['c_name']
        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c = inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()

        gridtps, thetatps, Ipersp, gridaffine, thetaaffine = model(agnostic, c)

        warped_grid_affine = F.grid_sample(im_g, gridaffine, padding_mode='zeros')
        warped_mask_affine = F.grid_sample(cm, gridaffine, padding_mode='zeros')

        warped_mask = F.grid_sample(cm, gridtps, padding_mode='zeros')
        warped_cloth = F.grid_sample(c, gridtps, padding_mode='border')  #
        warped_grid = F.grid_sample(im_g, gridtps, padding_mode='zeros')
        overlayed_affine = (Ipersp * 0.7 + im * 0.3)
        overlayed_TPS = (warped_cloth * 0.7 + im * 0.3)

        visuals = [[im_h, shape, im_pose],
                   [c, warped_cloth, im_c],
                   [warped_grid, (warped_cloth * 0.7 + im * 0.3), im],
                   [Ipersp, (Ipersp * 0.7 + im * 0.3), warped_grid_affine]]

        save_images(warped_cloth, im_names, warp_cloth_dir)
        save_images(warped_mask * 2 - 1, im_names, warp_mask_dir)
        save_images(warped_grid, im_names, warp_grid_dir)
        save_images(Ipersp, im_names, Ipersp_dir)
        save_images(warped_mask_affine * 2 - 1 , im_names, warped_mask_affine_dir)
        save_images(warped_grid_affine , im_names, warped_grid_affine_dir)
        save_images(overlayed_affine , im_names, overlayed_affine_dir)
        save_images(overlayed_TPS , im_names, overlayed_TPS_dir)

        if (step + 1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step + 1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step + 1, t), flush=True)


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

        outputs = model(torch.cat([agnostic, c], 1))
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        visuals = [[im_h, shape, im_pose],
                   [c, 2 * cm - 1, m_composite],
                   [p_rendered, p_tryon, im]]

        save_images(p_tryon, im_names, try_on_dir)
        if (step + 1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step + 1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step + 1, t), flush=True)


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
    board = SummaryWriter(logdir=os.path.join(opt.tensorboard_dir, opt.name))

    # create model & train
    if opt.stage == 'GMM':
        model = LAGMM(opt)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_gmm(opt, train_loader, model, board)
    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_tom(opt, train_loader, model, board)
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)

    print('Finished test %s, named: %s!' % (opt.stage, opt.name))


if __name__ == "__main__":
    main()
