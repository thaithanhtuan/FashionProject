#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import LAGMM, UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint, GicLoss

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images
# python trainLA.py --name LAgmm_train_getData --stage GMM --workers 4 --save_count 5000 --shuffle


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)

    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 100)
    parser.add_argument("--keep_step", type=int, default = 100000)
    parser.add_argument("--decay_step", type=int, default = 100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt


def DT(x1,y1,x2,y2):
    dt = torch.sqrt(torch.mul(x1- x2, x1- x2) + torch.mul(y1-y2,y1-y2))
    return dt


def train_gmm(opt, train_loader, model, board):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    gicloss = GicLoss(opt)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        im_pose = inputs['pose_image'].cuda()
        im_h = inputs['head'].cuda()
        shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_c =  inputs['parse_cloth'].cuda()
        im_g = inputs['grid_image'].cuda()

        gridtps, thetatps, Ipersp, gridaffine, thetaaffine = model(agnostic, c)
        warped_cloth = F.grid_sample(Ipersp, gridtps, padding_mode='border') #
        warped_maskaffine = F.grid_sample(cm, gridaffine, padding_mode='zeros')
        warped_mask = F.grid_sample(warped_maskaffine, gridtps, padding_mode='zeros')

        warped_gridaffine = F.grid_sample(im_g, gridaffine, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, gridtps, padding_mode='zeros')

        visuals = [[im_h, shape, im_pose],
                   [c, warped_cloth, im_c],
                   [warped_grid, (warped_cloth * 0.7 + im * 0.3), im],
                   [Ipersp, (Ipersp * 0.7 + im * 0.3), warped_gridaffine]]

        Lwarp = criterionL1(warped_cloth, im_c)
        Lpersp = criterionL1(Ipersp, im_c)

        LgicTPS = gicloss(gridtps)
        LgicTPS = LgicTPS / (gridtps.shape[0] * gridtps.shape[1] * gridtps.shape[2])
        # shape of grid: N, Hout, Wout,2: N x 5 x 5 x 2
        # grid shape: N x 5 x 5 x 2

        loss = 0.7*Lpersp + 0.3*Lwarp + 40 * LgicTPS
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            board.add_scalar('0.5*Lpersp', (0.5*Lpersp).item(), step+1)
            board.add_scalar('40*LgicTPS', (40*LgicTPS).item(), step+1)
            board.add_scalar('0.5*Lwarp', (0.5*Lwarp).item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f, (40*LgicTPS): %.8f, 0.5*Lpersp: %.6f, 0.5*Lwarp: %.6f' % (step+1, t, loss.item(), (40*LgicTPS).item(), (0.5*Lpersp).item(), (0.5*Lwarp).item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))


def train_tom(opt, train_loader, model, board):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()

        outputs = model(torch.cat([agnostic, c],1))
        p_rendered, m_composite = torch.split(outputs, 3,1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite+ p_rendered * (1 - m_composite)

        visuals = [ [im_h, shape, im_pose],
                   [c, cm*2-1, m_composite*2-1],
                   [p_rendered, p_tryon, im]]

        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        loss_mask = criterionMask(m_composite, cm)
        loss = loss_l1 + loss_vgg + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            board.add_scalar('L1', loss_l1.item(), step+1)
            board.add_scalar('VGG', loss_vgg.item(), step+1)
            board.add_scalar('MaskL1', loss_mask.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f'
                    % (step+1, t, loss.item(), loss_l1.item(),
                    loss_vgg.item(), loss_mask.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))



def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

    # create dataset
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))

    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        model = LAGMM(opt)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_gmm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth'))
    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_tom(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'tom_final.pth'))
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)


    print('Finished training %s, nameed: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
