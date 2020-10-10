import argparse
import random
import math

from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from miscc.config import cfg
from miscc.config import cfg, cfg_from_file

from datasets import Dataset

from dataset import MultiResolutionDataset
from model import StyledGenerator, Discriminator
import os

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(dataset, batch_size, image_size=4):
    dataset.resolution = image_size
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1, drop_last=True)

    return loader


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult


subdataset_idx = None
def get_dataloader(imsize, batch_size):
    global subdataset_idx
    bshuffle = True
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    dataset = Dataset(cfg.DATA_DIR,
        imsize=imsize,
        transform=image_transform)

    if cfg.TRAIN.DATASET_SIZE != -1:
        if subdataset_idx is None:
            subdataset_idx = random.sample(
                range(0, len(dataset)), cfg.TRAIN.DATASET_SIZE)
        dataset = torch.utils.data.Subset(dataset, subdataset_idx)

    assert dataset
    print('training dataset size: ', len(dataset))

    num_gpu = len(cfg.GPU_ID.split(','))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size * num_gpu,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))
    return dataloader


def train(args, generator, discriminator):

    torch.autograd.set_detect_anomaly(True)
    step = int(math.log2(args.init_size)) - 2
    resolution = 4 * 2 ** step
    # loader = sample_data(
    #     dataset, args.batch.get(resolution, args.batch_default), resolution
    # )
    batch_size = args.batch.get(resolution, args.batch_default)
    loader = get_dataloader(resolution, batch_size)
    data_loader = iter(loader)

    adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
    adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

    pbar = tqdm(range(3_000_000))

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    used_sample = 0

    max_step = int(math.log2(args.max_size)) - 2
    final_progress = False

    criterion = nn.BCELoss(reduce=False).cuda()
    criterion_one = nn.BCELoss().cuda()

    for i in pbar:
        discriminator.zero_grad()

        alpha = min(1, 1 / args.phase * (used_sample + 1))

        if (resolution == args.init_size and args.ckpt is None) or final_progress:
            alpha = 1

        if used_sample > args.phase * 2:
            used_sample = 0
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True
                ckpt_step = step + 1

            else:
                alpha = 0
                ckpt_step = step

            resolution = 4 * 2 ** step

            # loader = sample_data(
            #     dataset, args.batch.get(resolution, args.batch_default), resolution
            # )
            # data_loader = iter(loader)
            batch_size = args.batch.get(resolution, args.batch_default)
            loader = get_dataloader(resolution, batch_size)
            data_loader = iter(loader)

            torch.save(
                {
                    'generator': generator.module.state_dict(),
                    'discriminator': discriminator.module.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'g_running': g_running.state_dict(),
                },
                f'checkpoint/train_step-{ckpt_step}.model',
            )

            adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
            adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

        try:
            real_image, _, _, _, masks, aux_masks = next(data_loader)

        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image, _, _, _, masks, aux_masks = next(data_loader)

        used_sample += real_image.shape[0]

        b_size = real_image.size(0)
        real_image = real_image.cuda()
        masks = masks.cuda()
        aux_masks = aux_masks.cuda()

        # discriminator(real_image * masks, step=step, alpha=alpha)

        if args.loss == 'wgan-gp':
            real_predict = discriminator(real_image * masks, step=step, alpha=alpha)
            real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
            (-real_predict).backward()

        elif args.loss == 'r1':
            real_image.requires_grad = True
            real_scores = discriminator(real_image * masks, step=step, alpha=alpha)
            real_predict = F.softplus(-real_scores).mean()
            real_predict.backward(retain_graph=True)

            grad_real = grad(
                outputs=real_scores.sum(), inputs=real_image, create_graph=True
            )[0]
            grad_penalty = (
                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
            ).mean()
            grad_penalty = 10 / 2 * grad_penalty
            grad_penalty.backward()
            if i%10 == 0:
                grad_loss_val = grad_penalty.item()

        # _fg = masks == 0
        # rev_masks = torch.zeros_like(masks)
        # rev_masks.masked_fill_(_fg, 1.0)

        # _, real_logits_glb = discriminator(real_image * masks, step=step, alpha=alpha)
        # real_logits_lcl, _ = discriminator(real_image, step=step, alpha=alpha, mask=rev_masks)

        if args.mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(
                4, b_size, code_size, device='cuda'
            ).chunk(4, 0)
            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]

        else:
            gen_in1, gen_in2 = torch.randn(2, b_size, code_size, device='cuda').chunk(
                2, 0
            )
            gen_in1 = gen_in1.squeeze(0)
            gen_in2 = gen_in2.squeeze(0)

        fake_image = generator(gen_in1, step=step, alpha=alpha)
        fake_predict = discriminator(fake_image * aux_masks, step=step, alpha=alpha)
        # _, fake_logits_glb = discriminator(fake_image * aux_masks, step=step, alpha=alpha)

        # fake_labels = torch.zeros_like(real_logits_glb)
        # real_labels = torch.ones_like(real_logits_glb)
        # errD_fake = criterion_one(fake_logits_glb, fake_labels)  # Real/Fake loss for the fake image
        # errD_real = criterion_one(real_logits_glb, real_labels)  # Real/Fake loss for the real image
        # errD0 = (errD_real + errD_fake) * cfg.TRAIN.BG_LOSS_WT_GLB

        # fake_logits_lcl, _ = discriminator(fake_image, step=step, alpha=alpha)

        # fake_labels = torch.zeros_like(real_logits_lcl[1])
        # ext, output, fnl_masks = real_logits_lcl
        # weights_real = torch.ones_like(output)
        # real_labels = torch.ones_like(output)

        # # print(fnl_masks.size())

        # # for i in range(batch_size):
        # invalid_patch = fnl_masks != 0.0
        # weights_real.masked_fill_(invalid_patch, 0.0)

        # norm_fact_real = weights_real.sum()
        # norm_fact_fake = weights_real.shape[0]*weights_real.shape[1]*weights_real.shape[2]*weights_real.shape[3]
        # real_logits_lcl = ext, output

        # # Real/Fake loss for 'real background' (on patch level)
        # errD_real_uncond = criterion(real_logits_lcl[1], real_labels)
        # errD_real_uncond = torch.mul(errD_real_uncond, weights_real)  # Masking output units which correspond to receptive fields which lie within the boundin box
        # errD_real_uncond = errD_real_uncond.mean()

        # errD_fake_uncond = criterion(fake_logits_lcl[1], fake_labels)  # Real/Fake loss for 'fake background' (on patch level)
        # errD_fake_uncond = errD_fake_uncond.mean()

        # if norm_fact_real > 0:    # Normalizing the real/fake loss for background after accounting the number of masked members in the output.
        #     errD_real = errD_real_uncond * ((norm_fact_fake * 1.0) / (norm_fact_real * 1.0))
        # else:
        #     errD_real = errD_real_uncond

        # errD_fake = errD_fake_uncond
        # errD1 = (errD_real + errD_fake) * cfg.TRAIN.BG_LOSS_WT_LCL

        # errD_real_uncond_classi = criterion(real_logits_lcl[0], weights_real)
        # errD_real_uncond_classi = errD_real_uncond_classi.mean()
        # errD_classi = errD_real_uncond_classi * cfg.TRAIN.BG_CLASSI_WT

        # errD = errD0 #+ errD1 + errD_classi
        # errD.backward()

        if args.loss == 'wgan-gp':
            fake_predict = fake_predict.mean()
            fake_predict.backward()

            eps = torch.rand(b_size, 1, 1, 1).cuda()
            x_hat = eps * real_image.data + (1 - eps) * fake_image.data
            x_hat.requires_grad = True
            hat_predict = discriminator(x_hat * masks, step=step, alpha=alpha)
            grad_x_hat = grad(
                outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
            )[0]
            grad_penalty = (
                (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
            ).mean()
            grad_penalty = 10 * grad_penalty
            grad_penalty.backward()
            if i%10 == 0:
                grad_loss_val = grad_penalty.item()
                disc_loss_val = (-real_predict + fake_predict).item()

        elif args.loss == 'r1':
            fake_predict = F.softplus(fake_predict).mean()
            fake_predict.backward()
            if i%10 == 0:
                disc_loss_val = (real_predict + fake_predict).item()

        # if i % 1000 == 0:
        #     disc_loss_val = errD.item()

        d_optimizer.step()

        if (i + 1) % n_critic == 0:
            generator.zero_grad()

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            fake_image = generator(gen_in2, step=step, alpha=alpha)
            predict = discriminator(fake_image * aux_masks, step=step, alpha=alpha)

            # _, outputs = discriminator(fake_image * aux_masks, step=step, alpha=alpha)
            # real_labels = torch.ones_like(outputs[1])
            # errG0 = criterion_one(outputs[1], real_labels)
            # errG0 = errG0 * cfg.TRAIN.BG_LOSS_WT_GLB

            # outputs, _ = discriminator(fake_image, step=step, alpha=alpha)
            # real_labels = torch.ones_like(outputs[1])
            # errG1 = criterion_one(outputs[1], real_labels)
            # errG1 = errG1 * cfg.TRAIN.BG_LOSS_WT_LCL

            # errG_classi = criterion_one(outputs[0], real_labels) # Background/Foreground classification loss for the fake background image (on patch level)
            # errG_classi = errG_classi * cfg.TRAIN.BG_CLASSI_WT

            # errG = errG0 #+ errG1 + errG_classi

            if args.loss == 'wgan-gp':
                loss = -predict.mean()

            elif args.loss == 'r1':
                loss = F.softplus(-predict).mean()

            if i%10 == 0:
                gen_loss_val = loss.item()
                # gen_loss_val = errG.item()

            # errG.backward()
            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator.module)

            requires_grad(generator, False)
            requires_grad(discriminator, True)

        if (i + 1) % 10000 == 0:
            images = []

            gen_i, gen_j = args.gen_sample.get(resolution, (10, 5))

            with torch.no_grad():
                for _ in range(gen_i):
                    images.append(
                        g_running(
                            torch.randn(gen_j, code_size).cuda(), step=step, alpha=alpha
                        ).data.cpu()
                    )

            utils.save_image(
                torch.cat(images, 0),
                f'sample/{str(i + 1).zfill(6)}.png',
                nrow=gen_i,
                normalize=True,
                range=(-1, 1),
            )

        if (i + 1) % 100000 == 0:
            torch.save(
                g_running.state_dict(), f'checkpoint/{str(i + 1).zfill(6)}.model'
            )

        state_msg = (
            f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
        )
        # state_msg = (
        #     f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
        #     f'Alpha: {alpha:.5f}'
        # )
        pbar.set_description(state_msg)


if __name__ == '__main__':
    code_size = 512
    batch_size = 16
    n_critic = 1

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    parser.add_argument('--path', type=str, default=None, help='path of specified dataset')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/train.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='-1')
    # parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    # parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument(
        '--phase',
        type=int,
        default=600_000,
        help='number of samples used for each training phases',
    )
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--init_size', default=8, type=int, help='initial image size')
    parser.add_argument('--max_size', default=1024, type=int, help='max image size')
    parser.add_argument(
        '--ckpt', default=None, type=str, help='load from previous checkpoints'
    )
    parser.add_argument(
        '--no_from_rgb_activate',
        action='store_true',
        help='use activate in from_rgb (original implementation)',
    )
    parser.add_argument(
        '--mixing', action='store_true', help='use mixing regularization'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='wgan-gp',
        choices=['wgan-gp', 'r1'],
        help='class of gan loss',
    )

    args = parser.parse_args()

    # if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)

    if args.gpu_id != '-1':
        cfg.GPU_ID = args.gpu_id

    if args.path is not None:
        cfg.DATA_DIR = args.path

    s_gpus = cfg.GPU_ID.split(',')
    gpus = [int(ix) for ix in s_gpus]
    num_gpus = len(gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID
    # torch.cuda.set_device(gpus[0])

    generator = nn.DataParallel(StyledGenerator(code_size)).cuda()
    discriminator = nn.DataParallel(
        Discriminator(from_rgb_activate=not args.no_from_rgb_activate)
    ).cuda()
    g_running = StyledGenerator(code_size).cuda()
    g_running.train(False)

    g_optimizer = optim.Adam(
        generator.module.generator.parameters(), lr=args.lr, betas=(0.0, 0.99)
    )
    g_optimizer.add_param_group(
        {
            'params': generator.module.style.parameters(),
            'lr': args.lr * 0.01,
            'mult': 0.01,
        }
    )
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

    accumulate(g_running, generator.module, 0)

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)

        generator.module.load_state_dict(ckpt['generator'])
        discriminator.module.load_state_dict(ckpt['discriminator'])
        g_running.load_state_dict(ckpt['g_running'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    # dataset = MultiResolutionDataset(args.path, transform)

    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 12}

    else:
        args.lr = {}
        args.batch = {}

    args.gen_sample = {512: (8, 4), 1024: (4, 2)}

    args.batch_default = 12

    train(args, generator, discriminator)
