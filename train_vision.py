import os

import sys
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
import torchvision
import torch.optim as optim
import numpy as np
from utils.utils import init_distributed_mode, AverageMeter, reduce_tensor, accuracy
from utils.logger import setup_logger
import clip

from pathlib import Path
import yaml
import pprint
from dotmap import DotMap

import datetime
import shutil
from contextlib import suppress

from clip.model import Transformer, LayerNorm, Transformer_text, CLIP

from modules.video_clip import video_header
from utils.Augmentation import get_augmentation, multiple_samples_collate 
from utils.solver import _lr_scheduler

from functools import reduce
from clip.clip import tokenize
from einops import rearrange

from collections import OrderedDict

class VideoCLIP(nn.Module):
    def __init__(self, clip_model, config) :
        super(VideoCLIP, self).__init__()
        self.visual = clip_model.visual
        self.n_seg = config.data.num_segments
        self.drop_out = nn.Dropout(p=config.network.drop_fc)

        self.fc_proj = nn.Linear(config.network.pyramid.n_embs[-1], config.data.num_classes)  # ori


    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def forward(self, image):
        BT = image.size(0)
        T = self.n_seg
        B = BT // T
        image_features = self.visual(image)

        logits_per_image = self.fc_proj(image_features)

        return logits_per_image

def epoch_saving(epoch, model, optimizer, filename):
    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, filename) #just change to your preferred folder/filename

def best_saving(working_dir, epoch, model, optimizer):
    best_name = '{}/model_best.pt'.format(working_dir)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, best_name)  # just change to your preferred folder/filename


def update_dict(dict):
    new_dict = {}
    for k, v in dict.items():
        new_dict[k.replace('module.', '')] = v
    return new_dict

def get_parser():
    parser = argparse.ArgumentParser('CLIP4Time training and evaluation script for video classification', add_help=False)
    parser.add_argument('--config', '-cfg', type=str, default='clip.yaml', help='global config file')
    parser.add_argument('--log_time', default='001')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')                        
    parser.add_argument("--local_rank", type=int,
                        help='local rank for DistributedDataParallel')
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )

    args = parser.parse_args()
    return args




def main(args):
    global best_prec1
    """ Training Program """
    init_distributed_mode(args)
    if args.distributed:
        print('[INFO] turn on distributed train', flush=True)
    else:
        print('[INFO] turn off distributed train', flush=True)
        os.environ['MASTER_PORT'] = '2334'
        os.environ['NPROC_PER_NODE'] = '1'
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=1, rank=0)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = DotMap(config)

    working_dir = os.path.join('./exp_onehot', config['data']['dataset'], config['network']['arch'], args.log_time)

    if 'something' in config['data']['dataset']:
        from datasets.sth import Video_dataset
    else:
        from datasets.kinetics import Video_dataset

    if dist.get_rank() == 0:
        Path(working_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, working_dir)
        shutil.copy('train_vision.py', working_dir)
        shutil.copy('clip/adapter.py', working_dir)
        shutil.copy('clip/attn.py', working_dir)
        shutil.copy('clip/AttnCBlock.py', working_dir)
        shutil.copy('clip/base_function.py', working_dir)
        shutil.copy('clip/Bottleneck.py', working_dir)
        shutil.copy('clip/ResidualAttentionBlock.py', working_dir)
        shutil.copy('clip/Transformer.py', working_dir)
        shutil.copy('clip/VisualTransformer.py', working_dir)
        if 'eva' in args.config:
            shutil.copy('eva_clip/eva_vit_model.py', working_dir)
        else:
            shutil.copy('clip/model.py', working_dir)


    # build logger, print env and config
    logger = setup_logger(output=working_dir,
                          distributed_rank=dist.get_rank(),
                          name=f'Text4Vis')
    logger.info("------------------------------------")
    logger.info("Environment Versions:")
    logger.info("- Python: {}".format(sys.version))
    logger.info("- PyTorch: {}".format(torch.__version__))
    logger.info("- TorchVison: {}".format(torchvision.__version__))
    logger.info("------------------------------------")
    pp = pprint.PrettyPrinter(indent=4)
    logger.info(pp.pformat(config))
    logger.info("------------------------------------")
    logger.info("storing name: {}".format(working_dir))

    device = "cpu"
    if torch.cuda.is_available():        
        device = "cuda"
        cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    # fix the seed for reproducibility
    seed = config.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # get fp16 model and weight
    model_name = config.network.arch
    if model_name in ["EVA02-CLIP-L-14", "EVA02-CLIP-L-14-336", "EVA02-CLIP-bigE-14", "EVA02-CLIP-bigE-14-plus"]:
        # get evaclip model start ########
        weight_path = {
            "EVA02-CLIP-L-14": './clip-pretrain/EVA02_CLIP_L_psz14_s4B.pt',
            "EVA02-CLIP-L-14-336": './clip-pretrain/EVA02_CLIP_L_336_psz14_s6B.pt',
            "EVA02-CLIP-bigE-14":'./clip-pretrain/EVA02_CLIP_E_psz14_s4B.pt',
            "EVA02-CLIP-bigE-14-plus":'./clip_pretrain/EVA02_CLIP_E_psz14_plus_s9B.pt'
        }
        from eva_clip import create_model_and_transforms
        model, _, preprocess = create_model_and_transforms(model_name, pretrained=weight_path[model_name], force_custom_clip=True, T=config.data.num_segments, side_dim=config.network.side_dim)
        clip_state_dict = model.state_dict()
        # get evaclip model end ########

    else:
        # get fp16 model and weight
        import clip
        model, clip_state_dict = clip.load(
            config.network.arch,
            device='cpu',jit=False,
            internal_modeling=config.network.tm,
            T=config.data.num_segments,
            dropout=config.network.drop_out,
            emb_dropout=config.network.emb_dropout,
            pretrain=config.network.init,
            joint_st = config.network.joint_st,
            side_dim=config.network.side_dim,
            download_root='./clip_pretrain',
            config=config) # Must set jit=False for training  ViT-B/32

    transform_train = get_augmentation(True, config)
    transform_val = get_augmentation(False, config)

    logger.info('train transforms: {}'.format(transform_train.transforms))
    logger.info('val transforms: {}'.format(transform_val.transforms))

    train_data = Video_dataset(
        config.data.train_root, config.data.train_list,
        config.data.label_list, num_segments=config.data.num_segments * config.network.t_ratio,
        modality=config.data.modality,
        image_tmpl=config.data.image_tmpl, random_shift=config.data.random_shift,
        transform=transform_train, dense_sample=config.data.dense, num_sample=config.data.num_sample)

    ################ Few-shot data for training ###########
    if config.data.shot:
        cls_dict = {}
        for item  in train_data.video_list:
            if item.label not in cls_dict:
                cls_dict[item.label] = [item]
            else:
                cls_dict[item.label].append(item)
        import random
        select_vids = []
        K = config.data.shot
        for category, v in cls_dict.items():
            slice = random.sample(v, K)
            select_vids.extend(slice)
        n_repeat = len(train_data.video_list) // len(select_vids)
        train_data.video_list = select_vids * n_repeat
        # print('########### number of videos: {} #########'.format(len(select_vids)))
    ########################################################
    if config.data.num_sample > 1:
        collate_func = multiple_samples_collate
    else:
        collate_func = None

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)                       
    train_loader = DataLoader(train_data,
        batch_size=config.data.batch_size, num_workers=config.data.workers,
        sampler=train_sampler, drop_last=False, collate_fn=collate_func)

    val_data = Video_dataset(
        config.data.val_root, config.data.val_list, config.data.label_list,
        random_shift=False, num_segments=config.data.num_segments * config.network.t_ratio,
        modality=config.data.modality,
        image_tmpl=config.data.image_tmpl,
        transform=transform_val, dense_sample=config.data.dense)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    val_loader = DataLoader(val_data,
        batch_size=config.data.batch_size,num_workers=config.data.workers,
        sampler=val_sampler, drop_last=False)

    model_onehot = VideoCLIP(model, config)

    # freeze model
    n_trainable_params = 0
    if config.network.my_fix_clip:
        for name, param in model_onehot.named_parameters():
            if 'side' in name or 'fc_proj' in name or 'logit_scale' in name or 'prompt' in name or 'lora_bias' in name \
                    or 'positional_embedding_time' in name or 'temporal_class_embedding' in name or 'video' in name \
                    or 'adapt' in name or 'top' in name or 'fine' in name: #  or 'attnpool' in name
                param.requires_grad = True
                print(name, 'True')
                n_trainable_params += param.numel()
            else:
                param.requires_grad = False
                print(name, 'False')
        print('Total trainable params:', n_trainable_params, '(%.2f M)' % (n_trainable_params / 1000000))

    ############# criterion #############
    mixup_fn = None
    if config.solver.mixup:
        logger.info("=> Using Mixup")
        from timm.loss import SoftTargetCrossEntropy
        criterion = SoftTargetCrossEntropy()     
        # smoothing is handled with mixup label transform
        from utils.mixup import Mixup
        mixup_fn = Mixup(
            mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
            prob=1.0, switch_prob=0.5, mode='batch',
            label_smoothing=0.1, num_classes=config.data.num_classes)

    elif config.solver.smoothing:
        logger.info("=> Using label smoothing: 0.1")
        from timm.loss import LabelSmoothingCrossEntropy
        criterion = LabelSmoothingCrossEntropy(smoothing=config.solver.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()


    start_epoch = config.solver.start_epoch
    
    if config.pretrain:
        if os.path.isfile(config.pretrain):
            logger.info("=> loading checkpoint '{}'".format(config.pretrain))
            checkpoint = torch.load(config.pretrain, map_location='cpu')
            model_onehot.load_state_dict(checkpoint['model_state_dict'], strict=False)
            del checkpoint
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.resume))

    if config.network.sync_bn:
        bn_lr = config.solver.lr
    else:
        bn_lr = config.solver.lr# / config.solver.grad_accumulation_steps

    clip_params = []
    other_params = []
    bn_params = []

    for name, param in model_onehot.named_parameters():
        if 'bn' in name and 'side' in name:
            bn_params.append(param)
        elif 'visual' in name and 'control_point' not in name and 'time_embedding' not in name:
            clip_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.AdamW([{'params': clip_params, 'lr': config.solver.lr * config.solver.clip_ratio},
                             {'params': other_params, 'lr': config.solver.lr},
                             {'params': bn_params, 'lr': bn_lr}],
                            betas=config.solver.betas, lr=config.solver.lr, eps=1e-8,
                            weight_decay=config.solver.weight_decay)

    if config.resume:
        if os.path.isfile(config.resume):
            logger.info("=> loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume, map_location='cpu')
            model_onehot.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            start_epoch = checkpoint['epoch'] + 1
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                   .format(config.evaluate, checkpoint['epoch']))
            del checkpoint
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.pretrain))

    lr_scheduler = _lr_scheduler(config, optimizer)
        
    if args.distributed:
        model_onehot = DistributedDataParallel(model_onehot.cuda(), device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model_onehot.module

    scaler = GradScaler() if args.precision == "amp" else None

    best_prec1 = 0.0
    if config.solver.evaluate:
        logger.info(("===========evaluate==========="))
        prec1 = validate(
            start_epoch,
            val_loader, device, 
            model_onehot, config, logger)
        return

    for epoch in range(start_epoch, config.solver.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)        
        train(model_onehot, train_loader, optimizer, criterion, scaler,
              epoch, device, lr_scheduler, config, mixup_fn, logger)

        if (epoch+1) % config.logging.eval_freq == 0:  # and epoch>0
            if config.logging.skip_epoch is not None and epoch in config.logging.skip_epoch:
                continue
            prec1 = validate(epoch, val_loader, device, model_onehot, config, logger)

            if dist.get_rank() == 0:
                is_best = prec1 >= best_prec1
                best_prec1 = max(prec1, best_prec1)
                logger.info('Testing: {}/{}'.format(prec1,best_prec1))
                logger.info('Saving:')
                filename = "{}/last_model.pt".format(working_dir)

                epoch_saving(epoch, model_without_ddp, optimizer, filename)
                if is_best:
                    best_saving(working_dir, epoch, model_without_ddp, optimizer)


def train(model, train_loader, optimizer, criterion, scaler,
          epoch, device, lr_scheduler, config, mixup_fn, logger):
    """ train a epoch """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    end = time.time()
    for i,(images, list_id) in enumerate(train_loader):
        if config.solver.type != 'monitor':
            if (i + 1) == 1 or (i + 1) % 10 == 0:
                lr_scheduler.step(epoch + i / len(train_loader))
        # lr_scheduler.step()
        # optimizer.zero_grad()
        data_time.update(time.time() - end)
        images = images.view((-1,config.data.num_segments * config.network.t_ratio,3)+images.size()[-2:])

        if mixup_fn is not None:
            images = images.transpose(1, 2)  # b t c h w -> b c t h w
            images, list_id = mixup_fn(images, list_id)
            images = images.transpose(1, 2)

        list_id = list_id.to(device)
        b,t,c,h,w = images.size()
        images= images.view(-1,c,h,w) # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class

        with model.no_sync():
            with autocast():
                logits = model(images)
                loss = criterion(logits, list_id)
                # loss regularization
                # loss = loss / config.solver.grad_accumulation_steps
            if scaler is not None:
                # back propagation
                scaler.scale(loss).backward()
            else:
                loss.backward()

        # prec1, prec5 = accuracy(logits, list_id, topk=(1, 5))
        # top1.update(prec1.item(), logits.size(0))
        losses.update(loss.item(), logits.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        cur_iter = epoch * len(train_loader) +  i
        max_iter = config.solver.epochs * len(train_loader)
        eta_sec = batch_time.avg * (max_iter - cur_iter + 1)
        eta_sec = str(datetime.timedelta(seconds=int(eta_sec)))

        if i % config.logging.print_freq == 0:
            logger.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.2e}, eta: {3}\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        #  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                             epoch, i, len(train_loader), eta_sec, batch_time=batch_time, data_time=data_time, 
                            #  top1=top1,
                             loss=losses, lr=optimizer.param_groups[-1]['lr'])))  # TODO

def validate(epoch, val_loader, device, model, config, logger):
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (image, class_id) in enumerate(val_loader):
            image = image.view((-1, config.data.num_segments * config.network.t_ratio, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            class_id = class_id.to(device)
            image_input = image.to(device)
            image_input = image_input.view(-1, c, h, w)
            logits = model(image_input)  # B 400
            prec = accuracy(logits, class_id, topk=(1, 5))
            prec1 = reduce_tensor(prec[0])
            prec5 = reduce_tensor(prec[1])

            top1.update(prec1.item(), class_id.size(0))
            top5.update(prec5.item(), class_id.size(0))

            if i % config.logging.print_freq == 0:
                logger.info(
                    ('Test: [{0}/{1}]\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                         i, len(val_loader), top1=top1, top5=top5)))

    logger.info(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5)))
    return top1.avg


if __name__ == '__main__':
    args = get_parser() 
    main(args)

