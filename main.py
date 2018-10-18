import os
import shutil
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import copy

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset_surgical import get_training_set, get_validation_set, get_test_set
from utils import Logger
from train import train_epoch
from validation import val_epoch
import test

if __name__ == '__main__':
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        #opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)

    # class_weight = torch.tensor([3.18450445, 0.34834707, 1.81115213,
    #                              0.45577419, 3.7747372, 1.72608003,
    #                              4.44822003]).cuda()
    #
    # criterion = nn.CrossEntropyLoss(weight=class_weight)

    criterion = nn.CrossEntropyLoss()

    fps=5

    if not opt.no_cuda:
        criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        spatial_transform = Compose([
            crop_method,
            RandomHorizontalFlip(),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration)
        target_transform = ClassLabel()
        training_data = get_training_set(opt, spatial_transform,
                                         temporal_transform, target_transform)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, '{}_{}fps_train_tunefull.log'.format(str(opt.model)+str(opt.model_depth), fps)),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, '{}_{}fps_train_batch_tunefull.log'.format(str(opt.model)+str(opt.model_depth), fps)),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

        # if opt.nesterov:
        #     dampening = 0
        # else:
        #     dampening = opt.dampening
        # optimizer = optim.SGD(
        #     parameters,
        #     lr=opt.learning_rate,
        #     momentum=opt.momentum,
        #     dampening=dampening,
        #     weight_decay=opt.weight_decay,
        #     nesterov=opt.nesterov)
        # scheduler = lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 'min', patience=opt.lr_patience)

        scheduler=None
        optimizer = optim.Adam(parameters, lr=opt.learning_rate)

    if not opt.no_val:
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = ClassLabel()
        validation_data = get_validation_set(
            opt, spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            # batch_size=opt.batch_size,
            batch_size=10,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path,
                         '{}_{}fps_val_tunefull.log'.format(str(opt.model)+str(opt.model_depth), fps)),
            ['epoch', 'loss', 'acc'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))

        if not opt.no_cuda:
            checkpoint = torch.load(opt.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            checkpoint = torch.load(opt.resume_path, map_location='cpu')
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

        assert opt.arch == checkpoint['arch']
        opt.begin_epoch = checkpoint['epoch']

        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    print('run')
    best_val_acc = float("-inf")
    patience = 8
    wait = 0
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
        if not opt.no_val:
            validation_loss, validation_acc = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)

            if i % opt.checkpoint == 0:

                saved_file_path = os.path.join(opt.result_path,
                                      'save_{}_{}_{}fps_tunefull.pth'.format(i, str(opt.model)+str(opt.model_depth), fps))

                if validation_acc > best_val_acc:
                    wait = 0
                    best_file_path = os.path.join(opt.result_path,
                                                  'model_best_{}_{}_{}fps_tunefull.pth'.format(i, str(opt.model)+str(opt.model_depth), fps))

                    # copy the previously saved model file (during training)
                    # as best model file
                    if os.path.isfile(saved_file_path):
                        print('val acc improved from {} to {}'.format(best_val_acc, validation_acc))
                        print('saving {} as {}'.format(saved_file_path, best_file_path))
                        shutil.copyfile(saved_file_path, best_file_path)
                        best_val_acc = validation_acc
                else:
                    wait += 1
                    if wait >= patience:
                        # restore best weights before early stopping
                        checkpoint = torch.load(best_file_path)
                        model.load_state_dict(checkpoint['state_dict'])
                        break # early stopping

        if not opt.no_train and not opt.no_val:
            if scheduler is not None:
                scheduler.step(validation_loss)

    if opt.test:
        spatial_transform = Compose([
            Scale(int(opt.sample_size / opt.scale_in_test)),
            CornerCrop(opt.sample_size, opt.crop_position_in_test),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        # target_transform = VideoID()
        target_transform = ClassLabel()

        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)

        test_logger = Logger(
            os.path.join(opt.result_path,
                         '{}_{}fps_test_tunefull.log'.format(str(opt.model)+str(opt.model_depth), fps)),
            ['acc'])

        # test.test(test_loader, model, opt, test_data.class_names)
        avg_acc = test.my_test(test_loader, model, opt,test_logger)
        print('avg accuracy on test set {}'.format(avg_acc))
