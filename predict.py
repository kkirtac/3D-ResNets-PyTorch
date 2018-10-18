import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
import subprocess
from models import resnet
import functools
import copy
from more_itertools import consecutive_groups
from more_itertools import chunked
from PIL import Image
import torch.utils.data as data
from spatial_transforms import (
    Compose, Normalize, Scale, CornerCrop, ToTensor)
from temporal_transforms import LoopPadding
from target_transforms import ClassLabel
import argparse
import errno


def calc_conf_matrix(y_test, y_pred,
                     class_names, save_dir,
                     save_filename='confmat.jpg'):
    """
    calculates confusion matrix and plots it.
    Save the resulting plot into save_dir with name save_filename.

    :param y_test: ground truth phase labels.
    :param y_pred: predicted phase labels.
    :param class_names: list of phase labels.
    :param save_dir: save the figure to this directory.
    :param save_filename: save the figure with this name.
    :return: None
    """

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, save_dir,
                          save_filename,
                          classes=class_names,
                          normalize=True,
                          title='Normalized confusion matrix')

    # plt.show()

def plot_confusion_matrix(cm, save_dir,
                          save_filename,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, save_filename))


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def get_default_image_loader():
    return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def make_dataset(video_dir_path, video_phase_annotation_path,
                 phase_list, sample_duration):
    """
    Construct dataset of samples from a given video directory path.

    Each sample is a python dictionary containing the video path and
    indices of 16-frame portions from that video file, as well as the
    associated class label of that portion.

    video_phase_annotation_path file is supposed to contain a frame index and
    corresponding class label (surgical phase) at each of its row.

    :param root_path: Absolute path to the root directory of video and timestamp files.
    :param phase_list: List of all possible phases (classes)
    :param subset: training, validation, or testing.
    :param idx_subset: list of exact video file indices for the chosen subset.
    :param sample_duration: number of frames each sample contains
    :return: list of samples.
    """

    class_to_idx = {phase_list[i]: i for i in range(len(phase_list))}

    dataset = []

    df = pd.read_csv(video_phase_annotation_path, delim_whitespace=True)

    sample = {
        'video': video_dir_path,
        'video_id': os.path.basename(video_dir_path),
    }

    for phase in phase_list:

        df_phase = df.loc[df['Phase'] == phase]

        for group in consecutive_groups(df_phase['Frame'] + 1):

            for chunk in chunked(group, sample_duration):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = chunk
                sample_j['label'] = class_to_idx[phase]
                dataset.append(sample_j)

    return dataset


class SurgicalDataset(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 video_dir_path,
                 video_phase_annotation_path,
                 phase_list,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):


        self.data = make_dataset(
            video_dir_path, video_phase_annotation_path,
            phase_list, sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()
        self.class_names = phase_list

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def extract_frames(in_video_path, out_dir_path):
    """
    extracts frames of the in_video_path and preprocesses each frame.
    Creates a folder in out_dir_path, which has the same name with video file, and saves
    resultant frames into that folder.
    :param in_video_path: full file path to the video input
    :param out_dir_path: directory path to save the preprocessed frames.
    :return: None
    """
    name, ext = os.path.splitext(os.path.basename(os.path.abspath(in_video_path)))

    try:
        os.makedirs(os.path.abspath(out_dir_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # raises the error again

    dst_dir_path = os.path.join(os.path.abspath(out_dir_path), name)

    try:
        if os.path.exists(dst_dir_path):
            if not os.path.exists(os.path.join(dst_dir_path, 'image_00001.jpg')):
                os.rmdir(dst_dir_path)
                print('remove {}'.format(dst_dir_path))
                os.mkdir(dst_dir_path)
        else:
            os.mkdir(dst_dir_path)
    except:
        print(dst_dir_path)

    cmd = 'ffmpeg -i {} -vf crop=800:480:20:0 {}/image_%05d.jpg'.format(os.path.abspath(in_video_path), dst_dir_path)
    print(cmd)
    subprocess.call(cmd, shell=True)
    print('\n')

    if os.path.isdir(dst_dir_path):
        if os.path.isfile(os.path.join(dst_dir_path, 'image_00001.jpg')):
            return True


def get_model(opt):

    model = resnet.resnet18(
        num_classes=7,
        shortcut_type='A',
        sample_size=112,
        sample_duration=16)

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)

        if opt.model_path:
            print('loading trained model {}'.format(opt.model_path))
            pretrain = torch.load(opt.model_path)

            model.load_state_dict(pretrain['state_dict'])

    else:
        if opt.model_path:
            print('loading trained model {}'.format(opt.model_path))
            pretrain = torch.load(opt.model_path, map_location='cpu')

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in pretrain['state_dict'].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            model.load_state_dict(new_state_dict)

    return model

def get_dataloader(opt):

    mean = [ 110.63666788 / 255, 103.16065604 / 255,
             96.29023126 / 255]
    std = [ 1, 1, 1]

    norm_method = Normalize(mean, std)

    spatial_transform = Compose([
        Scale(112),
        CornerCrop(112, 'c'),
        ToTensor(255), norm_method
    ])

    temporal_transform = LoopPadding(16)
    target_transform = ClassLabel()

    test_data = SurgicalDataset(os.path.abspath(opt.frames_path),
                                os.path.abspath(opt.video_phase_annotation_path),
                                opt.class_names,
                                spatial_transform=spatial_transform,
                                temporal_transform=temporal_transform,
                                target_transform=target_transform,
                                sample_duration=16)

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    return test_loader


def run_prediction(opt):

    model = get_model(opt)

    data_loader = get_dataloader(opt)

    print('test')

    model.eval()

    with torch.no_grad():

        batch_time = AverageMeter()
        data_time = AverageMeter()
        end_time = time.time()

        try:
            os.makedirs(opt.result_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # raises the error again

        opt.result_path = os.path.abspath(opt.result_path)

        result_filepath = os.path.join(opt.result_path, opt.prediction_result_filename)
        resultfile = open(result_filepath, 'w', newline='')
        fieldnames = ['frame', 'prediction', 'ground-truth']
        writer = csv.DictWriter(resultfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            # clip, target = data_loader.dataset[i]

            inputs = Variable(inputs)
            outputs = model(inputs)
            sample_duration = inputs.size(2)

            outputs = F.softmax(outputs, dim=-1)

            pred_max_idx = outputs.topk(1)[1]
            pred_label = opt.class_names[pred_max_idx.item()]

            for j in range(i*sample_duration,(i+1)*sample_duration):
                writer.writerow({'frame': j,
                                 'prediction': pred_max_idx.item(),
                                 'ground-truth': targets.item()})

            print(pred_label, pred_max_idx.item())

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('[{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time))

        resultfile.close()

        if os.path.isfile(result_filepath):
            df = pd.read_csv(result_filepath)

            y_test = df['ground-truth']
            y_pred = df['prediction']

            filename, ext = os.path.splitext(os.path.basename(result_filepath))
            save_filename = filename+'.jpg'

            print('calculating and saving confusion matrix', end='...')

            calc_conf_matrix(y_test, y_pred,
                             opt.class_names,
                             opt.result_path,
                             save_filename=save_filename)

            print('done')

            print('confusion matrix {} is saved into {}'.format(save_filename, opt.result_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--video_path',
        required=False,
        type=str,
        help='Path to the input video file')
    parser.add_argument(
        '--frames_path',
        required=False,
        type=str,
        help='Directory path of video (frames)')
    parser.add_argument(
        '--video_phase_annotation_path',
        required=True,
        type=str,
        help='Phase annotation file path of the given video')
    parser.add_argument(
        '--result_path',
        required=False,
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--model_path',
        required=True,
        default='',
        type=str,
        help='Trained model (.pth)')
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--no_frame_extraction', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_frame_extraction=False)
    parser.add_argument(
        '--prediction_result_filename',
        required=False,
        default='',
        type=str,
        help='Output Predictions')

    opt = parser.parse_args()

    opt.class_names = ['Preparation',
                       'CalotTriangleDissection',
                       'ClippingCutting',
                       'GallbladderDissection',
                       'GallbladderPackaging',
                       'CleaningCoagulation',
                       'GallbladderRetraction']

    # first, extract frames and preprocess each frame, before running prediction

    frames_ready = False

    if not opt.no_frame_extraction:
        if opt.video_path:
            if not opt.frames_path:
                video_name, ext = os.path.splitext(os.path.basename(os.path.abspath(opt.video_path)))
                opt.frames_path = os.path.join(os.path.abspath(os.getcwd()), video_name)
                print('Destination frames directory not specified. '
                      'Extracting frames to {}'.format())

            frames_ready = extract_frames(opt.video_path, opt.frames_path)

        else:
            print('Missing video input file. '
                  'Please specify a video input file by setting --video_path option.'
                  'Hit a key to terminate...')
            input()

    else:
        if not opt.frames_path:
            print('Missing video frames directory. '
                  'Please specify a video frames directory by setting --frames_path option. '
                  'Hit a key to terminate...')
            input()
        else:
            if os.path.isdir(opt.frames_path):
                if not os.path.isfile(os.path.join(opt.frames_path, 'image_00001.jpg')):
                    print('Invalid video frames directory. '
                          'Please make sure the directory contains frames. '
                          'Hit a key to terminate...')
                    input()
                else:
                    frames_ready = True
            else:
                print('Invalid video frames directory. '
                      'Please specify a video frames directory by setting --frames_path option. '
                      'Hit a key to terminate...')
                input()

    if not opt.result_path:
        opt.result_path = os.path.join(os.path.abspath(os.getcwd()), 'results')

    if not opt.prediction_result_filename:
        if opt.video_path:
            video_name, ext = os.path.splitext(os.path.basename(os.path.abspath(opt.video_path)))
        else:
            if opt.frames_path:
                video_name = os.path.basename(os.path.abspath(opt.frames_path))

        opt.prediction_result_filename = 'predictions_{}.csv'.format(video_name)


    if frames_ready:
        run_prediction(opt)
