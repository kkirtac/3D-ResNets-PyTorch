import torch
import torch.utils.data as data
from PIL import Image
import os
import functools
import copy
import pandas as pd
from more_itertools import consecutive_groups
from more_itertools import chunked

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


def make_dataset(root_path, phase_list, subset, idx_subset, sample_duration):
    """
    Construct dataset of samples from given root directory path.
    Each sample is a python dictionary containing the video path and
    indices of 16-frame portions from that video file, as well as the
    associated class label of that portion.
    There must exist a video file(.mp4) and associated timestamp file(.txt) under root_path.
    Each timestamp file is supposed to contain a frame index and
    corresponding class label (surgical phase) at each of its row.
    :param root_path: Absolute path to the root directory of video and timestamp files.
    :param phase_list: List of all possible phases (classes)
    :param subset: training, validation, or testing.
    :param idx_subset: list of exact video file indices for the chosen subset.
    :param sample_duration: number of frames each sample contains
    :return: list of samples.
    """

    phase_root = os.path.join(root_path, 'phase_annotations')
    phase_filename_list = sorted([f for f in os.listdir(phase_root) if f.endswith('.txt')])

    frames_root = os.path.join(root_path, 'frames')
    video_filename_list = sorted([f for f in os.listdir(frames_root) if os.path.isdir(os.path.join(frames_root, f))])

    class_to_idx = {phase_list[i]: i for i in range(len(phase_list))}

    dataset = []

    for v, phase_filename in enumerate(phase_filename_list):

        if v in idx_subset:

            phase_filepath = os.path.join(phase_root, phase_filename)

            df = pd.read_csv(phase_filepath, delim_whitespace=True)

            sample = {
                'video': os.path.join(frames_root, video_filename_list[v]),
                'video_id': video_filename_list[v].split('.')[0],
                'subset': subset
            }

            for phase in phase_list:

                df_phase = df.loc[df['Phase'] == phase]

                for group in consecutive_groups(df_phase['Frame']+1):

                    for chunk in chunked(group, sample_duration):

                        if len(chunk) == sample_duration:
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
                 root_path,
                 phase_list,
                 subset,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):

        # create training/validation/testing indices
        if subset in ['training']:
            idx_subset=[0, 1, 2, 3, 6, 7, 10, 12,
                        14, 15, 17, 18, 19, 20, 22,
                        23, 24, 25, 26, 27, 28, 29,
                        30, 31, 32, 34, 35, 36, 37,
                        38, 39, 40, 41, 42, 43, 44,
                        45, 46, 47, 48, 49, 50, 51,
                        52, 53, 56, 57, 58, 59, 61,
                        62, 63, 65, 67, 68, 69, 71,
                        72, 73, 74, 75, 76, 78, 79]
        elif subset in ['validation']:
            idx_subset=[66, 8, 70, 4, 21, 64, 16, 9]
        elif subset in ['testing']:
            idx_subset=[77, 5, 13, 54, 55, 33, 60, 11]
        else:
            print('Missing subset name. '
                  'Error initiliazing the object.')
            return

        self.data = make_dataset(
            root_path, phase_list, subset, idx_subset, sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

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
