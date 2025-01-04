import os
import pandas as pd
import numpy as np
import random
from torchvision import transforms
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
import csv
from PIL import Image


class ClipSubstractMean(object):
    def __init__(self, b=104, g=117, r=123):
        self.means = np.array((101., 98., 90.))  # B=90.25, G=97.66, R=101.41

    def __call__(self, buffer):
        new_buffer = buffer - self.means
        return new_buffer


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=(112, 112)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, buffer):
        h, w = buffer.shape[1], buffer.shape[2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        new_buffer = np.zeros((buffer.shape[0], new_h, new_w, 3))
        for i in range(buffer.shape[0]):
            image = buffer[i, :, :, :]
            image = image[top: top + new_h, left: left + new_w]
            new_buffer[i, :, :, :] = image

        return new_buffer


class CenterCrop(object):
    """Crop the image in a sample at the center.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=(112, 112)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, buffer):
        h, w = buffer.shape[1], buffer.shape[2]
        new_h, new_w = self.output_size

        top = int(round(h - new_h) / 2.)
        left = int(round(w - new_w) / 2.)

        new_buffer = np.zeros((buffer.shape[0], new_h, new_w, 3))
        for i in range(buffer.shape[0]):
            image = buffer[i, :, :, :]
            image = image[top: top + new_h, left: left + new_w]
            new_buffer[i, :, :, :] = image

        return new_buffer


class RandomHorizontalFlip(object):
    """Horizontally flip the given Images randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __call__(self, buffer):
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, buffer):
        # swap color axis because
        # numpy image: batch_size x T x H x W x C
        # torch image: batch_size x C X T x H X W
        img = torch.from_numpy(buffer.transpose((3, 0, 1, 2)))

        return img.float().div(255)


class HMDBDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            root_dir (str): path to n_frames_jpg folders.
            info_list (str): path to annotation file.
            split(str): whether create trainset. Default='train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
    """

    def __init__(self, root_dir, info_list, split='train', clip_len=50):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.split = split
        if split == 'train':
            self.transform = transforms.Compose(
                [ClipSubstractMean(),
                 RandomCrop(),
                 RandomHorizontalFlip(),
                 ToTensor()])
        else:
            self.transform = transforms.Compose(
                [ClipSubstractMean(),
                 CenterCrop(),
                 ToTensor()])

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, index):
        while True:
            # Loading and preprocessing.
            video_path = self.landmarks_frame.iloc[index, 0]
            labels = self.landmarks_frame.iloc[index, 1]
            buffer = self.get_resized_frames_per_video(video_path)

            if buffer is not None:
                if self.transform:
                    buffer = self.transform(buffer)
                return buffer, torch.from_numpy(np.array(labels))
        
            # If buffer is None, increment the index and continue
            index = (index + 1) % len(self.landmarks_frame)

    def get_resized_frames_per_video(self, dir_name):
        video_jpgs_path = os.path.join(self.root_dir, dir_name)
        # get the random continuous 16 frame
        data = pd.read_csv(os.path.join(video_jpgs_path, 'n_frames'), delimiter=' ', header=None)
        frame_count = data[0][0]

        if frame_count < self.clip_len:
            return None
        
        video_x = np.empty((self.clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        #image_start = random.randint(1, frame_count - self.clip_len)
        image_start = 1
        for i in range(self.clip_len):
            s = "%05d" % (i + image_start)
            image_name = 'image_' + s + '.jpg'
            image_path = os.path.join(video_jpgs_path, image_name)
            tmp_image = cv2.imread(image_path)
            tmp_image = cv2.resize(tmp_image, (self.resize_width, self.resize_height))
            tmp_image = np.array(tmp_image).astype(np.float32)
            tmp_image = tmp_image[:, :, ::-1]    # BGR -> RGB
            video_x[i, :, :, :] = tmp_image

        return video_x
    
def cache_invalid_videos(root_dir, required_size=(320, 180)):
    invalid_videos = set()
    for video_id in os.listdir(root_dir):
        video_path = os.path.join(root_dir, video_id)
        if os.path.isdir(video_path):
            for fname in os.listdir(video_path):
                if fname.endswith(".jpg"):
                    frame_path = os.path.join(video_path, fname)
                    with Image.open(frame_path) as img:
                        if img.size != required_size:
                            invalid_videos.add(video_id)
                            break
    return invalid_videos

class CharadesRGBDataset(Dataset):
    def __init__(self, root_dir, transform=None, clip_len=50, fps=24, split="train", split_ratio=0.8, seed=42):
        self.root_dir = root_dir
        self.transform = transform
        self.clip_len = clip_len
        self.fps = fps

        # Cache invalid videos
        self.invalid_videos = cache_invalid_videos(root_dir)

        # Get all valid video directories
        all_videos = [os.path.join(root_dir, video_id) for video_id in os.listdir(root_dir)
                      if os.path.isdir(os.path.join(root_dir, video_id)) and video_id not in self.invalid_videos]

        # Split the dataset
        random.seed(seed)
        random.shuffle(all_videos)
        split_idx = int(len(all_videos) * split_ratio)

        if split == "train":
            self.videos = all_videos[:split_idx]
        elif split == "val":
            self.videos = all_videos[split_idx:]
        else:
            raise ValueError("Invalid split value. Must be 'train' or 'val'.")

    def _get_videos(self):
        """
        Collects all video directories in the root directory.
        """
        videos = []
        for video_id in os.listdir(self.root_dir):
            video_path = os.path.join(self.root_dir, video_id)
            if os.path.isdir(video_path):
                videos.append(video_path)
        return videos

    def _get_frame_paths(self, video_path):
        """
        Collects all frame paths for a given video directory.
        """
        frame_paths = sorted([
            os.path.join(video_path, fname)
            for fname in os.listdir(video_path)
            if fname.endswith(".jpg")
        ])
        return frame_paths

    def __len__(self):
        """
        Total number of videos in the dataset.
        """
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        video_id = os.path.basename(video_path)

        # Get all frame paths for the video
        frame_paths = self._get_frame_paths(video_path)
        num_frames = len(frame_paths)

        # Ensure there are enough frames
        if num_frames < self.clip_len:
            print(f"[Warning] Video {video_id} has fewer than {self.clip_len} frames. Skipping...")
            return self.__getitem__((idx + 1) % len(self.videos))  # Skip to the next video

        # Randomly choose the starting frame
        start_frame = random.randint(0, num_frames - self.clip_len)
        selected_frame_paths = frame_paths[start_frame:start_frame + self.clip_len]

        # Load all frames for the video
        frames = [Image.open(frame_path).convert("RGB") for frame_path in selected_frame_paths]

        # Pass frames as a list to the transformation pipeline
        if self.transform:
            try:
                frames = self.transform(frames)  # Pass the list of frames
            except TypeError:  # If the transform expects individual frames
                frames = [self.transform(img) for img in frames]

        # Convert the transformed frames to a tensor
        frames = torch.stack(frames)  # Shape: (clip_len, 3, H, W)
        return frames, video_id

"""if __name__ == '__main__':
    # usage
    root_list = '/home/ran/mnt1/Dataset/hmdb51_n_frames/'
    info_list = '/home/ran/mnt1/Dataset/hmdb51_TrainTestlist/hmdb51_test.txt'

    #trainUCF101 = HMDBDataset(root_list, info_list,
                              )
    testUCF101 = HMDBDataset(root_list, info_list,
                              transform=transforms.Compose(
                                  [ClipSubstractMean(),
                                   CenterCrop(),
                                   ToTensor()]))

    dataloader = DataLoader(trainUCF101, batch_size=8, shuffle=True, num_workers=0)
    for i_batch, (images, targets) in enumerate(dataloader):
        print(i_batch, images.size(), targets.size())"""