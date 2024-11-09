import os
import pandas as pd
import cv2

def get_video_length(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file does not exist {video_path}")
        return 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length

def process_train_file(train_file, output_csv, base_dir):
    data = []
    with open(train_file, 'r') as f:
        for line in f:
            vpath, label = line.strip().split()
            full_vpath = os.path.join(base_dir, vpath)
            print(f"Processing video: {full_vpath}")  # Debug print
            label = int(label)
            vlen = get_video_length(full_vpath)
            data.append([full_vpath, vlen, label])
    
    df = pd.DataFrame(data, columns=['vpath', 'vlen', 'label'])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, header=False)
    print(f"Generated {output_csv} at {os.path.abspath(output_csv)}")  # Debug print

def process_test_file(test_file, output_csv, base_dir):
    data = []
    with open(test_file, 'r') as f:
        for line in f:
            vpath = line.strip()
            full_vpath = os.path.join(base_dir, vpath)
            print(f"Processing video: {full_vpath}")  # Debug print
            label = vpath.split('/')[0]
            vlen = get_video_length(full_vpath)
            data.append([full_vpath, vlen, label])
    
    df = pd.DataFrame(data, columns=['vpath', 'vlen', 'label'])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, header=False)
    print(f"Generated {output_csv} at {os.path.abspath(output_csv)}")  # Debug print

# Base directory where the video files are located
base_dir = '/home/zanh/DualStreamModel/DPC/process_data/data/ucf101'

# Paths to the input text files and output CSV files
train_files = [
    '/home/zanh/DualStreamModel/DPC/process_data/data/ucf101/trainlist01.txt',
    '/home/zanh/DualStreamModel/DPC/process_data/data/ucf101/trainlist02.txt',
    '/home/zanh/DualStreamModel/DPC/process_data/data/ucf101/trainlist03.txt'
]
test_files = [
    '/home/zanh/DualStreamModel/DPC/process_data/data/ucf101/testlist01.txt',
    '/home/zanh/DualStreamModel/DPC/process_data/data/ucf101/testlist02.txt',
    '/home/zanh/DualStreamModel/DPC/process_data/data/ucf101/testlist03.txt'
]
train_output_csvs = [
    '/home/zanh/DualStreamModel/DPC/process_data/data/ucf101/train_split01.csv',
    '/home/zanh/DualStreamModel/DPC/process_data/data/ucf101/train_split02.csv',
    '/home/zanh/DualStreamModel/DPC/process_data/data/ucf101/train_split03.csv'
]
test_output_csvs = [
    '/home/zanh/DualStreamModel/DPC/process_data/data/ucf101/test_split01.csv',
    '/home/zanh/DualStreamModel/DPC/process_data/data/ucf101/test_split02.csv',
    '/home/zanh/DualStreamModel/DPC/process_data/data/ucf101/test_split03.csv'
]

# Process the train files
for train_file, train_output_csv in zip(train_files, train_output_csvs):
    if os.path.exists(train_file):
        process_train_file(train_file, train_output_csv, base_dir)
    else:
        print(f"Train file {train_file} does not exist")

# Process the test files
for test_file, test_output_csv in zip(test_files, test_output_csvs):
    if os.path.exists(test_file):
        process_test_file(test_file, test_output_csv, base_dir)
    else:
        print(f"Test file {test_file} does not exist")