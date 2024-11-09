import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist

# Create directories
os.makedirs('mnist_videos', exist_ok=True)
os.makedirs('moving_dots_videos', exist_ok=True)

# Function to save video
def save_video(frames, filename):
    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

"""# Generate MNIST video dataset
def generate_mnist_videos():
    (x_train, y_train), _ = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
    x_train = np.repeat(x_train, 3, axis=-1)  # Depth dimension
    x_train = np.array([cv2.resize(img, (128, 128)) for img in x_train])  # Resize to 128x128

    labels = []
    for i, (img, label) in enumerate(zip(x_train, y_train)):
        frames = [img] * 20  # Time dimension
        video_filename = f'mnist_videos/mnist_video_{i}.mp4'
        save_video(frames, video_filename)
        labels.append({'filename': video_filename, 'label': label})

    pd.DataFrame(labels).to_csv('mnist_videos/labels.csv', index=False)"""

# Generate moving dots video dataset
# Generate moving dots video dataset
def generate_moving_dots_videos():
    directions = [
        'up', 'down', 'left', 'right', 'clockwise', 'counterclockwise', 'stationary',
        'zigzag', 'random', 'diagonal_up_right', 'diagonal_up_left', 'diagonal_down_right', 
        'diagonal_down_left', 'spiral_in', 'spiral_out', 'bounce', 'wave_horizontal', 
        'wave_vertical', 'ellipse', 'figure_eight', 'random_walk', 'square', 
        'triangle', 'pentagon', 'hexagon', 'star'
    ]
    labels = []

    for i, direction in enumerate(directions):
        for j in range(200):
            frames = np.zeros((20, 128, 128, 3), dtype=np.uint8)
            num_dots = np.random.randint(1, 5)  # Random number of dots between 1 and 5
            dots = [(np.random.randint(10, 118), np.random.randint(10, 118)) for _ in range(num_dots)]  # Random starting points

            for t in range(20):
                frame = np.zeros((128, 128, 3), dtype=np.uint8)
                new_dots = []
                for (start_x, start_y) in dots:
                    x, y = start_x, start_y  # Initialize x and y with start positions
                    if direction == 'up':
                        y = max(10, start_y - t * 2)
                    elif direction == 'down':
                        y = min(118, start_y + t * 2)
                    elif direction == 'left':
                        x = max(10, start_x - t * 2)
                    elif direction == 'right':
                        x = min(118, start_x + t * 2)
                    elif direction == 'clockwise':
                        angle = t * 18
                        x = int(start_x + 10 * np.cos(np.radians(angle)))
                        y = int(start_y + 10 * np.sin(np.radians(angle)))
                    elif direction == 'counterclockwise':
                        angle = t * -18
                        x = int(start_x + 10 * np.cos(np.radians(angle)))
                        y = int(start_y + 10 * np.sin(np.radians(angle)))
                    elif direction == 'stationary':
                        pass  # x and y remain the same
                    elif direction == 'zigzag':
                        if t % 2 == 0:
                            x = min(118, start_x + t * 1)
                        else:
                            x = max(10, start_x - t * 1)
                        y = min(118, start_y + t * 1)
                    elif direction == 'random':
                        x = np.random.randint(10, 118)
                        y = np.random.randint(10, 118)
                    elif direction == 'diagonal_up_right':
                        x = min(118, start_x + t * 1)
                        y = max(10, start_y - t * 1)
                    elif direction == 'diagonal_up_left':
                        x = max(10, start_x - t * 1)
                        y = max(10, start_y - t * 1)
                    elif direction == 'diagonal_down_right':
                        x = min(118, start_x + t * 1)
                        y = min(118, start_y + t * 1)
                    elif direction == 'diagonal_down_left':
                        x = max(10, start_x - t * 1)
                        y = min(118, start_y + t * 1)
                    elif direction == 'spiral_in':
                        angle = t * 18
                        radius = 10 - t * 0.5
                        x = int(start_x + radius * np.cos(np.radians(angle)))
                        y = int(start_y + radius * np.sin(np.radians(angle)))
                    elif direction == 'spiral_out':
                        angle = t * 18
                        radius = t * 0.5
                        x = int(start_x + radius * np.cos(np.radians(angle)))
                        y = int(start_y + radius * np.sin(np.radians(angle)))
                    elif direction == 'bounce':
                        if t % 2 == 0:
                            y = max(10, start_y - t * 1)
                        else:
                            y = min(118, start_y + t * 1)
                    elif direction == 'wave_horizontal':
                        y = int(start_y + 5 * np.sin(np.radians(t * 36)))
                    elif direction == 'wave_vertical':
                        x = int(start_x + 5 * np.sin(np.radians(t * 36)))
                    elif direction == 'ellipse':
                        angle = t * 18
                        x = int(start_x + 20 * np.cos(np.radians(angle)))
                        y = int(start_y + 10 * np.sin(np.radians(angle)))
                    elif direction == 'figure_eight':
                        angle = t * 18
                        x = int(start_x + 10 * np.cos(np.radians(angle)))
                        y = int(start_y + 10 * np.sin(np.radians(2 * angle)))
                    elif direction == 'random_walk':
                        x = np.clip(start_x + np.random.randint(-2, 3), 10, 118)
                        y = np.clip(start_y + np.random.randint(-2, 3), 10, 118)
                    elif direction == 'square':
                        if t % 4 == 0:
                            x = min(118, start_x + t * 1)
                        elif t % 4 == 1:
                            y = min(118, start_y + t * 1)
                        elif t % 4 == 2:
                            x = max(10, start_x - t * 1)
                        else:
                            y = max(10, start_y - t * 1)
                    elif direction == 'triangle':
                        if t % 3 == 0:
                            x = min(118, start_x + t * 1)
                            y = max(10, start_y - t * 1)
                        elif t % 3 == 1:
                            x = max(10, start_x - t * 1)
                            y = min(118, start_y + t * 1)
                        else:
                            x = min(118, start_x + t * 1)
                            y = min(118, start_y + t * 1)
                    elif direction == 'pentagon':
                        angle = t * 72
                        x = int(start_x + 10 * np.cos(np.radians(angle)))
                        y = int(start_y + 10 * np.sin(np.radians(angle)))
                    elif direction == 'hexagon':
                        angle = t * 60
                        x = int(start_x + 10 * np.cos(np.radians(angle)))
                        y = int(start_y + 10 * np.sin(np.radians(angle)))
                    elif direction == 'star':
                        angle = t * 144
                        x = int(start_x + 10 * np.cos(np.radians(angle)))
                        y = int(start_y + 10 * np.sin(np.radians(angle)))

                    # Ensure dots stay within bounds
                    x = np.clip(x, 10, 118)
                    y = np.clip(y, 10, 118)
                    cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)  # Smaller dot size
                    new_dots.append((x, y))  # Update new_dots with new positions
                dots = new_dots  # Update dots with new positions

                frames[t] = frame

            video_filename = f'moving_dots_videos/moving_dots_{direction}_{j}.mp4'
            save_video(frames, video_filename)
            labels.append({'filename': video_filename, 'label': direction})

    pd.DataFrame(labels).to_csv('moving_dots_videos/labels.csv', index=False)

# Generate datasets
#generate_mnist_videos()
generate_moving_dots_videos()