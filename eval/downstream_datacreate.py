import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import imageio

def create_moving_mnist(num_videos=3000, num_frames=100, image_size=64, digit_size=28, num_digits=2):
    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = x_train / 255.0

    videos = np.zeros((num_videos, num_frames, image_size, image_size), dtype=np.float32)
    digit_labels = np.zeros(num_videos, dtype=np.int32)
    motion_labels = np.zeros(num_videos, dtype=np.int32)

    motion_types = [
        'spiral',
        'sine_wave',
        'random_walk',
        'square_wave',
        'rapid_wiggle',
        'hexagon'
    ]

    for i in range(num_videos):
        digit_indices = np.random.choice(len(x_train), num_digits, replace=False)
        digits = [x_train[idx] for idx in digit_indices]
        digit_classes = [y_train[idx] for idx in digit_indices]

        # Combine digit classes into a single label
        digit_label = int(''.join(map(str, sorted(digit_classes))))
        digit_labels[i] = digit_label

        # Assign a random motion type
        motion_type = np.random.choice(len(motion_types))
        motion_labels[i] = motion_type

        start_positions = np.random.randint(0, image_size - digit_size, (num_digits, 2))
        velocities = np.random.uniform(-1, 1, (num_digits, 2))
        times = np.zeros(num_digits)

        for t in range(num_frames):
            for d in range(num_digits):
                start_x, start_y = start_positions[d]
                velocity_x, velocity_y = velocities[d]

                # Apply motion based on the selected motion type
                if motion_type == 0:  # spiral
                    angle = 2 * np.pi * t / num_frames
                    radius = t / num_frames * 10
                    start_x += radius * np.cos(angle)
                    start_y += radius * np.sin(angle)
                elif motion_type == 1:  # sine_wave
                    start_x += velocity_x
                    start_y += np.sin(t * 0.2) * 2
                elif motion_type == 2:  # random_walk
                    start_x += np.random.normal(0, 0.5)
                    start_y += np.random.normal(0, 0.5)
                elif motion_type == 3:  # square_wave
                    start_x += velocity_x
                    start_y += np.sign(np.sin(t * 0.2)) * 2
                elif motion_type == 4:  # rapid_wiggle
                    start_x += np.sin(t * 10) * 2
                    start_y += np.cos(t * 10) * 2
                elif motion_type == 5:  # hexagon
                    angle = (t % 6) * (np.pi / 3)
                    start_x += np.cos(angle) * 2
                    start_y += np.sin(angle) * 2

                # Ensure digits loop back within the frame
                start_x = start_x % image_size
                start_y = start_y % image_size

                start_positions[d] = [start_x, start_y]

                x, y = int(start_x), int(start_y)
                digit = digits[d]

                # Place the digit on the frame
                for dx in range(digit_size):
                    for dy in range(digit_size):
                        new_x = (x + dx) % image_size
                        new_y = (y + dy) % image_size
                        videos[i, t, new_x, new_y] = max(videos[i, t, new_x, new_y], digit[dx, dy])

    return videos, digit_labels, motion_labels, motion_types

def save_moving_mnist_gif(videos, digit_labels, motion_labels, motion_types, output_dir='output_gifs', num_gifs=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_videos, num_frames, image_size, _ = videos.shape

    for i in range(min(num_videos, num_gifs)):
        frames = []
        for t in range(num_frames):
            fig, ax = plt.subplots()
            ax.imshow(videos[i, t], cmap='gray')
            ax.set_title(f"Digit: {digit_labels[i]}, Motion: {motion_types[motion_labels[i]]}")
            ax.axis('off')
            fig.canvas.draw()

            # Convert plot to image
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)
            plt.close(fig)

        # Save as GIF
        imageio.mimsave(os.path.join(output_dir, f'video_{i}_digit_{digit_labels[i]}_motion_{motion_types[motion_labels[i]]}.gif'), frames, fps=10)

# Generate the dataset
videos, digit_labels, motion_labels, motion_types = create_moving_mnist()

# Save the dataset and labels as .npy files
np.save('downstream_mnist_videos.npy', videos)
np.save('downstream_mnist_digit_labels.npy', digit_labels)
np.save('downstream_mnist_motion_labels.npy', motion_labels)

# Save only 5 GIFs
save_moving_mnist_gif(videos, digit_labels, motion_labels, motion_types, num_gifs=20)

print(f"Dataset shape: {videos.shape}")
print(f"Digit labels shape: {digit_labels.shape}")
print(f"Motion labels shape: {motion_labels.shape}")

# Print unique digit labels and their counts
unique_digit_labels, digit_counts = np.unique(digit_labels, return_counts=True)
print("Unique digit labels and their counts:")
for label, count in zip(unique_digit_labels, digit_counts):
    print(f"Digit Label {label}: {count}")

# Print unique motion labels and their counts
unique_motion_labels, motion_counts = np.unique(motion_labels, return_counts=True)
print("\nUnique motion labels and their counts:")
for label, count in zip(unique_motion_labels, motion_counts):
    print(f"Motion Label {motion_types[label]}: {count}")