import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import imageio

def create_moving_mnist(num_videos=10000, num_frames=100, image_size=64, digit_size=28, num_digits=2):
    (x_train, _), (_, _) = mnist.load_data()
    x_train = x_train / 255.0

    videos = np.zeros((num_videos, num_frames, image_size, image_size), dtype=np.float32)

    for i in range(num_videos):
        for d in range(num_digits):
            digit = x_train[np.random.randint(0, x_train.shape[0])]
            start_x, start_y = np.random.randint(0, image_size - digit_size, 2)
            velocity_x, velocity_y = np.random.uniform(-1, 1, 2)
            time = 0  # Initialize time for non-linear movement

            for t in range(num_frames):
                # Update position with non-linear component
                start_x += velocity_x + 0.5 * np.sin(time)
                start_y += velocity_y + 0.5 * np.cos(time)
                time += 0.1  # Increment time

                # Bounce off the edges
                if start_x < 0 or start_x + digit_size > image_size:
                    velocity_x = -velocity_x
                    start_x = max(0, min(start_x, image_size - digit_size))
                if start_y < 0 or start_y + digit_size > image_size:
                    velocity_y = -velocity_y
                    start_y = max(0, min(start_y, image_size - digit_size))

                x = int(start_x)
                y = int(start_y)

                # Handle wrapping around the edges
                for dx in range(digit_size):
                    for dy in range(digit_size):
                        new_x = (x + dx) % image_size
                        new_y = (y + dy) % image_size
                        videos[i, t, new_x, new_y] = max(videos[i, t, new_x, new_y], digit[dx, dy])

    return videos

def save_moving_mnist_gif(videos, output_dir='output_gifs', num_gifs=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_videos, num_frames, image_size, _ = videos.shape

    for i in range(min(num_videos, num_gifs)):
        frames = []
        for t in range(num_frames):
            fig, ax = plt.subplots()
            ax.imshow(videos[i, t], cmap='gray')
            ax.axis('off')
            fig.canvas.draw()

            # Convert plot to image
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)
            plt.close(fig)

        # Save as GIF
        imageio.mimsave(os.path.join(output_dir, f'video_{i}.gif'), frames, fps=10)

# Generate the dataset
videos = create_moving_mnist()

# Save the dataset as a .npy file
np.save('moving_mnist_videos.npy', videos)

# Save only 5 GIFs
save_moving_mnist_gif(videos, num_gifs=5)