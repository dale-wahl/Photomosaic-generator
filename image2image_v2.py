import argparse
import cv2
import numpy as np
import glob
from itertools import product
from PIL import Image

box = 5

def get_args():
    parser = argparse.ArgumentParser("Wahl Analytic Photo Collage Maker")
    parser.add_argument("--input", type=str, default="data/input.jpg", help="Path to input image")
    parser.add_argument("--output", type=str, default="data/output.jpg", help="Path to output image")
    parser.add_argument("--pool", type=str, default="image_pool", help="Path to directory containing component images")
    parser.add_argument("--stride", type=int, default=30, help="Size of each component image")
    parser.add_argument("--overlay", type=float, default=0, help="Add transparent overlay of main image; 0 does not add any overlay")
    parser.add_argument("--scale", type=float, default=1.0, help="Scaling factor for thumbnail and output size")
    parser.add_argument("--grid_size", type=int, default=5, help="Base grid size factor for dividing images")
    args = parser.parse_args()
    return args


def get_component_images(path, size=30, grid_size=5):
    images = []
    avg_colors = []

    for image_path in glob.glob("{}/*.png".format(path)) + glob.glob("{}/*.jpg".format(path)):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Crop the image to make it square
        h, w, _ = image.shape
        if h > w:
            crop_start = (h - w) // 2
            image = image[crop_start:crop_start + w, :]
        elif w > h:
            crop_start = (w - h) // 2
            image = image[:, crop_start:crop_start + h]

        # Resize the cropped square image to the target size
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
        images.append(image)

        # Divide the image into a grid and calculate average colors
        block_size = size // grid_size  # Use the provided grid_size directly
        avg_color_grid = []
        for i in range(grid_size):
            for j in range(grid_size):
                block = image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                avg_color_grid.append(np.mean(block, axis=(0, 1)))
        avg_colors.append(avg_color_grid)

    return images, np.array(avg_colors)

def add_overlay(main_image, overlay_image, output_image="final.png", blend=0.5):
    background = Image.open(main_image)
    overlay = Image.open(overlay_image)

    # Ensure the overlay matches the background dimensions
    if overlay.size != background.size:
        overlay = overlay.resize(background.size, Image.Resampling.LANCZOS)

    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")

    # Blend the images
    new_img = Image.blend(background, overlay, blend)

    # Save the final image
    if not output_image.endswith(".png"):
        raise ValueError("Currently only PNG supported")
    new_img.save(output_image, "PNG")


def main(opt):
    # Apply scaling factor to stride and blank image dimensions
    scaled_stride = int(opt.stride * opt.scale)  # Use scaled stride for all calculations
    input_image = cv2.imread(opt.input, cv2.IMREAD_COLOR)
    height, width, num_channels = input_image.shape
    scaled_height, scaled_width = int(height * opt.scale), int(width * opt.scale)
    blank_image = np.zeros((scaled_height, scaled_width, 3), np.uint8)

    # Calculate max grid size compatible with user-provided grid size and stride
    x_by_x_grid = max(i for i in range(1, opt.grid_size + 1) if opt.stride % i == 0)

    # Debugging: Print dimensions
    print(f"Original dimensions: {height}x{width}")
    print(f"Scaled dimensions: {scaled_height}x{scaled_width}")
    print(f"Stride: {opt.stride}")
    print(f"Scaled stride: {scaled_stride}")
    print(f"Grid size: {x_by_x_grid}x{x_by_x_grid}")

    # Get component images with scaled size and consistent grid size
    images, avg_colors = get_component_images(opt.pool, scaled_stride, grid_size=x_by_x_grid)

    used_images = []
    for i, j in product(range(int(width / opt.stride)), range(int(height / opt.stride))):
        # Calculate slice indices using scaled stride
        start_y = int(j * opt.stride)
        end_y = int((j + 1) * opt.stride)
        start_x = int(i * opt.stride)
        end_x = int((i + 1) * opt.stride)

        # Ensure slice indices are within bounds
        if end_y > height or end_x > width:
            print(f"Skipping slice ({start_x}, {start_y}, {end_x}, {end_y}) - out of bounds")
            continue

        partial_input_image = input_image[start_y:end_y, start_x:end_x, :]


        # Divide the partial input image into a grid and calculate average colors
        block_size = max(1, opt.stride // x_by_x_grid)  # Ensure block_size is at least 1
        partial_avg_color_grid = []
        for x in range(x_by_x_grid):
            for y in range(x_by_x_grid):
                block = partial_input_image[x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size]
                if block.size == 0:  # Skip empty slices
                    continue
                partial_avg_color_grid.append(np.mean(block, axis=(0, 1)))

        # Do not reuse image within certain range
        avoid = []
        for blarg in range(box, 0, -1):
            begin = -(blarg * int(height / opt.stride)) - box
            end = -(blarg * int(height / opt.stride)) + box + 1
            avoid += used_images[begin: end]
        avoid += used_images[-box:]

        # Calculate the distance between the gridsz
        distance_matrix = np.array([
            np.linalg.norm(np.array(partial_avg_color_grid) - avg_color_grid)
            for avg_color_grid in avg_colors
        ])

        mask = np.zeros(distance_matrix.size, dtype=bool)
        mask[avoid] = True

        a = np.ma.array(distance_matrix, mask=mask)
        idx = np.argmin(a)

        used_images.append(idx)
        blank_image[j * scaled_stride: (j + 1) * scaled_stride, i * scaled_stride: (i + 1) * scaled_stride, :] = images[idx]

    # Debugging: Confirm temp.jpg dimensions
    print(f"New image dimensions: {blank_image.shape}")

    if type(opt.overlay) == float and (0 < opt.overlay < 1):
        # Save the blank image before applying the overlay
        temp_output = "data/temp/temp.jpg"
        cv2.imwrite(temp_output, blank_image)

        # Add overlay to the blank image
        output_name = ".".join(opt.output.split(".")[:-1]) + ".png"
        add_overlay(main_image=temp_output, overlay_image=opt.input, output_image=output_name, blend=opt.overlay)
    else:
        cv2.imwrite(opt.output, blank_image)


if __name__ == '__main__':
    opt = get_args()
    main(opt)
    print("All done")
