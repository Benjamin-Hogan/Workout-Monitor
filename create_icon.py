#!/usr/bin/env python3
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Create directory for icons
if not os.path.exists('icons'):
    os.makedirs('icons')

# Function to create a circular icon with gradient and dumbbell


def create_workout_icon(size=1024):
    # Create a transparent background
    icon = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(icon)

    # Create a circular gradient background
    for i in range(size):
        for j in range(size):
            # Calculate distance from center
            distance = np.sqrt((i - size/2)**2 + (j - size/2)**2)
            if distance < size/2:  # Only inside the circle
                # Create a blue to purple gradient
                ratio = distance / (size/2)
                r = int(41 + ratio * (125 - 41))
                g = int(128 + ratio * (116 - 128))
                b = int(185 + ratio * (196 - 185))
                draw.point((i, j), fill=(r, g, b, 255))

    # Draw dumbbell
    bar_width = size * 0.1
    bar_length = size * 0.6
    bar_x_start = (size - bar_length) / 2
    bar_y = size / 2

    # Draw the bar
    draw.rectangle(
        [(bar_x_start, bar_y - bar_width/4),
         (bar_x_start + bar_length, bar_y + bar_width/4)],
        fill=(220, 220, 220, 255)
    )

    # Draw the weights
    weight_radius = size * 0.15
    for x in [bar_x_start, bar_x_start + bar_length]:
        draw.ellipse(
            [(x - weight_radius, bar_y - weight_radius),
             (x + weight_radius, bar_y + weight_radius)],
            fill=(50, 50, 50, 255),
            outline=(80, 80, 80, 255),
            width=5
        )

        # Inner circle on weights
        inner_radius = weight_radius * 0.6
        draw.ellipse(
            [(x - inner_radius, bar_y - inner_radius),
             (x + inner_radius, bar_y + inner_radius)],
            fill=(70, 70, 70, 255),
            outline=(100, 100, 100, 255),
            width=3
        )

    # Save in different formats and sizes
    icon.save('icons/workout_icon.png')

    # Create .icns format for macOS (requires png2icns or iconutil)
    icon_16 = icon.resize((16, 16), Image.LANCZOS)
    icon_32 = icon.resize((32, 32), Image.LANCZOS)
    icon_64 = icon.resize((64, 64), Image.LANCZOS)
    icon_128 = icon.resize((128, 128), Image.LANCZOS)
    icon_256 = icon.resize((256, 256), Image.LANCZOS)
    icon_512 = icon.resize((512, 512), Image.LANCZOS)

    # Save all sizes
    icon_16.save('icons/icon_16x16.png')
    icon_32.save('icons/icon_32x32.png')
    icon_64.save('icons/icon_64x64.png')
    icon_128.save('icons/icon_128x128.png')
    icon_256.save('icons/icon_256x256.png')
    icon_512.save('icons/icon_512x512.png')
    icon.save('icons/icon_1024x1024.png')

    print("Icons created successfully in the 'icons' directory")

    # Create iconset directory structure for macOS
    iconset_dir = 'icons/workout.iconset'
    if not os.path.exists(iconset_dir):
        os.makedirs(iconset_dir)

    # Save icons in iconset format
    icon_16.save(f'{iconset_dir}/icon_16x16.png')
    icon_32.save(f'{iconset_dir}/icon_16x16@2x.png')
    icon_32.save(f'{iconset_dir}/icon_32x32.png')
    icon_64.save(f'{iconset_dir}/icon_32x32@2x.png')
    icon_128.save(f'{iconset_dir}/icon_128x128.png')
    icon_256.save(f'{iconset_dir}/icon_128x128@2x.png')
    icon_256.save(f'{iconset_dir}/icon_256x256.png')
    icon_512.save(f'{iconset_dir}/icon_256x256@2x.png')
    icon_512.save(f'{iconset_dir}/icon_512x512.png')
    icon.save(f'{iconset_dir}/icon_512x512@2x.png')

    print("Iconset created successfully. Run the following command to create the .icns file:")
    print("iconutil -c icns icons/workout.iconset -o app_icon.icns")


if __name__ == "__main__":
    create_workout_icon()
    print("Icon generation complete!")
