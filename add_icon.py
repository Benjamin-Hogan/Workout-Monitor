#!/usr/bin/env python3
import os
import sys
from PIL import Image, ImageDraw


def create_dumbbell_icon(output_path, size=1024):
    """Create a simple dumbbell icon"""
    # Create a new image with a blue background
    icon = Image.new('RGBA', (size, size), (41, 128, 185, 255))
    draw = ImageDraw.Draw(icon)

    # Draw dumbbell
    bar_width = size * 0.1
    bar_length = size * 0.6
    bar_x_start = (size - bar_length) / 2
    bar_y = size / 2

    # Draw the bar
    draw.rectangle(
        [(bar_x_start, bar_y - bar_width/4),
         (bar_x_start + bar_length, bar_y + bar_width/4)],
        fill=(236, 240, 241, 255)  # Light gray
    )

    # Draw the weights
    weight_radius = size * 0.15
    for x in [bar_x_start, bar_x_start + bar_length]:
        draw.ellipse(
            [(x - weight_radius, bar_y - weight_radius),
             (x + weight_radius, bar_y + weight_radius)],
            fill=(52, 73, 94, 255),  # Dark blue-gray
            outline=(236, 240, 241, 255),  # Light gray
            width=5
        )

        # Inner circle on weights
        inner_radius = weight_radius * 0.6
        draw.ellipse(
            [(x - inner_radius, bar_y - inner_radius),
             (x + inner_radius, bar_y + inner_radius)],
            fill=(127, 140, 141, 255),  # Medium gray
        )

    # Save the icon
    icon.save(output_path)
    print(f"Icon saved to {output_path}")


def main():
    # Create the icon
    icon_path = "workout_icon.png"
    create_dumbbell_icon(icon_path)

    print("\nTo set this icon for your RunWorkoutApp.command file:")
    print("1. Right-click on the icon image and select 'Copy'")
    print("2. Right-click on RunWorkoutApp.command and select 'Get Info'")
    print("3. Click on the small icon in the top-left corner of the info window")
    print("4. Press Cmd+V to paste the icon")
    print("5. Close the info window")


if __name__ == "__main__":
    main()
