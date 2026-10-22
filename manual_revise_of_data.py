from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
import os

image_folder = "images/train"
label_folder = "labels/train"

image_files = sorted(os.listdir(image_folder))

for image_file in image_files:
    if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    label_file = os.path.splitext(image_file)[0] + ".txt"
    label_path = os.path.join(label_folder, label_file)

    if not os.path.exists(label_path):
        print(f"No label found for {image_file}")
        continue

    image = Image.open(os.path.join(image_folder, image_file))
    img_width, img_height = image.size

    with open(label_path) as file:
        labels = file.readlines()

    if len(labels) != 1:
        print(f"Skipping {image_file}: expected exactly one box")
        continue

    label = labels[0].split()

    class_id = int(label[0])
    x_center = float(label[1])
    y_center = float(label[2])
    width = float(label[3])
    height = float(label[4])

    x1 = (x_center - width / 2) * img_width
    y1 = (y_center - height / 2) * img_height
    x2 = (x_center + width / 2) * img_width
    y2 = (y_center + height / 2) * img_height

    fig, ax = plt.subplots(figsize=(9, 7))
    plt.subplots_adjust(bottom=0.15)

    ax.imshow(image)
    ax.set_title(image_file)

    # Existing box
    original_box = plt.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        fill=False, edgecolor="red", linewidth=2
    )
    ax.add_patch(original_box)

    # Current box coordinates
    current_box = [x1, y1, x2, y2]
    action = ["next"]

    def on_select(eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return

        new_x1 = min(eclick.xdata, erelease.xdata)
        new_y1 = min(eclick.ydata, erelease.ydata)
        new_x2 = max(eclick.xdata, erelease.xdata)
        new_y2 = max(eclick.ydata, erelease.ydata)

        current_box[:] = [new_x1, new_y1, new_x2, new_y2]

        original_box.set_xy((new_x1, new_y1))
        original_box.set_width(new_x2 - new_x1)
        original_box.set_height(new_y2 - new_y1)
        original_box.set_edgecolor("green")

        fig.canvas.draw_idle()

    selector = RectangleSelector(
        ax, on_select,
        useblit=True,
        button=[1],
        interactive=True,
        props=dict(edgecolor="green", fill=False, linewidth=2)
    )

    # Save corrected bounding box to original TXT file
    def save_label(event):
        new_x1, new_y1, new_x2, new_y2 = current_box

        # Keep coordinates inside the image
        new_x1 = max(0, min(img_width, new_x1))
        new_x2 = max(0, min(img_width, new_x2))
        new_y1 = max(0, min(img_height, new_y1))
        new_y2 = max(0, min(img_height, new_y2))

        if new_x2 <= new_x1 or new_y2 <= new_y1:
            print("Invalid box. Draw a new box before saving.")
            return

        # Convert back to normalized YOLO coordinates
        new_center_x = ((new_x1 + new_x2) / 2) / img_width
        new_center_y = ((new_y1 + new_y2) / 2) / img_height
        new_width = (new_x2 - new_x1) / img_width
        new_height = (new_y2 - new_y1) / img_height

        # Overwrite original label file
        with open(label_path, "w") as file:
            file.write(
                f"{class_id} {new_center_x:.8f} "
                f"{new_center_y:.8f} {new_width:.8f} "
                f"{new_height:.8f}\n"
            )

        original_box.set_edgecolor("blue")
        fig.canvas.draw_idle()

        print(f"Saved corrected label: {label_path}")

    # Create Save Label button
    save_ax = fig.add_axes([0.40, 0.03, 0.20, 0.06])
    save_button = Button(save_ax, "Save Label")
    save_button.on_clicked(save_label)

    # Keyboard shortcuts
    def on_key(event):
        if event.key == "s":
            save_label(event)

        elif event.key == "n":
            action[0] = "next"
            plt.close(fig)

        elif event.key == "q":
            action[0] = "quit"
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show(block=True)

    if action[0] == "quit":
        break