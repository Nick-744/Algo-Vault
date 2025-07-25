from GUI_image_interface import DragAndDropImage
from PyQt5.QtWidgets import QApplication
from PIL import Image
from time import time
import os, sys

def resize_and_pad_image(image_path:    str,
                         target_width:  int,
                         target_height: int) -> Image:
    img = Image.open(image_path).convert('RGB') # Load image

    # Calculate aspect ratios
    target_ratio = target_width / target_height
    img_ratio    = img.width / img.height

    # Determine new size
    if img_ratio > target_ratio:
        # Image is wider than target -> limit width, adjust height
        new_width  = target_width
        new_height = round(target_width / img_ratio)
    else:
        # Image is taller or same ratio -> limit height, adjust width
        new_height = target_height
        new_width  = round(target_height * img_ratio)

    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    # Create new image with white background
    new_img = Image.new(
        'RGB', (target_width, target_height), color = (255, 255, 255)
    )

    # Paste resized image centered
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_img.paste(resized_img, (paste_x, paste_y))

    return new_img;

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if __name__ == '__main__':
    app = QApplication(sys.argv)

    main_window = DragAndDropImage()
    
    def on_drop(path: str) -> None:
        try:
            resized_image = resize_and_pad_image(path, 1080, 1350)
            resized_image.save(
                os.path.join(
                    BASE_DIR,
                    'resized_images',
                    f'resized_image_{int(time())}.png')
            )
            print(f'Image saved successfully: {path}')
        except Exception as e:
            print(f'Error processing image: {e}')

        # app.quit();
        return;
    
    main_window.image_dropped.connect(on_drop)
    main_window.show()

    sys.exit(app.exec_())
