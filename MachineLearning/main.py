from my_settings import (
    model_file_name, prompt, negative_prompt, WIDTH, HEIGHT
)

from stable_diffusion_model import Image2ImageModel
from GUI_image_interface import DragAndDropImage
from PyQt5.QtWidgets import QApplication
from PIL import Image
from time import time
import sys, os

BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
GLOBAL_APP     = None

def on_image_dropped(path: str) -> None:
    print(f'\n-> Processing image: {path}')

    input_image = Image.open(path).convert('RGB')
    input_image = input_image.resize((WIDTH, HEIGHT))

    output_image = my_model.forward(
        prompt      = prompt,
        input_image = input_image,

        strength        = 0.2,
        guidance_scale  = 10,
        inference_steps = 50,

        negative_prompt = negative_prompt
    )

    output_dir = os.path.join(BASE_DIRECTORY, 'model_outputs')
    os.makedirs(output_dir, exist_ok = True)

    output_path = os.path.join(
        BASE_DIRECTORY, 'model_outputs', f'output_{int(time())}.png')
    output_image.save(output_path)
    print(f'-> Image saved to {output_path}')

    GLOBAL_APP.quit() # Quit the application after processing...

    return;

if __name__ == '__main__':
    model_path = os.path.join(BASE_DIRECTORY, model_file_name)
    my_model   = Image2ImageModel(model_path)

    GLOBAL_APP = QApplication(sys.argv)

    # Don't quit when last window closes,
    # we need to process the image first!
    GLOBAL_APP.setQuitOnLastWindowClosed(False)

    image_input_window = DragAndDropImage()
    
    # Connect signal!
    image_input_window.image_dropped.connect(on_image_dropped)

    image_input_window.show()
    GLOBAL_APP.exec_()
