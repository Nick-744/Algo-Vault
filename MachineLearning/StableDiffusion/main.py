from my_settings import (
    model_file_name, prompt, negative_prompt, WIDTH, HEIGHT
)

from stable_diffusion_model import Image2ImageModel
from multiprocessing import (Process, Queue)
from PIL import Image
from time import time
import sys, os

BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

def run_GUI_process(image_queue: Queue) -> None:
    ''' Run the GUI in a separate process '''

    from PyQt5.QtWidgets import QApplication
    from GUI_image_interface import DragAndDropImage
    
    app = QApplication(sys.argv)
    
    def on_image_received(image_path: str) -> None:
        # Put the image path in the queue for the main process
        image_queue.put(image_path)
        app.quit() # Close the GUI app

        return;
    
    window = DragAndDropImage()
    window.image_dropped.connect(on_image_received)
    window.show()
    
    app.exec_()

    return;

def process_image_with_model(model:      Image2ImageModel,
                             image_path: str) -> None:
    ''' Process the image with the ML model '''

    print(f'\n-> Processing image: {image_path}')

    input_image = Image.open(image_path).convert('RGB')
    input_image = input_image.resize((WIDTH, HEIGHT))

    output_image = model.forward(
        prompt      = prompt,
        input_image = input_image,

        strength        = 0.2,
        guidance_scale  = 10,
        inference_steps = 50,

        negative_prompt = negative_prompt
    )

    output_path = os.path.join(
        BASE_DIRECTORY, 'model_outputs', f'output_{int(time())}.png'
    )
    output_image.save(output_path)
    print(f'-> Image saved to {output_path}')

    return;

if __name__ == '__main__':
    # Load the model in the main process
    model_path = os.path.join(BASE_DIRECTORY, model_file_name)
    my_model   = Image2ImageModel(model_path)

    # Directory management
    output_dir = os.path.join(BASE_DIRECTORY, 'model_outputs')
    os.makedirs(output_dir, exist_ok = True)

    while True: # Loop to handle multiple image drops    
        # Create a queue for communication between processes
        image_queue = Queue()
        
        # Start GUI process
        gui_process = Process(
            target = run_GUI_process, args = (image_queue,)
        )
        gui_process.start()
        
        # Wait for an image path from the GUI process
        try:
            image_path = image_queue.get()
            
            # Wait for GUI process to finish
            gui_process.join()
            if gui_process.is_alive():
                gui_process.terminate()
            
            process_image_with_model(my_model, image_path)
            
            try:
                response = input(
                    '\nProcess another image? (y/n): '
                ).strip().lower()
                if (response != 'y') and (response != 'yes'):
                    break;
            except KeyboardInterrupt:
                break;
                
        except KeyboardInterrupt:
            print('\nKeyboardInterrupt detected - Exiting...')
            break;
        except:
            print('Error :(')
            break;
        finally:
            # Clean up GUI process if still running
            if gui_process.is_alive():
                gui_process.terminate()
                gui_process.join()

    print('\n- Application Terminated -')
