# Stable Diffusion Image-to-Image Model Wrapper.

from my_settings import (
    model_file_name, prompt, negative_prompt,
    WIDTH, HEIGHT
)

from diffusers import (
    StableDiffusionXLPipeline, DPMSolverMultistepScheduler
)
from time import time
from PIL import Image
import torch
import os

# Dummy safety checker to bypass filtering!
class _DummySafetyChecker:
    def __call__(self, images, **kwargs) -> tuple:
        return (images, [False] * len(images));

class Image2ImageModel:
    def __init__(self,
                 model_path: str = None,
                 model_id:   str = None) -> None:
        # Check if CUDA is available
        self.availableDevice = torch.cuda.is_available()

        print('Loading model...')
        if not model_id:
            self.model = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype     = torch.float16 if self.availableDevice \
                                                else torch.float32,
                use_safetensors = True
            )
        elif not model_path:
            self.model = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype     = torch.float16 if self.availableDevice \
                                                else torch.float32,
                use_safetensors = True
            )
        else:
            raise ValueError(
                'Either model_path or model_id must be provided!'
            );

        self.model.scheduler = DPMSolverMultistepScheduler.from_config(
            self.model.scheduler.config
        )

        # Disable safety checker...
        self.model.safety_checker = _DummySafetyChecker()

        self.device = 'cuda' if self.availableDevice else 'cpu'
        self.model  = self.model.to(self.device)
        print(f'-> Using device: {self.device}')

        return;

    def forward(self,
                prompt:          str,
                input_image:     Image.Image = None,

                strength:        float = 0.3,
                guidance_scale:  float = 10,
                inference_steps: int   = 50,

                negative_prompt: str = None) -> Image.Image:
        print('\nGenerating image...')

        if input_image is not None:
            result = self.model(
                prompt = prompt,
                image  = input_image,

                strength            = strength,
                guidance_scale      = guidance_scale,
                num_inference_steps = inference_steps,
                width = WIDTH, height = HEIGHT,

                negative_prompt = negative_prompt
            )
        else:
            result = self.model(
                prompt = prompt,

                strength            = strength,
                guidance_scale      = guidance_scale,
                num_inference_steps = inference_steps,

                negative_prompt = negative_prompt
            )

        output_image = result.images[0]
        print('-> Image generated successfully!')

        return output_image;

if __name__ == '__main__':
    BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
    model_path     = os.path.join(BASE_DIRECTORY, model_file_name)
    my_model       = Image2ImageModel(model_path)

    # Directory management
    output_dir = os.path.join(BASE_DIRECTORY, 'model_outputs')
    os.makedirs(output_dir, exist_ok = True)

    while True:
        output_image = my_model.forward(
            prompt = prompt,

            strength        = 0.85,
            guidance_scale  = 13.0,
            inference_steps = 50,

            negative_prompt = negative_prompt
        )

        output_path = os.path.join(
            BASE_DIRECTORY, 'model_outputs', f'output_{int(time())}.png'
        )
        output_image.save(output_path)
        print(f'-> Image saved to {output_path}')
