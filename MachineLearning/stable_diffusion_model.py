from my_settings import (WIDTH, HEIGHT)

from diffusers import (
    StableDiffusionXLPipeline, DPMSolverMultistepScheduler
)
from PIL import Image
import torch

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
                input_image:     Image.Image,

                strength:        float = 0.3,
                guidance_scale:  float = 10,
                inference_steps: int   = 50,

                negative_prompt: str = None) -> Image.Image:
        print('\nGenerating image...')
        result = self.model(
            prompt = prompt,
            image  = input_image,

            strength            = strength,
            guidance_scale      = guidance_scale,
            num_inference_steps = inference_steps,
            width = WIDTH, height = HEIGHT,

             negative_prompt = negative_prompt
        )

        output_image = result.images[0]
        print('-> Image generated successfully!')

        return output_image;
