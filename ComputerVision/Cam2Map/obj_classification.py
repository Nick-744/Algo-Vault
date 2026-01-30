from torchvision.models import (
    ResNet50_Weights, ResNet101_Weights, MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights
)
import torchvision.models as models
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torch
import cv2

# --- Core Idea --- #
# A model's weights are well-suited for extracting useful features from images.

def setup_model() -> tuple:
    ' See: https://docs.pytorch.org/vision/main/models.html '

    # --- Model Setup --- #
    weights = MobileNet_V3_Small_Weights.DEFAULT # Best available weights...
    model   = models.mobilenet_v3_small(weights = weights)

    # Remove the final classification layer!
    # fc -> ResNet & classifier[-1] -> MobileNet...
    if isinstance(model, models.ResNet):
        model.fc             = torch.nn.Identity()
    elif isinstance(model, models.MobileNetV3):
        model.classifier[-1] = torch.nn.Identity()
    # Use the model as a feature extractor (i.e., to get embeddings instead of class predictions),
    # by replacing the final fully connected layer with torch.nn.Identity().
    # This special layer simply returns its input unchanged, effectively removing the classification
    # step and allowing access the high-level features produced by the network.

    model.eval()
    # ***IMPORTANT*** - Certain layers, like dropout, behave differently during training and evaluation.
    # Setting the model to eval mode ensures consistent and correct behavior
    # when using the model for inference or feature extraction!

    # Move model to the GPU if available...
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    @torch.inference_mode() # Disable autograd - stricter and faster than torch.no_grad()
    def forward_pass(input_tensor: torch.Tensor) -> torch.Tensor:
        return model(input_tensor.to(device));

    # --- Image Transformation / Preprocessing --- #
    preprocess = weights.transforms()

    def preprocess_image(image: Image) -> torch.Tensor:
        return preprocess(image).unsqueeze(0);

    return (forward_pass, preprocess_image);

class VisualObject:
    objects_list = [] # Class variable to store all objects

    def __init__(self, name: str, embedding: torch.Tensor):
        self.name      = name
        self.embedding = F.normalize(embedding) # Normalize for faster similarity search!

        VisualObject.objects_list.append(self)

        return;

    def similarity(self, comparison_embedding: torch.Tensor) -> float:
        ' See: https://en.wikipedia.org/wiki/Cosine_similarity '

        # Cosine similarity:             cos(x, y) = (x · y) / (‖x‖₂ ‖y‖₂)
        # With L2-normalized embeddings: cos(x, y) = x · y = torch.mm(x, y.T)
        return torch.mm(self.embedding, comparison_embedding.T).item();

        # Alternative (normalizes on each call):
        # return F.cosine_similarity(self.embedding, comparison_embedding).item();

    @classmethod
    def find_best_match(cls, # Refers to the VisualObject class itself!
                        target_embedding: torch.Tensor,
                        threshold:        float = 0.5) -> tuple:
        ''' Find the object with the highest similarity to the target embedding.
        
        Args:
            target_embedding: The embedding to compare against. Remember to normalize it!
            threshold:        Minimum similarity threshold (default: 0.5)

        Returns:
            tuple: (best_object, similarity_score) or (None, 0.0) if no match above threshold '''

        if not cls.objects_list:
            raise ValueError('No objects available for matching...');

        best_object     = None
        best_similarity = 0.

        for obj in cls.objects_list:
            similarity = obj.similarity(target_embedding)
            if (similarity > best_similarity) and (similarity >= threshold):
                best_similarity = similarity
                best_object     = obj

        return (best_object, best_similarity);

def load_visual_data(forward_pass: callable, preprocess: callable) -> None:
    ''' Create VisualObjects from images in the visual_beacons_data directory. '''

    data_path   = Path(__file__).parent.joinpath('DATA', 'visual_beacons_data')
    image_files = list(data_path.glob('*.png')) + list(data_path.glob('*.jpg'))

    for image_path in image_files:
        image     = Image.open(image_path).convert('RGB')
        embedding = forward_pass(preprocess(image))
        VisualObject(image_path.stem, embedding)

    return;

def main():
    (forward_pass, preprocess) = setup_model()
    load_visual_data(forward_pass, preprocess)

    cam = cv2.VideoCapture(0) # Open webcam
    while True:
        (ret, frame) = cam.read()
        if not ret: # return status
            break;

        # OpenCV -> PIL -> PyTorch Tensor -> Embedding
        pil_frame       = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_embedding = forward_pass(preprocess(pil_frame))

        # Find the best matching object from the list
        (best_match, similarity) = VisualObject.find_best_match(F.normalize(frame_embedding))
        if best_match:
            text  = f'FOUND: {best_match.name} ({similarity:.2f})'
            color = (0, 255, 0)
        else:
            text  = f'NO MATCH FOUND'
            color = (0, 0, 255)

        # Result
        cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('Classification', frame)

        # Pauses for 1 millisecond to let OpenCV’s GUI
        # process events and returns an integer keycode...
        key = cv2.waitKey(1)
        if key == 27: # ESC key
            break;
        
        # Stop if the window is closed
        if cv2.getWindowProperty('Classification', cv2.WND_PROP_VISIBLE) < 1:
            break;

    cam.release()
    cv2.destroyAllWindows()

    return;

if __name__ == '__main__':
    main()
