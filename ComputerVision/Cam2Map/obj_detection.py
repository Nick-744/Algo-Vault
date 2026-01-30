from obj_classification import VisualObject, setup_model, load_visual_data
from PIL import Image
import numpy as np
import torch
import cv2

def sliding_window_classification(frame:        np.ndarray,
                                  forward_pass: callable,
                                  preprocess:   callable,
                                  win_div:      int = 4,
                                  step_div:     int = 3) -> list:
    ''' Scan frame with sliding window and classify each patch. '''
    
    results = []
    (h, w)  = frame.shape[:2]

    # Window/Patch size & step
    (win_w, win_h) = (w // win_div, h // win_div)
    step           = win_w // step_div

    # Convert once per frame...
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    for y in range(0, h - win_h + 1, step):
        for x in range(0, w - win_w + 1, step):
            pil_crop  = pil_frame.crop((x, y, x + win_w, y + win_h))
            embedding = forward_pass(preprocess(pil_crop))

            (best_object, similarity_score) = VisualObject.find_best_match(
                torch.nn.functional.normalize(embedding)
            )

            if best_object:
                results.append(((x, y, x + win_w, y + win_h), best_object.name, similarity_score))

    return results;

def main():
    (forward_pass, preprocess) = setup_model()
    load_visual_data(forward_pass, preprocess)

    cam = cv2.VideoCapture(0)
    while True:
        (ret, frame) = cam.read()
        if not ret:
            break;

        results = sliding_window_classification(frame, forward_pass, preprocess)
        # Draw one big rectangle that encloses all detections
        if results:
            xs1 = [bbox[0] for (bbox, _, _) in results]
            ys1 = [bbox[1] for (bbox, _, _) in results]
            xs2 = [bbox[2] for (bbox, _, _) in results]
            ys2 = [bbox[3] for (bbox, _, _) in results]
            (x1, y1, x2, y2) = (min(xs1), min(ys1), max(xs2), max(ys2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Best detection (name + similarity)
            best_idx  = max(range(len(results)), key = lambda i: results[i][2])
            best_name = results[best_idx][1]
            best_sim  = results[best_idx][2]
            label     = f'{best_name}: {best_sim:.2f}'
            cv2.putText(
                frame, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) == 27: # ESC key
            break;

        # Stop if the window is closed
        if cv2.getWindowProperty('Object Detection', cv2.WND_PROP_VISIBLE) < 1:
            break;

    cam.release()
    cv2.destroyAllWindows()

    return;

if __name__ == '__main__':
    main()
