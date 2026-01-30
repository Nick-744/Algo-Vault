from obj_classification import VisualObject, setup_model, load_visual_data
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np
import json
import cv2
import time

MAP_FILE = Path(__file__).parent.joinpath('DATA', 'map.png')
POS_FILE = Path(__file__).parent.joinpath(
    'DATA',
    'visual_beacons_data',
    'beacons_positions.json'
)

def load_positions(path: Path = POS_FILE) -> dict:
    ''' Load beacon positions from JSON. '''

    if not path.exists():
        raise FileNotFoundError(f'Positions file not found: {path}');

    with path.open('r', encoding = 'utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError('Positions file must be a JSON object mapping names to [x, y] arrays');

    positions = {}
    for (name, val) in data.items():
        if not (isinstance(val, list) and len(val) == 2):
            raise ValueError(f'Invalid value for "{name}": expected [x, y] array');
        try:
            x = float(val[0]); y = float(val[1])
        except Exception as e:
            raise ValueError(f'Coordinates for "{name}" must be numbers') from e;
        positions[str(name)] = (x, y)

    return positions;

def make_voronoi(map_img:   np.ndarray,
                 positions: dict,
                 highlight: str = None) -> np.ndarray:
    (h, w) = map_img.shape[:2]
    names  = list(positions.keys())
    pts    = []
    for name in names:
        xy = positions[name]

        x = float(xy[0])
        y = float(xy[1])

        # Clamp coordinates to map boundaries!
        x = min(max(x, 0.), float(w - 1))
        y = min(max(y, 0.), float(h - 1))
        pts.append((x, y))

    # OpenCV’s planar subdivision data structure
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for p in pts:
        subdiv.insert(p)

    (facets, _) = subdiv.getVoronoiFacetList([])
    overlay     = map_img.copy()
    for (i, f) in enumerate(facets): # Drawing timeee!
        poly = np.array(f, np.int32)
        name = names[i] if i < len(names) else f'p{i}'
        if highlight == name:
            # Draw a semi-transparent fill for the highlighted region
            _alpha = 0.3
            _color = (0, 255, 0)
            _tmp   = overlay.copy()
            cv2.fillPoly(_tmp, [poly], _color)
            overlay = cv2.addWeighted(_tmp, _alpha, overlay, 1 - _alpha, 0)
        cv2.polylines(overlay, [poly], True, (0, 0, 0), 1)
        # Draw site point and label (convert to ints for pixel coords)
        (px, py) = (int(pts[i][0]), int(pts[i][1]))
        cv2.circle(overlay, (px, py), 4, (0, 0, 255), -1)
        cv2.putText(overlay, name, (px + 5, py + 5), 1, 1, (0, 0, 0), 1)

    return overlay;

def main():
    (forward_pass, preprocess) = setup_model()
    load_visual_data(forward_pass, preprocess)

    positions = load_positions()
    map_img   = cv2.imread(str(MAP_FILE))
    if map_img is None:
        raise FileNotFoundError(f'Could not load map image: {MAP_FILE}');

    cam               = cv2.VideoCapture(1)
    highlight         = None
    last_highlight_ts = None # Monotonic timestamp of last valid highlight refresh

    # Validate that every known object has a position defined; fail fast if not!
    known_names = {obj.name for obj in VisualObject.objects_list}
    missing     = sorted([name for name in known_names if name not in positions])
    if missing:
        raise ValueError(
            'Missing positions for: ' + ', '.join(missing) + f'. Please add them to {POS_FILE}'
        );
    while True:
        (ret, frame) = cam.read()
        if not ret:
            break;

        pil_frame    = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        emb          = forward_pass(preprocess(pil_frame))
        (match, sim) = VisualObject.find_best_match(F.normalize(emb))
        # Refresh highlight timer on any valid match; persist highlight otherwise...
        if match and (match.name in positions):
            if highlight != match.name:
                highlight = match.name
            # Update refresh time even if the same highlight continues!
            last_highlight_ts = time.monotonic()

        # If no refresh within 5 seconds, clear highlight
        if (highlight is not None) and (last_highlight_ts is not None):
            if (time.monotonic() - last_highlight_ts) > 5.:
                highlight         = None
                last_highlight_ts = None

        # Show recognition status on the camera view...
        if match:
            txt   = f'FOUND: {match.name} ({sim:.2f})'
            color = (0, 255, 0)
        else:
            txt   = 'NO MATCH'
            color = (0, 0, 255)
        cv2.putText(frame, txt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('Camera', frame)

        map_view = make_voronoi(map_img, positions, highlight)
        cv2.imshow('Map', map_view)

        if cv2.waitKey(1) == 27:
            break;
    
        if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1 or \
           cv2.getWindowProperty('Map', cv2.WND_PROP_VISIBLE) < 1:
            break;

    cam.release()
    cv2.destroyAllWindows()

    return;

if __name__ == '__main__':
    main()
