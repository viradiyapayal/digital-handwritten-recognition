import numpy as np
from PIL import Image

def preprocess_for_model(img: Image.Image) -> np.ndarray:
    """Resize, normalize, and add batch dimension for MobileNetV2."""
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    return img_array