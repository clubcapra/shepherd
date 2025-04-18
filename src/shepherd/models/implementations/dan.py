"""
DAN model implementation.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from ..depth_model import DepthModel

import gc

class DAN(DepthModel):
    """
    DAN model implementation.
    """

    def load_model(self):
        """Load Depth Anything model."""
        self.model = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=self.device,
        )

    def unload_model(self):
        """Unload Depth Anything model to free up memory."""
        if hasattr(self, 'model'):
            # Since model is a pipeline, check if model exists and has a model attribute
            if self.model is not None and hasattr(self.model, 'model'):
                if next(self.model.model.parameters()).is_cuda:
                    del self.model.model
            self.model = None
        # Force garbage collection just for the deleted objects
        gc.collect()

    def preprocess(self, image: np.ndarray) -> Image.Image:
        """Preprocess image for depth estimation."""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        return Image.fromarray(image)

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth from image."""
        try:
            # Preprocess image
            pil_image = self.preprocess(image)

            # Get depth estimation
            with torch.no_grad():
                depth = self.model(pil_image)["depth"]

            return self.postprocess(depth)
        except (ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"Error estimating depth: {str(e)}")
            return np.zeros(image.shape[:2])

    def postprocess(self, output: Image.Image) -> np.ndarray:
        """Convert depth output to normalized numpy array."""
        # Convert PIL Image to numpy array
        depth_np = np.array(output)

        # Normalize depth values to 0-1 range
        depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())

        return depth_np
