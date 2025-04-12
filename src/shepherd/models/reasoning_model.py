"""
Segmentation model.
"""

from abc import abstractmethod
from typing import Dict, List

import numpy as np

from .base_model import BaseModel


class ReasoningModel(BaseModel):
    """
    Base class for segmentation models.
    """

    @abstractmethod
    def reason(self):
        """
        Runs inference on VLM based on image and question.
        Returns answer.
        """
