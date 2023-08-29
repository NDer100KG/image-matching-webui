import sys
from pathlib import Path
import subprocess
import torch
import logging
import cv2
import numpy as np

from ..utils.base_model import BaseModel

EPS = 1e-6
logger = logging.getLogger(__name__)

class ORB(BaseModel):
    # change to your default configs
    default_conf = {
        "max_keypoints": 2000,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        # Initiate ORB detector
        self.orb = cv2.ORB_create(nfeatures=conf["max_keypoints"])
        logger.info(f"ORB init done.")

    def _forward(self, data):
        # data: dict, keys: 'image'
        # image color mode: RGB
        # image value range in [0, 1]
        image = data["image"]

        image_np = (image.cpu().numpy()[0] * 255).astype(np.uint8)
        image_np = np.transpose(image_np, (1, 2, 0))

        # find the keypoints with ORB
        keypoints = self.orb.detect(image_np, None)
        # compute the descriptors with ORB
        keypoints, descriptors = self.orb.compute(image_np, keypoints)

        keypoints_np, scores_np, angles_np = [], [], []

        for _it in range(len(keypoints)):
            keypoints_np.append([keypoints[_it].pt[0], keypoints[_it].pt[1]])
            scores_np.append(keypoints[_it].response)
            angles_np.append(keypoints[_it].angle)

        keypoints = torch.from_numpy(np.array([keypoints_np]))
        scores = torch.from_numpy(np.array([scores_np]))
        angles = torch.from_numpy(np.array([angles_np]))
        descriptors = torch.from_numpy(np.array([descriptors]).transpose(0, 2, 1))

        return {
            "keypoints": keypoints,
            "scores": scores,
            "angles": angles, 
            "descriptors": descriptors,
        }
