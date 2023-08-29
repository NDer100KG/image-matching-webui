import torch
import cv2
import numpy as np

from ..utils.base_model import BaseModel

EPS = 1e-6


class Hamming(BaseModel):
    default_conf = {
        "cross_check": True,
    }
    required_inputs = ["descriptors0", "descriptors1"]

    def _init(self, conf):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=self.conf["cross_check"])
        pass

    def _forward(self, data):
        # Only batch 1 is supported
        if data["descriptors0"].size(-1) == 0 or data["descriptors1"].size(-1) == 0:
            matches0 = torch.full(
                data["descriptors0"].shape[:2],
                -1,
                device=data["descriptors0"].device,
            )
            return {
                "matches0": matches0,
                "matching_scores0": torch.zeros_like(matches0),
            }

        try:
            matches = self.bf.match(
                np.transpose(data["descriptors0"].cpu().numpy()[0], (1, 0)),
                np.transpose(data["descriptors1"].cpu().numpy()[0], (1, 0)),
            )
            matches = sorted(matches, key=lambda x: x.distance)
        except:
            matches = []

        if len(matches) > self.conf["max_matching"]:
            matches = matches[: self.conf["max_matching"]]

        matches0 = -1 * np.ones((1, data["descriptors0"].shape[-1])).astype(np.int64)
        scores0 = np.zeros((1, data["descriptors0"].shape[-1]))

        for match in matches:
            matches0[0, match.queryIdx] = match.trainIdx
            scores0[0, match.queryIdx] = EPS / (match.distance + EPS)

        return {
            "matches0": torch.from_numpy(matches0),
            "matching_scores0": torch.from_numpy(scores0),
        }
