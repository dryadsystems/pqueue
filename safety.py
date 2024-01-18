# In part from the stablediffusion txt2img.py
# Copyright (c) 2022 Robin Rombach and Patrick Esser and contributors
# Licensed under CreativeML Open RAIL-M
import pathlib
from typing import List
import numpy as np
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from PIL import Image
from transformers.models.auto.feature_extraction_auto import AutoFeatureExtractor


def toarray(img: Image.Image) -> np.ndarray:
    y = img.convert("RGB")
    y = np.array(y) / 255.0
    return y


# load safety model
safety_model_id = pathlib.Path("/app/safety")
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(
    safety_model_id / "feature_extractor"
)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(
    safety_model_id / "safety_checker"
)


def check_nsfw(imgs: List[Image.Image]) -> List[bool]:
    if isinstance(safety_checker, StableDiffusionSafetyChecker):
        safety_checker_input = safety_feature_extractor(imgs, return_tensors="pt")
        _, has_nsfw_concept = safety_checker(
            images=[toarray(img) for img in imgs],
            clip_input=safety_checker_input.pixel_values,
        )
        return has_nsfw_concept
    raise Exception("failed to load safety checker")
