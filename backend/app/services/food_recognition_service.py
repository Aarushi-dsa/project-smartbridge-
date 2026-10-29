"""Indian-food image classification service for Food Verse AI.

Uses a real Hugging Face image-classification model instead of ImageNet label
filtering and the old colour-based fallback. The model is downloaded on the
first successful startup and cached by Hugging Face for later runs.
"""
import os
import time
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification

from backend.app.schemas.food_schemas import FoodPredictionItem, FoodRecognitionResponse

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "usda_nutrition_db.json"

# This model explicitly includes masala_dosa, idli and samosa among its 20
# output classes. It is a community model, not a guarantee of perfect results.
MODEL_ID = os.getenv("FOODVERSE_FOOD_MODEL", "Subhash5/indian-food-classifier")
UNCERTAIN_THRESHOLD = float(os.getenv("FOODVERSE_CONFIDENCE_THRESHOLD", "0.40"))


def _display_label(label: str) -> str:
    aliases = {
        "masala_dosa": "Masala Dosa",
        "paani_puri": "Pani Puri",
        "kaathi_rolls": "Kathi Rolls",
        "butter_naan": "Butter Naan",
        "chole_bhature": "Chole Bhature",
        "dal_makhani": "Dal Makhani",
        "pav_bhaji": "Pav Bhaji",
        "kadai_paneer": "Kadai Paneer",
        "fried_rice": "Fried Rice",
    }
    key = label.strip().lower().replace(" ", "_").replace("-", "_")
    return aliases.get(key, key.replace("_", " ").title())


class FoodRecognitionService:
    def __init__(self) -> None:
        self.catalog: dict[str, Any] = {}
        self.processor = None
        self.model = None
        self.id2label: dict[int, str] = {}
        self.model_loaded = False
        self._load_catalog()
        self._init_model()

    def _load_catalog(self) -> None:
        try:
            with DATA_PATH.open("r", encoding="utf-8") as f:
                self.catalog = json.load(f)
        except (OSError, json.JSONDecodeError):
            self.catalog = {}

    def _init_model(self) -> None:
        try:
            print(f"[FoodRecognitionService] Loading Indian-food model: {MODEL_ID}")
            self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
            self.model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
            self.model.to("cpu")
            self.model.eval()
            self.id2label = {
                int(index): str(label)
                for index, label in self.model.config.id2label.items()
            }
            self.model_loaded = True
            print(
                f"[FoodRecognitionService] Model ready; "
                f"{len(self.id2label)} classes loaded."
            )
        except Exception as exc:
            # Do not silently return a made-up food based on image colour.
            self.model_loaded = False
            print(
                "[FoodRecognitionService] Could not load the Indian-food model. "
                "Check internet access and install backend requirements. "
                f"Details: {exc}"
            )

    def _metadata(self, label: str) -> tuple[str, str, str]:
        normalized = label.lower().replace(" ", "_").replace("-", "_")
        item = self.catalog.get(normalized, {})
        category = item.get("category", "Indian Food")
        serving = item.get("serving_description", "1 serving (estimate; confirm portion size)")
        notes = item.get(
            "health_notes",
            "Food category predicted from the uploaded image. Confirm the result before using it for nutrition tracking.",
        )
        return category, serving, notes

    def analyze_image(self, image_path: Path) -> FoodRecognitionResponse:
        start_time = time.time()
        try:
            with Image.open(image_path) as opened:
                pil_img = opened.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Could not open or decode image: {exc}") from exc

        width, height = pil_img.size
        image_relative_url = f"/uploads/{image_path.name}"

        if not self.model_loaded or self.model is None or self.processor is None:
            unknown = FoodPredictionItem(
                food_name="Unable to identify confidently",
                confidence=0.0,
                category="Unknown",
                serving_description="Not available",
                notes="The food model did not load, so no prediction was made. Check the backend terminal and model download.",
            )
            return FoodRecognitionResponse(
                success=False,
                primary_prediction=unknown,
                candidate_predictions=[],
                image_url=image_relative_url,
                model_name=f"Indian Food Classifier ({MODEL_ID}) — unavailable",
                processing_time_sec=round(time.time() - start_time, 3),
                is_uncertain=True,
                uncertainty_message="The model is unavailable. Check internet access and install the backend requirements, then restart FastAPI.",
                color_profile={"image_width": width, "image_height": height, "aspect_ratio": round(width / max(height, 1), 2)},
            )

        try:
            inputs = self.processor(images=pil_img, return_tensors="pt")
            with torch.inference_mode():
                logits = self.model(**inputs).logits[0]
                probs = torch.softmax(logits, dim=-1)
            top_count = min(4, int(probs.numel()))
            top_probs, top_ids = torch.topk(probs, k=top_count)
        except Exception as exc:
            raise RuntimeError(f"Indian-food model inference failed: {exc}") from exc

        ranked: list[FoodPredictionItem] = []
        for probability, class_id in zip(top_probs.tolist(), top_ids.tolist()):
            raw_label = self.id2label.get(int(class_id), str(class_id))
            display_name = _display_label(raw_label)
            category, serving, notes = self._metadata(raw_label)
            ranked.append(
                FoodPredictionItem(
                    food_name=display_name,
                    confidence=round(float(probability) * 100, 1),
                    category=category,
                    serving_description=serving,
                    notes=notes,
                )
            )

        primary = ranked[0]
        uncertain = primary.confidence < UNCERTAIN_THRESHOLD * 100
        message = None
        if uncertain:
            message = (
                f"The model's top prediction is {primary.confidence}% confidence. "
                "Please verify it. This model is trained for a limited set of food classes and may misclassify foods outside those classes."
            )
        else:
            message = (
                "Prediction from an Indian-food image classifier. Confidence is the model's softmax score, "
                "not a guarantee of correctness. Verify the food name before relying on nutrition estimates."
            )

        return FoodRecognitionResponse(
            success=True,
            primary_prediction=primary,
            candidate_predictions=ranked[1:],
            image_url=image_relative_url,
            model_name=f"Hugging Face Indian Food Classifier ({MODEL_ID})",
            processing_time_sec=round(time.time() - start_time, 3),
            is_uncertain=uncertain,
            uncertainty_message=message,
            color_profile={
                "image_width": width,
                "image_height": height,
                "aspect_ratio": round(width / max(height, 1), 2),
            },
        )


food_recognition_service = FoodRecognitionService()
