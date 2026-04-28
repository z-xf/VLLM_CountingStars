import math
import random
from typing import List, Tuple, Dict, Any
from src.core.models import Star, Distractor, SceneMetadata, Shape

class SceneGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rng = random.Random(config.get("seed", 42))
        
    def _is_overlapping(self, x: float, y: float, pixel_size: float, placed_shapes: List[Shape], margin: float = 2.0) -> bool:
        for shape in placed_shapes:
            dist = math.hypot(x - shape.x, y - shape.y)
            if dist < (pixel_size + shape.pixel_size + margin):
                return True
        return False
        
    def _sample_position(self, pixel_size: float, width: float, height: float, placed_shapes: List[Shape], max_retries: int = 100) -> Tuple[float, float]:
        for _ in range(max_retries):
            x = self.rng.uniform(pixel_size, width - pixel_size)
            y = self.rng.uniform(pixel_size, height - pixel_size)
            if not self._is_overlapping(x, y, pixel_size, placed_shapes):
                return x, y
        
        raise RuntimeError(f"Failed to place shape of size {pixel_size} after {max_retries} retries due to overlap. Increase image_size or reduce counts.")

    def generate_scene(self, difficulty: str = None) -> SceneMetadata:
        width, height = self.config["image_size"]
        colors = self.config["colors"]
        star_color = self.rng.choice(colors)
        
        # Determine difficulty level randomly or use provided
        difficulty = difficulty or self.rng.choice(["easy", "medium", "hard"])
        level_rules = self.config["levels"][difficulty]
        
        # Calculate overall objects
        min_objects = level_rules["objects_min"]
        max_objects = level_rules["objects_max"]
        total_objects = self.rng.randint(min_objects, max_objects)
        
        # Calculate Stars vs Distractors based on difficulty ratios
        star_ratio = self.rng.uniform(level_rules["star_ratio_min"], level_rules["star_ratio_max"])
        num_stars = max(2, int(total_objects * star_ratio)) # Minimum 2 stars required
        num_distractors = total_objects - num_stars

        # Hard clamp for easy distractors
        if difficulty == "easy" and num_distractors > 2:
            num_distractors = 2
            num_stars = total_objects - 2
        
        placed_shapes = []
        stars = []
        
        # Ensure at least one small and one large star
        sizes_needed = ["small", "large"]
        # Fill rest randomly
        sizes_needed.extend([self.rng.choice(["small", "large"]) for _ in range(num_stars - 2)])
        self.rng.shuffle(sizes_needed)
        
        for size_cat in sizes_needed:
            size_range = self.config["sizes"][size_cat]
            pixel_size = self.rng.uniform(*size_range)
            x, y = self._sample_position(pixel_size, width, height, placed_shapes)
            star = Star(
                type="star",
                color=star_color,
                size_category=size_cat,
                x=x,
                y=y,
                pixel_size=pixel_size
            )
            stars.append(star)
            placed_shapes.append(star)
            
        distractors = []
        if difficulty == "easy":
            distractor_shapes = self.rng.sample(self.config["distractor_shapes"], level_rules["max_distractor_types"])
            available_colors = [c for c in colors if c != star_color]
        else:
            distractor_shapes = self.rng.sample(self.config["distractor_shapes"], min(len(self.config["distractor_shapes"]), level_rules["max_distractor_types"]))
            available_colors = colors
            
        for _ in range(num_distractors):
            shape_type = self.rng.choice(distractor_shapes)
            color = self.rng.choice(available_colors)
            size_cat = self.rng.choice(["small", "large"])
            size_range = self.config["sizes"][size_cat]
            pixel_size = self.rng.uniform(*size_range)
            x, y = self._sample_position(pixel_size, width, height, placed_shapes)
            distractor = Distractor(
                type=shape_type,
                color=color,
                size_category=size_cat,
                x=x,
                y=y,
                pixel_size=pixel_size
            )
            distractors.append(distractor)
            placed_shapes.append(distractor)
            
        return SceneMetadata(
            width=width,
            height=height,
            difficulty=difficulty,
            stars=stars,
            distractors=distractors
        )
