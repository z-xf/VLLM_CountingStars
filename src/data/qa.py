import random
from typing import List
from src.core.models import SceneMetadata, QAPair

def generate_qa_pairs(scene: SceneMetadata, rng: random.Random) -> List[QAPair]:
    if not scene.stars:
        return []
    
    # Q1 asks star color. (All stars in the image share one color)
    star_color = scene.stars[0].color
    
    # Q2 asks total star count.
    total_stars = len(scene.stars)
    
    # Q3 asks count of small or large stars.
    target_size = rng.choice(["small", "large"])
    size_count = sum(1 for s in scene.stars if s.size_category == target_size)
    
    return [
        QAPair(question="What color are the stars in the image?", answer=star_color),
        QAPair(question="How many stars are there in the image?", answer=str(total_stars)),
        QAPair(question=f"How many {target_size} stars are there in the image?", answer=str(size_count))
    ]
