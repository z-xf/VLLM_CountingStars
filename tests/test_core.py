import pytest
import yaml
from pathlib import Path
from src.data.generator import SceneGenerator

@pytest.fixture
def config():
    config_path = Path("configs/dataset.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def test_scene_generator_rules(config):
    generator = SceneGenerator(config)
    for _ in range(100):
        scene = generator.generate_scene()
        
        # Rule 1: All stars have the same color
        star_colors = {star.color for star in scene.stars}
        assert len(star_colors) == 1
        
        # Rule 2: At least one small star and one large star
        star_sizes = {star.size_category for star in scene.stars}
        assert "small" in star_sizes
        assert "large" in star_sizes
