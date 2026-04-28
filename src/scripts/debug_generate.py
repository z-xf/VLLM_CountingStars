import yaml
from pathlib import Path
from src.data.generator import SceneGenerator
from src.data.renderer import ImageRenderer

def generate_samples(num_images=3, output_dir="output_samples"):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    with open("configs/dataset.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    generator = SceneGenerator(config)
    renderer = ImageRenderer(config)
    
    for i in range(num_images):
        scene = generator.generate_scene()
        img = renderer.render(scene)
        
        # Save image
        img_path = output_path / f"sample_{i+1}.png"
        img.save(img_path)
        print(f"Generated {img_path}")
        
        # Print basic metadata
        print(f"  Stars: {len(scene.stars)} ({scene.stars[0].color}) - Sizes: {[s.size_category for s in scene.stars]}")
        print(f"  Distractors: {len(scene.distractors)}")

if __name__ == "__main__":
    generate_samples()
