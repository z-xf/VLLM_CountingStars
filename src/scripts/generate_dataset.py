import argparse
import yaml
import random
from pathlib import Path
from src.data.generator import SceneGenerator
from src.data.renderer import ImageRenderer
from src.data.qa import generate_qa_pairs
from src.data.exporter import DatasetExporter

def main():
    parser = argparse.ArgumentParser(description="Counting Stars dataset generation.")
    parser.add_argument("--config", type=str, default="configs/dataset.yaml", help="Path to YAML config")
    parser.add_argument("--output", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--num_train_samples", type=int, default=10, help="Number of synthetic train samples to generate")
    parser.add_argument("--num_test_samples", type=int, default=10, help="Number of synthetic samples to generate")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Initialize generators
    rng = random.Random(config.get("seed", 42))
    generator = SceneGenerator(config)
    renderer = ImageRenderer(config)
    
    # Create distinct Train and Test exporters
    train_out = Path(args.output) / "train"
    test_out = Path(args.output) / "test"
    num_train_samples = args.num_train_samples
    num_test_samples = args.num_test_samples

    exporter_train = DatasetExporter(str(train_out), renderer)
    exporter_test = DatasetExporter(str(test_out), renderer)

    print(f"Generating Split Dataset: {num_train_samples} Train, {num_test_samples} Test with 40%/40%/20%  difficulty split...")
    
    train_counts = {"easy": int(0.4*num_train_samples), "medium": int(0.4*num_train_samples), "hard": int(0.2*num_train_samples)}
    test_counts = {"easy": int(0.2*num_test_samples), "medium": int(0.4*num_test_samples), "hard": int(0.4*num_test_samples)}

    # Generate Train Split
    scenes_train = []
    for diff, count in train_counts.items():
        for _ in range(count):
            scene = generator.generate_scene(difficulty=diff)
            scene.qa_pairs = generate_qa_pairs(scene, rng)
            scenes_train.append(scene)
    # Shuffle to avoid blocks of same difficulty
    rng.shuffle(scenes_train)
    exporter_train.export_dataset(scenes_train, prefix="train_")

    # Generate Test Split
    scenes_test = []
    for diff, count in test_counts.items():
        for _ in range(count):
            scene = generator.generate_scene(difficulty=diff)
            scene.qa_pairs = generate_qa_pairs(scene, rng)
            scenes_test.append(scene)
    rng.shuffle(scenes_test)
    exporter_test.export_dataset(scenes_test, prefix="test_")

    print(f"Dataset successfully generated at '{args.output}/'")

if __name__ == "__main__":
    main()
