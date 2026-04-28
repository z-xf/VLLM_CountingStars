import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import asdict
from src.core.models import SceneMetadata
from src.data.renderer import ImageRenderer

class DatasetExporter:
    def __init__(self, output_dir: str, renderer: ImageRenderer):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir = self.output_dir / "images"
        self.image_dir.mkdir(exist_ok=True)
        self.renderer = renderer
        
    def export_scene(self, scene: SceneMetadata, sample_id: str) -> Dict[str, Any]:
        """Renders the scene as an image and produces a flat JSON record."""
        # Render and save image
        img = self.renderer.render(scene)
        image_filename = f"{sample_id}.png"
        image_path = self.image_dir / image_filename
        img.save(image_path)
        
        # Flattened QA pairs to keep structure simple
        qa_list = [{"q": qa.question, "a": qa.answer} for qa in scene.qa_pairs]
        
        record = {
            "sample_id": sample_id,
            "image_path": str(image_path),
            "qa_pairs": qa_list,
            "metadata": asdict(scene)  # Include strict metadata for potential future checks or rewards
        }
        return record

    def export_dataset(self, scenes: List[SceneMetadata], prefix: str = "sample_"):
        """Exports a batch of generated scenes to PNGs and a combined formatted JSON file."""
        json_path = self.output_dir / "dataset.json"
        
        all_records = []
        for i, scene in enumerate(scenes):
            sample_id = f"{prefix}{i:05d}"
            record = self.export_scene(scene, sample_id)
            all_records.append(record)
            
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_records, f, indent=4)
