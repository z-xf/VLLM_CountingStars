# Synthetic Dataset Generation: CountingStars

The CountingStars project avoids the need for expensive manual annotation by employing a programmatically generated synthetic dataset. This allows us to train the VLLM on counting and multi-attribute recognition tasks with absolute mathematical ground truths, which is crucial for the Gradients in our GRPO reward functions.

## ⚙️ How it Works

A Python script generates images populated with a random number of "stars". Each star is assigned random attributes during synthesis:
- **Color**: e.g., Red, Blue, Green, Yellow, Orange.
- **Size**: e.g., Tiny, Small, Medium, Large, Huge.
- **Position**: Randomized (x, y) coordinates with collision avoidance to ensure stars don't overlap in the generated `.png`.

## 🚀 Generating Your Own Dataset

To generate a completely new batch of images and corresponding `dataset.json` ground truths, run the dataset generation script included in the `src/data` module.

```bash
# Example command (navigate to project root first)
cd CountingStars
python -m src.scripts.generate_dataset \
      --config ./configs/dataset.yaml \
      --output ./data/synthetic \
      --num_train_samples 100 \
      --num_test_samples 30
```

This will automatically create a new `train/` and `test/` folder populated with the random PNG stars and the pre-formatted JSON structures.

## 🧮 Ground Truth Generation

Because the images are generated programmatically, the script inherently knows the exact state of the image. It compiles a localized JSON dictionary that tracks the precise properties of the generated image. 

From this dictionary, deterministic QA (Question & Answer) pairs are automatically constructed:
1. **Q1 (Color of the largest)**: Calculated by sorting the generated objects array by pixel size area and returning the color property of the max object.
2. **Q2 (Total Count)**: Calculated via a simple `len()` of the generated objects array.
3. **Q3 (Color of the smallest)**: Calculated by sorting the objects array by area and returning the min object's color.

## 📄 Output Format

The generator outputs raw `.png` images and a unified `dataset.json` formatted perfectly to be consumed by `TRL`'s `GRPOTrainer` and our `test_pretrained.py` script.

```json
[
  {
    "sample_id": "train_0000",
    "image_path": "data/synthetic/train/images/train_0000.png",
    "qa_pairs": [
      {"q": "What is the color of the largest star?", "a": "green"},
      {"q": "How many stars are there in total?", "a": "5"},
      {"q": "What is the color of the smallest star?", "a": "red"}
    ],
    "metadata": {
      "difficulty": "medium"
    }
  }
]
```
