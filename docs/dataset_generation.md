# Synthetic Dataset Generation: CountingStars

The CountingStars project avoids the need for expensive manual annotation by employing a programmatically generated **synthetic dataset** (friendly to GRPO).

## ⚙️ How it Works

A Python script generates images populated with a random number of "stars". Each star is assigned random attributes during synthesis:
- **Color**: e.g., Red, Blue, Green, Yellow, Orange.
- **Size**: e.g., Tiny, Small, Medium, Large, Huge.
- **Position**: Randomized (x, y) coordinates.
- **Distracting Shapes**: Random non-star shapes (e.g., circles, triangles) are injected into the background to increase counting difficulty.
- **Overlap Policy**: Objects are placed using collision detection. Distractors may partially overlap with stars to simulate natural occlusion, but the stars themselves are strictly non-overlapping to guarantee deterministic counting.

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

From this dictionary, deterministic QA (Question & Answer) pairs are automatically constructed:
1. **Q1 (Color of stars)**: Only asks for the color of specific stars⭐ (All stars have only one color.).
2. **Q2 (Total Count)**: Only counts the total number of true stars⭐ in the image.
3. **Q3 (Conditional Count)**: Asks for the number of stars matching a specific criteria, such as the number of large stars⭐ or small stars⭐.

## 📄 Output Format

The generator outputs raw `.png` images and a unified `dataset.json` formatted perfectly to be consumed by `TRL`'s `GRPOTrainer` and our `test_pretrained.py` script.

```json
[
    {
        "sample_id": "train_00000",
        "image_path": "data\\synthetic\\train\\images\\train_00000.png",
        "qa_pairs": [
            {
                "q": "What color are the stars in the image?",
                "a": "red"
            },
            {
                "q": "How many stars are there in the image?",
                "a": "3"
            },
            {
                "q": "How many small stars are there in the image?",
                "a": "1"
            }
        ],
        "metadata": {
            "width": 256,
            "height": 256,
            "difficulty": "medium",
            "stars": [
                {
                    "type": "star",
                    "color": "red",
                    "size_category": "small",
                    "x": 174.50902996620547,
                    "y": 182.5322586109807,
                    "pixel_size": 9.711495373681522
                },
                {
                    "type": "star",
                    "color": "red",
                    "size_category": "large",
                    "x": 160.42812821441112,
                    "y": 125.32984698411522,
                    "pixel_size": 27.83361528484287
                },
                {
                    "type": "star",
                    "color": "red",
                    "size_category": "large",
                    "x": 68.09555400391615,
                    "y": 34.30925269779351,
                    "pixel_size": 21.89897867267637
                }
            ],
            "distractors": [
                {
                    "type": "triangle",
                    "color": "red",
                    "size_category": "small",
                    "x": 128.66411034263675,
                    "y": 210.1801613949333,
                    "pixel_size": 11.505065566753828
                },
                {
                    "type": "circle",
                    "color": "red",
                    "size_category": "small",
                    "x": 28.853153611441876,
                    "y": 28.724366789325252,
                    "pixel_size": 13.108055272064728
                },
                {
                    "type": "circle",
                    "color": "yellow",
                    "size_category": "small",
                    "x": 86.00362833686405,
                    "y": 73.28691207911145,
                    "pixel_size": 13.649576763374537
                },
                {
                    "type": "triangle",
                    "color": "blue",
                    "size_category": "large",
                    "x": 89.7493636753822,
                    "y": 180.40904233337318,
                    "pixel_size": 23.957858467912544
                }
            ],
            "qa_pairs": [
                {
                    "question": "What color are the stars in the image?",
                    "answer": "red"
                },
                {
                    "question": "How many stars are there in the image?",
                    "answer": "3"
                },
                {
                    "question": "How many small stars are there in the image?",
                    "answer": "1"
                }
            ]
        }
    }
]
```
