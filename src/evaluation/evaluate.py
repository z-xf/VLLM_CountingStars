import os
import json
import argparse
import re
import matplotlib.pyplot as plt
import numpy as np
from src.training.rewards import extract_answers

def evaluate_dir(results_dir, references):
    """
    Scans a directory of inference results (.txt files) and compares them 
    against the ground truth references.
    
    Returns:
        stats (dict): A dictionary containing accuracy metrics overall and broken down by difficulty.
    """
    # Initialize a dictionary to track totals and correct predictions
    stats = {
        "total": 0,
        "q1_color_correct": 0,    # How many times it correctly answered Q1 (Color)
        "q2_number_correct": 0,   # How many times it correctly answered Q2 (Number of stars)
        "q3_size_correct": 0,     # How many times it correctly answered Q3 (Size of stars)
        "difficulty": {           # Breakdown by difficulty level
            "easy": {"correct": 0, "total": 0, "q1": 0, "q2": 0, "q3": 0},
            "medium": {"correct": 0, "total": 0, "q1": 0, "q2": 0, "q3": 0},
            "hard": {"correct": 0, "total": 0, "q1": 0, "q2": 0, "q3": 0}
        },
        "fully_correct": 0        # How many times it got all 3 questions right
    }

    if not os.path.exists(results_dir):
        print(f"Warning: Results directory {results_dir} not found.")
        return stats

    # Iterate through every .txt file in the results directory
    txt_files = [f for f in os.listdir(results_dir) if f.endswith(".txt")]
    for filename in txt_files:
        filepath = os.path.join(results_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract the sample ID using regex (e.g. "Sample: train_0001")
        sample_match = re.search(r"Sample:\s*([A-Za-z0-9_]+)", content)
        if not sample_match:
            continue
        sample_id = sample_match.group(1).strip()

        # Skip if we don't have ground truth for this sample
        if sample_id not in references:
            continue

        ref = references[sample_id]
        expected = ref["expected"]
        difficulty = ref["difficulty"]

        # Parse the actual generated output. 
        # The inference script separates the prompt/metadata and the generated answer with "---"
        parts = content.split("-" * 40)
        if len(parts) < 2:
            continue
        generated_text = parts[-1].strip()

        # Use the same regex logic used during training to parse the model's <answer> tags
        extracted = extract_answers(generated_text)

        # Check if the extracted answers exactly match the expected ground truths
        is_q1 = (len(extracted) > 0 and extracted[0] == expected[0])
        is_q2 = (len(extracted) > 1 and extracted[1] == expected[1])
        is_q3 = (len(extracted) > 2 and extracted[2] == expected[2])

        # Update absolute totals
        stats["total"] += 1
        stats["difficulty"][difficulty]["total"] += 1

        # Update specific question accuracy
        if is_q1:
            stats["q1_color_correct"] += 1
            stats["difficulty"][difficulty]["q1"] += 1
        if is_q2:
            stats["q2_number_correct"] += 1
            stats["difficulty"][difficulty]["q2"] += 1
        if is_q3:
            stats["q3_size_correct"] += 1
            stats["difficulty"][difficulty]["q3"] += 1

        # Update fully correct tracking
        if is_q1 and is_q2 and is_q3:
            stats["fully_correct"] += 1
            stats["difficulty"][difficulty]["correct"] += 1

    return stats

def plot_results(labels, all_stats, save_path):
    """
    Generates a grouped bar chart comparing the accuracy of multiple methods.
    """
    print(f"\nGenerating plot at {save_path}...")
    metrics = ["Overall", "Q1 (Color)", "Q2 (Number)", "Q3 (Size)"]
    x = np.arange(len(metrics)) # Base positions for the 4 metric groups
    width = 0.8 / len(labels)   # Dynamically calculate bar width so they fit together
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars iteratively for each evaluated method (e.g. Base Model vs Trained Model)
    for i, (label, stats) in enumerate(zip(labels, all_stats)):
        total = stats["total"] if stats["total"] > 0 else 1
        ov_acc = stats['fully_correct'] / total * 100
        q1_acc = stats['q1_color_correct'] / total * 100
        q2_acc = stats['q2_number_correct'] / total * 100
        q3_acc = stats['q3_size_correct'] / total * 100
        
        values = [ov_acc, q1_acc, q2_acc, q3_acc]
        
        # Calculate offset to group bars side by side properly
        offset = (i - len(labels)/2 + 0.5) * width
        rects = ax.bar(x + offset, values, width, label=label)
        ax.bar_label(rects, fmt='%.1f', padding=3)

    # Styling the plot
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Evaluation Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 110) # Cap at 110 so the text labels don't get cutoff at 100%
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    print("Plot saved successfully.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate CountingStars synthetic dataset inference results.")
    
    # We use nargs='+' to accept a list of directories in the command line 
    # Example: --results_dirs ./dir1 ./dir2
    parser.add_argument("--results_dirs", type=str, nargs='+', required=True, help="Directories containing inference .txt files (one or more)")
    parser.add_argument("--labels", type=str, nargs='+', help="Labels for the methods being evaluated (must match number of dirs)")
    parser.add_argument("--dataset_json", type=str, default="data/synthetic/test/dataset.json", help="Path to the test dataset json")
    parser.add_argument("--save_plot", type=str, default="./output/evaluation_plot.png", help="Path to save the comparison plot")
    args = parser.parse_args()

    if not os.path.exists(args.dataset_json):
        print(f"Error: Dataset JSON {args.dataset_json} not found.")
        return

    # 1. Load ground truth JSON dataset to understand what the correct answers should be
    with open(args.dataset_json, "r", encoding="utf-8") as f:
        ds = json.load(f)

    # 2. Build a reference dictionary mapped by 'sample_id' for fast lookups (O(1) time complexity)
    references = {}
    for ex in ds:
        qas = ex.get("qa_pairs", ex.get("metadata", {}).get("qa_pairs", []))
        expected_answers = [str(qa["a" if "a" in qa else "answer"]).strip().lower() for qa in qas]
        references[ex["sample_id"]] = {
            "difficulty": ex.get("metadata", {}).get("difficulty", "medium"),
            "expected": expected_answers
        }

    # If the user didn't specify custom plot labels, default them to the directory paths
    labels = args.labels if args.labels and len(args.labels) == len(args.results_dirs) else args.results_dirs
    
    all_stats = []
    
    # 3. Evaluate each directory one by one
    for label, results_dir in zip(labels, args.results_dirs):
        print("\n" + "="*50)
        print(f" Evaluating: {label} ({results_dir})")
        print("="*50)
        
        stats = evaluate_dir(results_dir, references)
        all_stats.append(stats)
        
        total = stats["total"]
        if total == 0:
            print("No valid results found to evaluate.")
            continue
            
        print(f"Total Samples Evaluated: {total}")
        print(f"Overall Full Correct (All 3 Qs): {stats['fully_correct']}/{total} ({stats['fully_correct']/total*100:.2f}%)")
        print(f"Color Accuracy (Q1): {stats['q1_color_correct']}/{total} ({stats['q1_color_correct']/total*100:.2f}%)")
        print(f"Total Number Accuracy (Q2): {stats['q2_number_correct']}/{total} ({stats['q2_number_correct']/total*100:.2f}%)")
        print(f"Size Number Accuracy (Q3): {stats['q3_size_correct']}/{total} ({stats['q3_size_correct']/total*100:.2f}%)\n")

        print("--- Accuracy by Difficulty (Full Correct) ---")
        for diff in ["easy", "medium", "hard"]:
            d_total = stats["difficulty"][diff]["total"]
            d_corr = stats["difficulty"][diff]["correct"]
            if d_total > 0:
                q1_corr = stats["difficulty"][diff]["q1"]/d_total*100
                q2_corr = stats["difficulty"][diff]["q2"]/d_total*100
                q3_corr = stats["difficulty"][diff]["q3"]/d_total*100
                print(f"{diff.capitalize():<7} | Full Correct: {d_corr}/{d_total} ({d_corr/d_total*100:>5.2f}%) | Q1: {q1_corr:>5.1f}% | Q2: {q2_corr:>5.1f}% | Q3: {q3_corr:>5.1f}%")

    # 4. If we successfully evaluated at least one directory, generate the comparison plot
    if len(all_stats) > 0:
        plot_results(labels, all_stats, args.save_plot)

if __name__ == "__main__":
    main()
