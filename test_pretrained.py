import os
import json
import argparse
from PIL import Image
from unsloth import FastVisionModel
from src.training.prompts import SYSTEM_PROMPT, build_vllm_user_prompt
from src.training.rewards import calculate_reward

parser = argparse.ArgumentParser(description="Evaluate the trained GRPO model")
parser.add_argument("--model_path", type=str, default="./results/CountingStars_GRPO_Qwen3.5_4B_multiple_GPU/lora_model", help="Path to the trained LoRA model checkpoint (e.g., outputs/checkpoint-300) or a base model (e.g., ../Qwen3.5-4B)")
parser.add_argument("--output_dir", type=str, default="./output/CountingStars_Eval_Trained", help="Directory to save evaluation results")
parser.add_argument("--dataset_path", type=str, default="data/synthetic/test/dataset.json", help="Path to test dataset")
parser.add_argument("--is_base_model", action="store_true", help="Set this flag if evaluating the base Qwen model before training")
args = parser.parse_args()

# Load the trained Vision-Language Model
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=args.model_path,
    load_in_4bit=False,  # False for 16-bit
    use_gradient_checkpointing="unsloth",
)

# By default, from_pretrained tries to find a PEFT config.
# If we set is_base_model, we just don't load adapters.
if not args.is_base_model:
    print("Assuming a LoRA adapter is at model_path. To skip, pass --is_base_model.")
else:
    print("Testing RAW base model (no adapters).")

FastVisionModel.for_inference(model)  # Enable faster inference

# Load the synthetic dataset
with open(args.dataset_path, "r", encoding="utf-8") as f:
    ds = json.load(f)

# Perform inference and save the results
path = args.output_dir
os.makedirs(path, exist_ok=True)

total_reward = 0

for i, ex in enumerate(ds):
    print(f"Processing sample {i}...")
    
    # On Linux, pathlib doesn't treat backslashes as separators by default,
    # so we explicitly replace them first for true cross-platform behavior.
    image_path = ex["image_path"].replace("\\", "/")
    
    qa_pairs = ex["qa_pairs"]
    
    # Ensure there are exactly 3 pairs as per our generation logic
    q1 = qa_pairs[0]["q"]
    q2 = qa_pairs[1]["q"]
    q3 = qa_pairs[2]["q"]
    
    expected_answers = [qa_pairs[0]["a"], qa_pairs[1]["a"], qa_pairs[2]["a"]]
    
    user_prompt = build_vllm_user_prompt(q1, q2, q3)
    
    # Load the image
    image = Image.open(image_path).convert("RGB")
    
    # Vision-Language Models require specific message structuring indicating image presence
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_prompt}
        ]}
    ]

    # Use the FastVisionModel's associated tokenizer (which wraps the processor) 
    # to format the image + text arrays.
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # FastVisionModel usually passes the image directly into tokenizer/processor
    inputs = tokenizer(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to("cuda")
    
    res = model.generate(**inputs, max_new_tokens=1024)

    # Decode and strip prompt output securely
    generated_ids = [ids[len(inputs["input_ids"][0]):] for ids in res]
    generated = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    
    # Calculate deterministic reward
    reward = calculate_reward(expected_answers, generated)
    total_reward += reward
    
    print(f"Expected: {expected_answers}")
    print(f"Generated: {generated}")
    print(f"Reward: {reward}\n")

    with open(f"{path}/{i}.txt", "w", encoding="utf-8") as f:
        f.write(f"Sample: {ex['sample_id']}\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Expected Answers: {expected_answers}\n")
        f.write(f"Reward: {reward}\n")
        f.write("-" * 40 + "\n")
        f.write(generated)

print(f"Average Reward: {total_reward / len(ds):.3f}")
