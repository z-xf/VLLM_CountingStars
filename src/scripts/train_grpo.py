import os
import json
import argparse

# Parse CLI arguments first
parser = argparse.ArgumentParser(description="GRPO Fine-Tuning for CountingStars")
parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU IDs to run on (e.g., '0' or '0,1')")
parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset size for quick debugging")
parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of training steps (overrides epochs if > 0)")
args = parser.parse_args()

# Set visible GPUs BEFORE importing torch or unsloth
# If running with torchrun/accelerate and CUDA_VISIBLE_DEVICES is already set externally, DO NOT override it!
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from PIL import Image
import torch
from datasets import Dataset

from unsloth import FastVisionModel, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer

# Monkey patch DistributedDataParallel to expose 'config' for UnslothGRPOTrainer
from torch.nn.parallel import DistributedDataParallel as DDP
if not hasattr(DDP, "config"):
    DDP.config = property(lambda self: self.module.config)

from src.training.prompts import SYSTEM_PROMPT, build_vllm_user_prompt
from src.training.rewards import calculate_reward

def count_stars_reward_func(prompts, completions, expected_answers, **kwargs):
    """
    TRL GRPO reward function wrapper.
    prompts: list of prompts
    completions: list of generated completions
    expected_answers: passed from the dataset columns
    """
    rewards = []
    # GRPO trainer passes a list for completions, sometimes a list of lists depending on num_generations
    for completion, expected in zip(completions, expected_answers):
        # The completion is usually a string, or a list of generated tokens
        comp_text = completion[0]["content"] if isinstance(completion, list) and isinstance(completion[0], dict) else completion
        
        reward = calculate_reward(expected, comp_text)
        rewards.append(float(reward))
    return rewards

def format_dataset(json_path):
    """
    Loads dataset.json and formats it for TRL's GRPOTrainer.
    GRPOTrainer generally expects a 'prompt' column (can be messages) and we also pass 'expected_answers'.
    For vision models, we need to pass the images inside the prompt or as a separate 'images' column.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        ds = json.load(f)
        
    formatted_data = {
        "prompt": [],
        "images": [],
        "expected_answers": []
    }
    
    for ex in ds:
        # Convert Windows backslashes to Unix forward slashes
        image_path = ex["image_path"].replace("\\", "/")
        qa_pairs = ex["qa_pairs"]
        
        q1 = qa_pairs[0]["q"]
        q2 = qa_pairs[1]["q"]
        q3 = qa_pairs[2]["q"]
        ans = [qa_pairs[0]["a"], qa_pairs[1]["a"], qa_pairs[2]["a"]]
        
        user_prompt = build_vllm_user_prompt(q1, q2, q3)
        image = Image.open(image_path).convert("RGB")
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt}
            ]}
        ]
        
        formatted_data["prompt"].append(messages)
        formatted_data["images"].append([image])
        formatted_data["expected_answers"].append(ans)
        
    return Dataset.from_dict(formatted_data)

def main():

    # Force wandb offline mode
    os.environ["WANDB_MODE"] = "offline"
    
    MODEL_NAME = "../Qwen3.5-4B"
    TRAIN_DATA_PATH = "./data/synthetic/train/dataset.json"
    TEST_DATA_PATH = "./data/synthetic/test/dataset.json"
    OUTPUT_DIR = "./results/CountingStars_GRPO_Qwen3.5_4B_multiple_GPU"
    
    # 1. Load Model and Tokenizer
    print("Loading model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        load_in_4bit=True, # Use 4-bit for memory-efficient LoRA training
        use_gradient_checkpointing="unsloth",
    )
    
    # Disable torch.compile integration with Unsloth temporarily
    # This prevents the accelerate AlignDevicesHook crash
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
    
    # 2. Add LoRA Adapter
    print("Adding LoRA adapter...")
    model = FastVisionModel.get_peft_model(
        model,
        r=32, # Increased from 16 to 32 for better reasoning capacity
        lora_alpha=32,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_rslora=False,
    )
    
    # 3. Load Dataset
    print("Preparing dataset...")
    train_dataset = format_dataset(TRAIN_DATA_PATH)
    eval_dataset = format_dataset(TEST_DATA_PATH)

    if args.max_samples:
        train_dataset = train_dataset.select(range(min(len(train_dataset), args.max_samples)))
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), max(1, args.max_samples // 4))))
        print(f"Debug Mode: Limited Train Dataset to {len(train_dataset)} and Eval Dataset to {len(eval_dataset)}")
    
    # 4. Configure GRPO
    eval_freq = 10 if args.max_steps <= 0 else min(10, max(1, args.max_steps // 2))

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=eval_freq,
        save_strategy="steps",
        save_steps=eval_freq,
        save_total_limit=2,
        load_best_model_at_end=True,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4, # Number of completions to generate per prompt for GRPO
        #max_prompt_length=512, # not exist
        max_completion_length=128, # Reduced from 512 (CountingStars answers are very short)
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        optim="adamw_8bit",
        report_to="wandb",
        ddp_find_unused_parameters=False, # Required for some multi-GPU setups
        remove_unused_columns=False, # Required so our expected_answers column isn't dropped
    )
    
    # 5. Initialize Trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[count_stars_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, # Pass evaluation dataset
    )
    
    # 6. Train
    print("Starting GRPO Fine-Tuning...")
    trainer.train()
    
    # 7. Save LoRA Adapter
    print("Saving LoRA model...")
    model.save_pretrained(f"{OUTPUT_DIR}/lora_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_model")
    print("Training finished!")

if __name__ == "__main__":
    main()
