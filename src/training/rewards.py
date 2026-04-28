from typing import Dict, Any, Optional, List
import re

def get_difficulty_weights() -> List[float]:
    """Returns the predefined weights for Q1, Q2, and Q3."""
    return [0.5, 1.0, 1.5]

def extract_answers(completion: str) -> List[str]:
    """Helper to extract all exact answer texts from <answer> tags."""
    return [match.strip().lower() for match in re.findall(r'<answer>\s*(.*?)\s*</answer>', completion, re.IGNORECASE | re.DOTALL)]

def calculate_reward(expected_answers: List[str], completion: str) -> float:
    """
    Computes a purely deterministic reward:
    reward = sum over i for each question (difficulty_weight_i * (correctness_i + 0.1 * format_bonus_i))
    
    Format bonus logic: +1.0 if exactly 3 <answer> tags are well-formed.
    Correctness logic: +1.0 if the string directly matches expected_answer.
    """
    weights = get_difficulty_weights()
    
    # Extract answers and check format
    extracted = extract_answers(completion)
    has_exactly_three_answers = len(extracted) == 3
    
    format_bonus = 1.0 if has_exactly_three_answers else 0.0
    
    total_reward = 0.0
    
    # Evaluate correctness per question
    for i in range(3):
        weight = weights[i]
        correctness = 0.0
        
        expected_clean = str(expected_answers[i]).strip().lower()
        
        if i < len(extracted) and extracted[i] == expected_clean:
            correctness = 1.0
            
        total_reward += weight * (correctness + 0.1 * format_bonus)
        
    return total_reward
