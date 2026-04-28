import pytest
from src.training.rewards import calculate_reward, get_difficulty_weight

def test_difficulty_weights():
    assert get_difficulty_weight("What color are the stars in the image?") == 0.5
    assert get_difficulty_weight("How many stars are there in the image?") == 1.0
    assert get_difficulty_weight("How many small stars are there in the image?") == 1.5
    assert get_difficulty_weight("How many large stars are there in the image?") == 1.5

def test_calculate_reward_correct_and_format():
    question = "What color are the stars in the image?"
    expected = "blue"
    completion = "<think>Wait, let me look. Ah, they are all blue.</think>\n<answer>blue</answer>"
    
    # Weight: 0.5
    # Correctness: 1.0
    # Format bonus: 1.0
    # Total: 0.5 * (1.0 + 0.1 * 1.0) = 0.55
    assert calculate_reward(question, expected, completion) == 0.55
    
def test_calculate_reward_incorrect_but_formatted():
    question = "How many stars are there in the image?"
    expected = "5"
    completion = "<think>Let me count 1,2, 3</think>\n<answer>3</answer>"
    
    # Weight: 1.0
    # Correctness: 0.0
    # Format bonus: 1.0
    # Total: 1.0 * (0.0 + 0.1 * 1.0) = 0.1
    assert calculate_reward(question, expected, completion) == 0.1
    
def test_calculate_reward_unformatted():
    question = "How many large stars are there in the image?"
    expected = "2"
    completion = "There are 2 large stars."
    
    # Weight: 1.5
    # Correctness: 0.0 (Failed to extract due to no <answer> tag)
    # Format bonus: 0.0
    # Total: 1.5 * (0.0 + 0.1 * 0.0) = 0.0
    assert calculate_reward(question, expected, completion) == 0.0
