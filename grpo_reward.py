"""
Reward functions for GRPO-based output format control.

Goal: Train the model to generate ONLY:
  "The character's emotional state is X, Y, Z."
and STOP — no extra reasoning text after the emotion labels.

Key insight: The model's raw output (before ### truncation) may contain
extra reasoning text. We reward the raw output being clean and short.
"""

import re


TARGET_PREFIX = "The character's emotional state is "


def reward_format(raw_response: str) -> float:
    """
    Reward for the RAW output matching the target format exactly.
    The raw output should be ONLY "The character's emotional state is X, Y, Z."
    with NO extra reasoning text after the period.
    """
    # [Reward implementation removed - paper under review]
    raise NotImplementedError("Contact authors for details")

def reward_brevity(raw_response: str, n_generated_tokens: int = 0) -> float:
    """
    Reward shorter raw outputs. The ideal output is ~10-15 tokens.
    A clean "The character's emotional state is happy, excited." is ~12 tokens.
    Verbose reasoning outputs are 100+ tokens.
    """
    # [Reward implementation removed - paper under review]
    raise NotImplementedError("Contact authors for details")

def reward_no_reasoning(raw_response: str) -> float:
    """
    Penalize raw outputs that contain reasoning/explanation patterns.
    These patterns should NOT appear in the raw output at all.
    """
    # [Reward implementation removed - paper under review]
    raise NotImplementedError("Contact authors for details")

def reward_clean_stop(raw_response: str) -> float:
    """
    Reward for cleanly stopping after the emotion sentence.
    Check if there's only ONE sentence in the raw output.
    """
    # [Reward implementation removed - paper under review]
    raise NotImplementedError("Contact authors for details")

def reward_emotion_overlap(raw_response: str, gt_emotions: str) -> float:
    """
    Reward based on overlap between predicted and ground-truth emotions.
    Only scores if the format is correct.
    """
    # [Reward implementation removed - paper under review]
    raise NotImplementedError("Contact authors for details")

def compute_reward(raw_response: str, gt_emotions: str = None,
    # [Reward implementation removed - paper under review]
    raise NotImplementedError("Contact authors for details")
