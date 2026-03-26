from __future__ import annotations

from random import Random

from memory_manager import ReMemorizeMemoryManager
from reward_manager import (
    RewardCalibrator,
    RewardComponents,
    RewardWeights,
    calibrated_composite_reward,
    composite_reward,
    training_objective,
)
from scorers import BaseRewardScorer, HeuristicRewardScorer, LLMJudgeRewardScorer
from trainer import ReMemorizeTrainer, TrainerConfig
from policy_manager import (
    build_policy_sample,
)

__all__ = [
    "ReMemorizeMemoryManager",
    "ReMemorizeTrainer",
    "TrainerConfig",
    "RewardComponents",
    "RewardCalibrator",
    "RewardWeights",
    "BaseRewardScorer",
    "HeuristicRewardScorer",
    "LLMJudgeRewardScorer",
    "build_policy_sample",
    "calibrated_composite_reward",
    "composite_reward",
    "training_objective",
]


if __name__ == "__main__":
    rng = Random(42)
    T, d_k, d_v = 80, 32, 32
    keys = [[rng.gauss(0.0, 1.0) for _ in range(d_k)] for _ in range(T)]
    values = [[rng.gauss(0.0, 1.0) for _ in range(d_v)] for _ in range(T)]

    mm = ReMemorizeMemoryManager(
        d_key=d_k,
        d_value=d_v,
        d_memory=64,
        seed=7,
    )
    stats = mm.update(keys, values)
    display_stats = {k: v for k, v in stats.items() if k != "readouts"}
    print("Memory update stats:", display_stats)

    r = composite_reward(
        r_factual=0.84,
        r_coherence=0.78,
        r_completeness=0.62,
        r_hallucination=0.21,
        weights=RewardWeights(alpha=1.0, beta=0.3, gamma=0.1, delta=0.8),
    )
    print("Composite reward:", round(r, 4))
    print("Training objective:", round(training_objective(supervised_loss=1.2, reward=r, lambda_rl=0.5), 4))

    calibrator = RewardCalibrator(ema_alpha=0.1, clip_z=2.5, factual_floor=0.5, factual_floor_penalty=0.2)
    cstats = calibrated_composite_reward(
        RewardComponents(
            factual=0.84,
            coherence=0.78,
            completeness=0.62,
            hallucination=0.21,
        ),
        calibrator=calibrator,
    )
    print("Calibrated reward:", round(cstats.calibrated_reward, 4))

    # Build a trainer and run one PPO + one GRPO step.
    trainer = ReMemorizeTrainer(
        memory_manager=mm,
        reward_calibrator=calibrator,
        reward_weights=RewardWeights(alpha=1.0, beta=0.3, gamma=0.1, delta=0.8),
        config=TrainerConfig(ppo_lr=1e-5, clip_eps=0.2, entropy_coef=0.005, group_size=4),
        seed=42,
    )
    key_windows = []
    rewards = []
    for i in range(8):
        start = i * 4
        key_windows.append(keys[start : start + 32])
        rewards.append(RewardComponents(
            factual=0.65 + 0.03 * (i % 4),
            coherence=0.60 + 0.02 * (i % 3),
            completeness=0.50 + 0.01 * (i % 5),
            hallucination=0.30 - 0.02 * (i % 4),
        ))
    batch = trainer.collect_policy_batch(key_windows, rewards)

    ppo_stats = trainer.ppo_step(batch)
    print("PPO stats:", ppo_stats)

    grpo_stats = trainer.grpo_step(batch)
    print("GRPO stats:", grpo_stats)
