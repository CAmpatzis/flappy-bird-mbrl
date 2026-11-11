import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from ppo_agent import PPOAgent, train, evaluate
import os
import sys

if 'ppo_agent' in sys.modules:
    del sys.modules['ppo_agent']
from ppo_agent import PPOAgent, train, evaluate

if __name__ == "__main__":
    print("Initializing PPO Agent for Flappy Bird")
    agent = PPOAgent()

    print("\n" + "="*50)
    print("INITIAL EVALUATION (1000 episodes)")
    print("="*50)
    mean_score, max_score = evaluate(agent, 10)
    print(f"\nFinal Results - Mean Score: {mean_score:.2f}, Max Score: {max_score}")

    print("\n" + "="*50)
    print("TRAINING (100,000 episodes)")
    print("="*50)
    train(agent, 100000)

    print("\n" + "="*50)
    print("FINAL EVALUATION (1000 episodes)")
    print("="*50)
    mean_score, max_score = evaluate(agent, 10)
    print(f"\nFinal Results - Mean Score: {mean_score:.2f}, Max Score: {max_score}")

    print("\nTraining complete. Check TensorBoard with: tensorboard --logdir runs")