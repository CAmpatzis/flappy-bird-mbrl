import os
import warnings
warnings.filterwarnings('ignore')
os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Run pygame headless
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import cv2
import sys
from collections import deque
import pygame
import time
import pickle
from pathlib import Path
sys.path.append("flappy_bird_repo/game/")
import wrapped_flappy_bird as game
from vec_env import VecEnv

# ============= CHECKPOINT UTILITIES =============
def get_latest_checkpoint(checkpoint_dir='checkpoints/ppo'):
    """Get the latest checkpoint file if it exists."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None
    checkpoint_files = sorted(checkpoint_path.glob('checkpoint_*.pkl'))
    return checkpoint_files[-1] if checkpoint_files else None

def save_checkpoint(checkpoint_dir, episode, agent, rewards, timestamp):
    """Save training checkpoint."""
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_file = checkpoint_path / f'checkpoint_{episode:05d}_{timestamp}.pkl'
    checkpoint_data = {
        'episode': episode,
        'agent_state': {
            'policy': agent.policy.state_dict(),
            'optimizer': agent.optimizer.state_dict(),
        },
        'training_rewards': rewards,
        'hyperparams': {
            'gamma': agent.gamma,
            'eps_clip': agent.eps_clip,
            'batch_size': agent.batch_size,
        }
    }
    
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    # Clean up old checkpoints (keep last 3)
    all_checkpoints = sorted(checkpoint_path.glob('checkpoint_*.pkl'))
    for old_checkpoint in all_checkpoints[:-3]:
        old_checkpoint.unlink()
    
    return checkpoint_file

def load_checkpoint(checkpoint_file, agent):
    """Load checkpoint and resume training."""
    with open(checkpoint_file, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    # Restore agent state
    agent.policy.load_state_dict(checkpoint_data['agent_state']['policy'])
    agent.optimizer.load_state_dict(checkpoint_data['agent_state']['optimizer'])
    
    return checkpoint_data['episode'], checkpoint_data['training_rewards']

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 10 * 10, 512)
        self.actor = nn.Linear(512, 2)
        self.critic = nn.Linear(512, 1)
        self._init_weights()

    def _init_weights(self):
        for m in [self.conv1, self.conv2, self.conv3, self.fc1]:
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)
        # Small actor init → near-uniform initial policy (critical for exploration)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.actor(x), self.critic(x)

class PPOAgent:
    def __init__(self, lr=2.5e-4, gamma=0.99, eps_clip=0.2, k_epochs=4, lam=0.95, batch_size=256):
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")
        self.policy = ActorCritic().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.lam = lam
        self.batch_size = batch_size
        self.entropy_coef = 0.1
        self.max_grad_norm = 0.5
        self.MseLoss = nn.MSELoss()

    def preprocess(self, image):
        image = image.transpose(1, 0, 2)  # pygame surfarray (W,H,C) -> (H,W,C)
        image = cv2.cvtColor(cv2.resize(image, (80, 80)), cv2.COLOR_BGR2GRAY)
        return (image / 255.0).astype(np.float32)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_logits, _ = self.policy(state)
            action_probs = F.softmax(action_logits, dim=-1).squeeze(0)
            dist = Categorical(action_probs)
            action = dist.sample()
            logprob = dist.log_prob(action)
        return action.item(), logprob

    def update(self, memory, num_envs=1):
        if len(memory.states) == 0:
            return None

        rewards = torch.tensor(memory.rewards, dtype=torch.float32, device=self.device)
        is_terminals = torch.tensor(memory.is_terminals, dtype=torch.float32, device=self.device)
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()

        with torch.no_grad():
            _, state_values = self.policy(old_states)
            state_values = state_values.squeeze(-1)

        # Compute GAE per-environment (buffer is interleaved: e0t0, e1t0, ..., e0t1, e1t1, ...)
        N = len(rewards)
        advantages = torch.zeros(N, device=self.device)

        if num_envs > 1:
            T = N // num_envs
            for e in range(num_envs):
                idx = list(range(e, T * num_envs, num_envs))
                env_rew = rewards[idx]
                env_term = is_terminals[idx]
                env_val = state_values[idx]
                vals = torch.cat([env_val, torch.zeros(1, device=self.device)])
                gae = 0.0
                for t in reversed(range(T)):
                    delta = env_rew[t] + self.gamma * vals[t + 1] * (1 - env_term[t]) - vals[t]
                    gae = delta + self.gamma * self.lam * (1 - env_term[t]) * gae
                    advantages[idx[t]] = gae
        else:
            values = torch.cat((state_values, torch.zeros(1, device=self.device)))
            gae = 0.0
            for t in reversed(range(N)):
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - is_terminals[t]) - values[t]
                gae = delta + self.gamma * self.lam * (1 - is_terminals[t]) * gae
                advantages[t] = gae

        returns = advantages + state_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        total_loss = 0.0
        value_loss = 0.0
        policy_loss = 0.0
        entropy_loss = 0.0

        dataset_size = len(rewards)
        for _ in range(self.k_epochs):
            perm = torch.randperm(dataset_size, device=self.device)
            for start_idx in range(0, dataset_size, self.batch_size):
                batch_idx = perm[start_idx:start_idx + self.batch_size]
                batch_states = old_states[batch_idx]
                batch_actions = old_actions[batch_idx]
                batch_logprobs = old_logprobs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                action_logits, state_values_batch = self.policy(batch_states)
                action_probs = F.softmax(action_logits, dim=-1)
                dist = Categorical(action_probs)
                logprobs = dist.log_prob(batch_actions)
                dist_entropy = dist.entropy().mean()

                ratios = torch.exp(logprobs - batch_logprobs)
                state_values_batch = state_values_batch.squeeze(-1)

                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss_batch = -torch.min(surr1, surr2).mean()
                value_loss_batch = self.MseLoss(state_values_batch, batch_returns)
                loss = policy_loss_batch + 0.5 * value_loss_batch - self.entropy_coef * dist_entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                value_loss += value_loss_batch.item()
                policy_loss += policy_loss_batch.item()
                entropy_loss += dist_entropy.item()



        return {
            'total': total_loss / self.k_epochs,
            'value': value_loss / self.k_epochs,
            'policy': policy_loss / self.k_epochs,
            'entropy': entropy_loss / self.k_epochs
        }

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

def train(agent, num_episodes, update_timestep=2000, checkpoint_dir='checkpoints/ppo', resume_from_checkpoint=False, checkpoint_interval=1000, num_envs=4):
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime
    writer = SummaryWriter(log_dir=f'runs/ppo_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    def make_env():
        import wrapped_flappy_bird as game
        return game.GameState()

    envs = VecEnv([make_env for _ in range(num_envs)])
    memory = Memory()

    all_rewards = []        # flat list of per-episode total rewards
    interval_rewards = []   # list of (episode, greedy_eval_reward)
    episode_reward = [0.0] * num_envs   # per-env accumulator
    max_game_score = float('-inf')
    total_episodes = 0
    timestep = 0
    flap_count = 0
    total_actions = 0

    # Resume from checkpoint if requested
    start_episode = 0
    if resume_from_checkpoint:
        latest = get_latest_checkpoint(checkpoint_dir)
        if latest:
            start_episode, saved_rewards = load_checkpoint(latest, agent)
            all_rewards = saved_rewards
            total_episodes = start_episode
            print(f"Resumed from episode {start_episode}")

    # Get initial observations and preprocess
    obs_list = envs.reset()
    state_frames = [deque(maxlen=4) for _ in range(num_envs)]
    for i in range(num_envs):
        frame = agent.preprocess(obs_list[i])
        for _ in range(4):
            state_frames[i].append(frame)

    while total_episodes < num_episodes:
        # Build states from frame stacks
        states = [np.stack(list(state_frames[i]), axis=0) for i in range(num_envs)]

        actions = []
        logprobs = []
        for i in range(num_envs):
            action, logprob = agent.select_action(states[i])
            actions.append(action)
            logprobs.append(logprob)
            flap_count += (action == 1)
            total_actions += 1

        action_vecs = [[0, 1] if a == 1 else [1, 0] for a in actions]

        # Frame skipping: repeat each action for 4 frames
        # This makes each decision more impactful and increases probability of passing pipes
        frame_skip = 4
        accumulated_rewards = [0.0] * num_envs
        env_done = [False] * num_envs
        skip_actions = list(action_vecs)
        for skip_step in range(frame_skip):
            obs_list, rewards_skip, dones_skip = envs.step(skip_actions)
            for i in range(num_envs):
                if not env_done[i]:
                    accumulated_rewards[i] += rewards_skip[i]
                    if dones_skip[i]:
                        env_done[i] = True
            # Send no-op for done envs on remaining skip steps
            skip_actions = [skip_actions[i] if not env_done[i] else [1, 0] for i in range(num_envs)]
        rewards_step = accumulated_rewards
        dones = env_done

        for i in range(num_envs):
            frame = agent.preprocess(obs_list[i])
            state_frames[i].append(frame)

            memory.states.append(torch.from_numpy(states[i]).float())
            memory.actions.append(torch.tensor(actions[i]))
            memory.logprobs.append(logprobs[i])
            # Reward shaping: reduce survival bonus to prevent "never flap" optimum
            # Raw: survival=0.1, pipe=1.0, death=-1.0
            # Shaped: survival=0.01, pipe=1.0, death=-1.0
            if rewards_step[i] > 0.5:
                shaped_reward = 1.0
            elif rewards_step[i] < -0.5:
                shaped_reward = -1.0
            else:
                shaped_reward = 0.01
            memory.rewards.append(shaped_reward)
            memory.is_terminals.append(dones[i])

            episode_reward[i] += rewards_step[i]  # Raw reward for logging

            if dones[i]:
                total_episodes += 1
                all_rewards.append(episode_reward[i])
                if episode_reward[i] > max_game_score:
                    max_game_score = episode_reward[i]
                episode_reward[i] = 0.0

                writer.add_scalar('Episode Reward', all_rewards[-1], total_episodes)

                if total_episodes % 100 == 0:
                    avg = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)
                    flap_rate = flap_count / max(1, total_actions)
                    print(f"Episode: {total_episodes}, Steps: {timestep}, AvgReward100: {avg:.2f}, MaxGameScore: {max_game_score:.2f}, FlapRate: {flap_rate:.2f}")

                # Checkpoint + interval greedy evaluation
                if total_episodes % checkpoint_interval == 0:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    save_checkpoint(checkpoint_dir, total_episodes, agent, all_rewards, timestamp)
                    print(f"  Checkpoint saved at episode {total_episodes}")
                    mean_score, _ = evaluate(agent, 10)
                    interval_rewards.append((total_episodes, mean_score))
                    print(f"  Greedy eval mean score: {mean_score:.2f}")

                # On crash, GameState auto-resets; the returned obs IS the new game's first frame
                frame = agent.preprocess(obs_list[i])
                state_frames[i] = deque(maxlen=4)
                for _ in range(4):
                    state_frames[i].append(frame)

                if total_episodes >= num_episodes:
                    break

        timestep += num_envs

        if len(memory.states) >= update_timestep:
            losses = agent.update(memory, num_envs=num_envs)
            memory.clear_memory()
            if losses:
                writer.add_scalar('Total Loss', losses['total'], timestep)
                writer.add_scalar('Value Loss', losses['value'], timestep)
                writer.add_scalar('Policy Loss', losses['policy'], timestep)
                writer.add_scalar('Entropy', losses['entropy'], timestep)

    envs.close()
    writer.close()
    return all_rewards, interval_rewards

def evaluate_worker(state_dict, num_episodes):
    import os, warnings
    warnings.filterwarnings('ignore')
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import cv2
    from collections import deque
    import sys
    sys.path.append("flappy_bird_repo/game/")
    import wrapped_flappy_bird as game

    def preprocess(image):
        image = image.transpose(1, 0, 2)  # pygame surfarray (W,H,C) -> (H,W,C)
        image = cv2.cvtColor(cv2.resize(image, (80, 80)), cv2.COLOR_BGR2GRAY)
        return (image / 255.0).astype(np.float32)

    model = ActorCritic()
    model.load_state_dict(state_dict)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    game_state = game.GameState()
    scores = []
    for _ in range(num_episodes):
        state_frames = deque(maxlen=4)
        for _ in range(4):
            do_nothing = [1, 0]
            image, _, _ = game_state.frame_step(do_nothing)
            state_frames.append(preprocess(image))
        state = np.stack(state_frames, axis=0).astype(np.float32)

        score = 0.0
        while True:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            action_probs, _ = model(state_tensor)
            action_probs = F.softmax(action_probs, dim=-1)
            action = torch.argmax(action_probs, dim=-1).item()
            action_vec = [0, 1] if action == 1 else [1, 0]
            image, reward, terminal = game_state.frame_step(action_vec)
            score += float(reward)

            if terminal:
                break
            state_frames.append(preprocess(image))
            state = np.stack(state_frames, axis=0)
        scores.append(score)
    return scores

def evaluate(agent, num_episodes=1000):
    state_dict = {k: v.cpu() for k, v in agent.policy.state_dict().items()}
    
    scores = evaluate_worker(state_dict, num_episodes)
    
    return np.mean(scores), np.max(scores)