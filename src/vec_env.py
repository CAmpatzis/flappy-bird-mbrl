import multiprocessing as mp
import numpy as np
import cloudpickle
import os


class CloudpickleWrapper:
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        self.x = cloudpickle.loads(ob)


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    # Suppress C-level warnings (e.g. libpng iCCP) by redirecting fd 2 to /dev/null
    import warnings
    warnings.filterwarnings('ignore')
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 2)
    os.close(_devnull)
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            # GameState.frame_step returns (image, reward, terminal)
            image, reward, terminal = env.frame_step(data)
            remote.send((image, reward, terminal))
        elif cmd == 'reset':
            # GameState has no reset(); send do-nothing to get initial frame
            image, _, _ = env.frame_step([1, 0])
            remote.send(image)
        elif cmd == 'close':
            remote.close()
            break
        else:
            raise NotImplementedError(f"Unknown command: {cmd}")


class VecEnv:
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        self.n_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.n_envs)])
        self.ps = [
            mp.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones = zip(*results)
        return list(obs), list(rews), list(dones)

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return [remote.recv() for remote in self.remotes]

    def reset_one(self, idx):
        self.remotes[idx].send(('reset', None))
        return self.remotes[idx].recv()

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
