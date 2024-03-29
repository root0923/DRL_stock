import torch.multiprocessing as mp
import torch
from collections import namedtuple
from Environment import StockTradingEnv_1D

Transition = namedtuple('Transition',
                        ('state', 'value', 'action', 'logproba', 'done', 'reward', 'mask', 'cum_reward', 'steps'))


def make_env(df, initial_amount, state_space, tickers, max_step):
    env_kwargs = {
        "initial_amount": initial_amount,
        "state_space": state_space,
        "tickers": tickers,
        "max_step": max_step
    }
    env = StockTradingEnv_1D(df=df, **env_kwargs)
    return env


class Episode(object):
    def __init__(self):
        self.episode = []

    def push(self, *args):
        self.episode.append(Transition(*args))

    def __len__(self):
        return len(self.episode)


class Memory(object):
    def __init__(self):
        self.memory = []
        self.num_episode = 0

    def push(self, epi: Episode):
        self.memory += epi.episode
        self.num_episode += 1

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


class EnvWorker(mp.Process):
    def __init__(self, df, remote, queue, lock, mini_batch_size, initial_amount, state_space, tickers, max_step):
        super(EnvWorker, self).__init__()
        self.remote = remote
        self.queue = queue
        self.lock = lock
        self.mini_batch_size = mini_batch_size

        self.env = make_env(df, initial_amount, state_space, tickers, max_step)

    def run(self):
        while True:
            command, policy = self.remote.recv()
            i_episode = 0
            if command == 'collect':
                while i_episode < self.mini_batch_size:
                    i_episode += 1
                    episode = Episode()
                    state = self.env.reset()
                    done = False
                    while not done:
                        with torch.no_grad():
                            mask = self.env.action_mask()
                            action, logproba, value = policy.select_action(state, mask, 'cpu', True)
                            action = action.data.cpu().numpy()
                            logproba = logproba.data.cpu().numpy()
                            value = value.data.cpu().numpy()

                        new_state, reward, trades, done, cum_reward, steps = self.env.step(action)

                        done = 1 if done else 0
                        episode.push(torch.tensor(state), torch.tensor(value), torch.tensor(action),
                                     torch.tensor(logproba), done, reward, mask, cum_reward, steps)
                        if done:
                            print("Reward : {} \t\t Trades: {} \t\t Days: {}".format(round(cum_reward), trades, steps))

                            with self.lock:
                                self.queue.put(episode)
                            break
                        state = new_state
            elif command == 'close':
                self.remote.close()
                self.env.close()
                break
            else:
                raise NotImplementedError()


class MemorySampler(object):
    def __init__(self, args):
        self.df = args.df
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.device = args.device

        self.queue = mp.Queue()
        self.lock = mp.Lock()

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_workers)])

        self.workers = [EnvWorker(self.df, remote, self.queue, self.lock, args.mini_batch_size, args.initial_amount, args.state_space, args.train_tickers, args.max_step)
                        for i, remote in enumerate(self.work_remotes)]
        for worker in self.workers:
            worker.daemon = True
            worker.start()
        for remote in self.work_remotes:
            remote.close()

    def sample(self, policy):
        policy.policy_old.to('cpu')
        memory = Memory()

        for remote in self.remotes:
            remote.send(('collect', policy))

        while memory.num_episode < self.batch_size:
            episode = self.queue.get(True)
            memory.push(episode)

        policy.policy_old.to(self.device)
        return memory

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for worker in self.workers:
            worker.join()
