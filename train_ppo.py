import os
from datetime import datetime
import pandas as pd
from Environment import StockTradingEnv_Test_Onetic
from PPO import PPO
from sampler_asyn import MemorySampler
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp

tic = '600585.SH'
log_dir = f'./log/ppo_{tic}'


class args(object):
    num_workers = 2
    mini_batch_size = 2
    batch_size = 4

    update_batch_size = 128

    max_step = 60
    state_space = 32

    val_episodes = 30
    save_model_freq = 50

    num_iterations = 100000

    gamma = 0.99

    K_epochs = 10
    eps_clip = 0.2

    lr_actor = 0.0001  # learning rate for actor network
    lr_critic = 0.0001  # learning rate for critic network

    initial_amount = 1000000
    device = 'cpu'

    train_file = f"./dataset/2006-2024/train_data_{tic}.csv"
    val_file = f"./dataset/2006-2024/test_data_{tic}.csv"

    train_tickers = list(pd.read_csv(train_file)['tic'].unique())
    val_tickers = list(pd.read_csv(val_file)['tic'].unique())

    train_data = pd.read_csv(train_file)
    train_data["date"] = pd.to_datetime(train_data["date"], format="%Y-%m-%d")
    train_data['index'] = train_data.groupby('tic').cumcount()
    train_data.set_index(['index'], inplace=True)
    train_data.index.names = [None]

    val_data = pd.read_csv(val_file)
    val_data["date"] = pd.to_datetime(val_data["date"], format="%Y-%m-%d")
    val_data['index'] = val_data.groupby('tic').cumcount()
    val_data.set_index(['index'], inplace=True)
    val_data.index.names = [None]

    df = train_data

    action_dim = 3


def train(writer):

    print("============================================================================================")

    sampler = MemorySampler(args)

    # ppo_agent.load("log/ppo_10days/models/model_399.pth")

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    ################### checkpointing ###################
    directory = log_dir + '/models'
    if not os.path.exists(directory):
        os.makedirs(directory)
    #####################################################
    cnt = 0

    max_reward = 0

    for i_iteration in range(args.num_iterations):
        memory = sampler.sample(ppo_agent)
        batch = memory.sample()
        rewards = list(batch.reward)
        done = list(batch.done)
        # start_idx = 0
        # for i in range(len(done)):
        #     if done[i] == 1:
        #         end_idx = i
        #         episode_rewards = rewards[i]
        #         length = end_idx-start_idx+1
        #         avg_rewards = episode_rewards / length
        #         for j in range(length):
        #             rewards[start_idx+j] = avg_rewards
        #         start_idx = end_idx + 1

        # ranges = []
        # start_idx = None
        # for i, action in enumerate(list(batch.action)):
        #     if action == 1 and start_idx is None:
        #         start_idx = i
        #     elif action == 2 and start_idx is not None:
        #         ranges.append((start_idx, i))
        #         start_idx = None
        #
        # # 更新奖励列表
        # for start, end in ranges:
        #     action2_reward = rewards[end]
        #     num_actions = end - start + 1
        #     avg_reward = action2_reward / num_actions
        #     for i in range(start, end + 1):
        #         rewards[i] = avg_reward

        ppo_agent.buffer.rewards.extend(rewards)
        ppo_agent.buffer.state_values.extend(list(batch.value))
        ppo_agent.buffer.is_terminals.extend(done)
        ppo_agent.buffer.actions.extend(list(batch.action))
        ppo_agent.buffer.states.extend(batch.state)
        ppo_agent.buffer.masks.extend(batch.mask)
        ppo_agent.buffer.logprobs.extend(list(batch.logproba))

        cum_reward = batch.cum_reward
        is_terminal = batch.done
        steps = batch.steps

        for i in range(len(cum_reward)):
            if is_terminal[i]:
                cnt += 1
                writer.add_scalar("episode/reward", round(cum_reward[i]), cnt)
                # writer.add_scalar("episode/steps", steps[i], cnt)

        # update PPO agent
        print(
            "--------------------------------------------------------------------------------------------")
        print("iteration {}: total_rewards {}.".format(i_iteration+1, round(sum(batch.reward))))
        writer.add_scalar("iteration/train_reward", round(sum(batch.reward)), i_iteration+1)
        ppo_agent.update(args.update_batch_size, writer)

        reward = val(ppo_agent, args.val_data)
        writer.add_scalar("iteration/val_reward", reward, i_iteration+1)

        if (i_iteration+1) % args.save_model_freq == 0 and i_iteration > 100:
            print(
                "--------------------------------------------------------------------------------------------")
            print("saving model...")
            ppo_agent.save(directory + f"/model_{i_iteration}.pth")
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print(
                "--------------------------------------------------------------------------------------------")
        if reward > max_reward:
            max_reward = reward
            print("--------------------------------------------------------------------------------------------")
            print("saving model...")
            ppo_agent.save(directory + f"/best_model_{i_iteration}.pth")
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print(
                "--------------------------------------------------------------------------------------------")
        # if reward > -200000 and reward != 0:
        #     ppo_agent.save(directory + f"/model_{i_iteration}.pth")

    # print total training time
    sampler.close()
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


def val(ppo_agent, val_data):

    env_kwargs = {
        "initial_amount": args.initial_amount,
        "state_space": args.state_space,
        "tickers": args.val_tickers,
        "max_step": args.max_step
    }
    env = StockTradingEnv_Test_Onetic(df=val_data, **env_kwargs)

    print("--------------------------------------------------------------------------------------------")


    state = env.reset()
    done = False

    while not done:
        mask = env.action_mask()

        # select action with policy
        action = ppo_agent.select_action(state, mask, args.device, False)
        state, rewards, trades, done, cum_reward, steps = env.step(action)

    # clear buffer
    ppo_agent.buffer.clear()

    print("Validation reward : {} \t\t Days: {} \t\t Trades: {}".format(
        round(cum_reward-args.initial_amount), steps, trades))

    print("============================================================================================")

    return round(cum_reward-args.initial_amount)


if __name__ == '__main__':
    mp.set_start_method('spawn')

    # initialize a PPO agent
    ppo_agent = PPO(args.state_space, args.action_dim, args.lr_actor, args.lr_critic, args.gamma, args.K_epochs, args.eps_clip)
    writer = SummaryWriter(logdir=log_dir, comment="-stock trading")
    train(writer)






