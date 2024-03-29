import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
# if (torch.cuda.is_available()):
#     device = torch.device('cuda:0')
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")
# print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.masks = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.masks[:]
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, shape, action_dim):
        super(ActorCritic, self).__init__()

        # self.conv = nn.Sequential(
        #     nn.Conv1d(shape[0], 32, 3),
        #     nn.MaxPool1d(3, 2),
        #     nn.ReLU(),
        #     nn.Conv1d(32, 64, 3),
        #     nn.MaxPool1d(3, 2),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 128, 3),
        #     nn.MaxPool1d(3, 2),
        #     nn.ReLU(),
        #     nn.Conv1d(128, 128, 3),
        #     nn.ReLU(),
        #     nn.Conv1d(128, 128, 3),
        #     nn.ReLU()
        # )

        # self.conv = nn.Sequential(
        #     nn.Conv1d(shape[0], 128, 5),
        #     nn.Tanh(),
        #     nn.Conv1d(128, 128, 5),
        #     nn.Tanh(),
        # )
        #
        # out_size = self._get_conv_out(shape)
        #
        # # actor
        # self.actor = nn.Sequential(
        #     nn.Linear(out_size, 512),
        #     nn.Tanh(),
        #     nn.Linear(512, action_dim),
        #     nn.Softmax(dim=-1)
        # )
        # # critic
        # self.critic = nn.Sequential(
        #     nn.Linear(out_size, 512),
        #     nn.Tanh(),
        #     nn.Linear(512, 1)
        # )

        self.actor = nn.Sequential(
            nn.Linear(shape, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(shape, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape[:]))
        return int(np.prod(o.size()))

    def forward(self):
        raise NotImplementedError

    def act(self, state, mask, if_train):
        # conv_out = self.conv(state).view(1, -1)
        #
        # action_probs = self.actor(conv_out)

        action_probs = self.actor(state)
        s = mask.sum()
        l = ((action_probs * (1 - mask)).sum() / s)
        action_probs = (action_probs + l) * mask
        if if_train:
            dist = Categorical(action_probs)

            action = dist.sample()
            action_logprob = dist.log_prob(action)
            # state_val = self.critic(conv_out)
            state_val = self.critic(state)

            return action.detach(), action_logprob.detach(), state_val.detach()
        else:
            max_index = torch.argmax(action_probs)
            return max_index

    def evaluate(self, state, action, masks):
        # conv_out = self.conv(state).view(state.size()[0], -1)
        #
        # action_probs = self.actor(conv_out)
        action_probs = self.actor(state)
        s = masks.sum(dim=1)
        l = ((action_probs * (1 - masks)).sum(dim=1) / s).unsqueeze(1)
        action_probs = (action_probs + l) * masks
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        # state_values = self.critic(conv_out)
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, shape, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(shape, action_dim).to(device)
        # self.optimizer = torch.optim.Adam([
        #     {'params': self.policy.conv.parameters(), 'lr': lr},
        #     {'params': self.policy.actor.parameters(), 'lr': lr},
        #     {'params': self.policy.critic.parameters(), 'lr': lr}
        # ])
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(shape, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.update_times = 0

    def select_action(self, state, mask, dev, if_train):

        if if_train:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(dev)
                mask = mask.to(dev)
                action, action_logprob, state_val = self.policy_old.act(state, mask, if_train)
                return action, action_logprob, state_val
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(dev)
                mask = mask.to(dev)
                action = self.policy_old.act(state, mask, if_train)
                return action

    def update(self, batch_size, writer):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_masks = torch.squeeze(torch.stack(self.buffer.masks, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        total_samples = old_states.size()[0]
        indices = np.arange(total_samples)
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            np.random.shuffle(indices)
            index = 0
            while index < total_samples:
                self.update_times += 1
                batch_indices = indices[index:index + batch_size]
                if len(batch_indices) == 1:
                    break
                batch_old_states = old_states[batch_indices].to(device)
                batch_old_masks = old_masks[batch_indices].to(device)
                batch_old_actions = old_actions[batch_indices].to(device)
                batch_old_logprobs = old_logprobs[batch_indices].to(device)
                batch_advantages = advantages[batch_indices].to(device)
                batch_rewards = rewards[batch_indices].to(device)

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(batch_old_states, batch_old_actions, batch_old_masks)

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - batch_old_logprobs.detach())

                # Finding Surrogate Loss
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages

                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, batch_rewards) - 0.01 * dist_entropy

                # record loss
                writer.add_scalar("episode/loss", torch.sum(loss).item(), self.update_times)

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

                index += batch_size

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
