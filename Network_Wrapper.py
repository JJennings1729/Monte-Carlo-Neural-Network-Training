import torch
import torch.nn as nn
import torch.nn.functional as F


class NNetWrapper(nn.Module):

    def __init__(
            self,
            neural_net,
            forward_batch_size=64,
            training_batch_size=64,
            epochs=5,
            lr=1e-3,
            grad_clip=5.0, 
            l2_reg=1e-4,
            value_loss_weight=1.0,
            warmup_steps=3000
        ):

        super().__init__()

        self.nnet = neural_net

        # Hyperparams
        self.lr = lr
        self.forward_batch_size = forward_batch_size
        self.training_batch_size = training_batch_size
        self.grad_clip = grad_clip
        self.epochs = epochs
        self.loss_weight = value_loss_weight

        self.optimizer = torch.optim.AdamW(
            self.nnet.parameters(),
            lr=self.lr,
            weight_decay=l2_reg
        )

        # Scheduler for warmup
        self.warmup = True
        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / warmup_steps)
        )


    def forward(self, x, legal_moves=None):
        """
        Forward pass into wrapped neural net, evaluated in chunks
        of self.batch_size to control memory usage.
        """
        # Fast path: small batch
        if x.size(0) <= self.forward_batch_size:
            return self.nnet(x, legal_moves)

        pis = []
        values = []

        for start in range(0, x.size(0), self.forward_batch_size):
            end = start + self.forward_batch_size

            x_chunk = x[start:end]

            if legal_moves is not None:
                legal_chunk = legal_moves[start:end]
            else:
                legal_chunk = None

            pi, v = self.nnet(x_chunk, legal_chunk)
            pis.append(pi)
            values.append(v)

        return (
            torch.cat(pis, dim=0),
            torch.cat(values, dim=0),
        )


    # Training function (policy + value loss, LR scheduler, clipping)
    def train_step(self, batch_states, batch_policies, batch_values):
        """
        Trains the neural network on a batch and returns:
        - final_policy_loss
        - final_value_loss
        - KL value (how much network learned)
        """

        device = next(self.parameters()).device

        batch_states = batch_states.to(device)
        batch_policies = batch_policies.to(device)
        batch_values = batch_values.to(device).view(-1, 1)

        N = batch_states.size(0)

        # We save initial policies to estimate network learning later
        self.eval()
        with torch.no_grad():
            old_logits, _ = self(batch_states)
            old_logp = torch.log(old_logits + 1e-8)
            old_p = old_logits

        self.train()
        final_policy_loss, final_value_loss = 0.0, 0.0

        for epoch in range(self.epochs):
            indices = torch.randperm(N, device=device)
            total_policy_loss = 0.0
            total_value_loss = 0.0

            for start in range(0, N, self.training_batch_size):
                end = start + self.training_batch_size
                idx = indices[start:end]

                states = batch_states[idx]
                target_pi = batch_policies[idx]
                target_v = batch_values[idx]

                # Forward pass
                pred_pi, pred_v = self(states)
                pred_v = pred_v.view(-1, 1)

                # Losses
                policy_loss = -(target_pi * torch.log(pred_pi + 1e-8)).sum(dim=1).mean()
                value_loss = F.mse_loss(pred_v, target_v)
                loss = policy_loss + self.loss_weight * value_loss

                # Backprop
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), self.grad_clip)

                self.optimizer.step()
                if self.warmup:
                    self.warmup_scheduler.step()

                # Accumulate epoch loss
                total_policy_loss += policy_loss.item() * states.size(0)
                total_value_loss += value_loss.item() * states.size(0)

            # Epoch-level average losses
            final_policy_loss = total_policy_loss / N
            final_value_loss = total_value_loss / N

        # KL evaluation (measures network learning, old || new)
        self.eval()
        with torch.no_grad():
            pred_pi, pred_v = self(batch_states)
            pred_v = pred_v.view(-1, 1)
            new_logp = torch.log(pred_pi + 1e-8)
            kl = (old_p * (old_logp - new_logp)).sum(dim=1).mean().item()

        return final_policy_loss, final_value_loss, kl