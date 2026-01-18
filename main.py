# Choose the game we want to work with
import Connect4 as Game

# Optimize PyTorch
import os
import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(1)

import time
import copy
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from Monte_Carlo_Search_and_Sampling import MonteCarlo
from Network_Wrapper import NNetWrapper
from Replay_Buffer import Buffer


def full_training():

    # Define all useful game constants (must be included in "Game" module)

    # Constants used in training
    replay_buffer_size = Game.replay_buffer_size  # Number of state-policy-value training samples
    replay_warmup_ratio = Game.replay_warmup_ratio  # Amount of buffer to be pre-filled before training begins
    warmup_episodes = Game.warmup_episodes  # Initial training episodes to gradually raise learning rate
    eval_interval = Game.eval_interval   # Every interval we evaluate current net vs. reference network
    eval_games = Game.eval_games   # Compare networks with this many games
    promote_thresh = Game.promote_thresh  # Save current network if this % of its games are won

    # Network initial parameters
    net_forward_batch_size = Game.net_forward_batch_size  # Two different batch sizes for simple evaluation and training
    net_training_batch_size = Game.net_training_batch_size
    lr_max = Game.net_max_learning_rate
    lr_min = Game.net_min_learning_rate
    num_training_samples = Game.num_training_samples  # No. of samples to train on each episode
    net_epochs = Game.net_epochs
    net_l2_reg = Game.net_l2_reg  # L2 reg ensures no network parameter becomes to big too early
    value_loss_weight = Game.value_loss_weight  # Optionally weigh value loss during training

    # Constants used in Monte Carlo Search
    num_games = Game.num_games  # Number of self-play games completed each episode
    simulations = Game.simulations
    determ_depth = Game.determ_depth  # After this depth in self-play game, we only play best moves
    rand_pos_depth = Game.rand_pos_depth  # Max depth at which to collect random starting positions for self-play games
    alpha = Game.alpha  # Dirichlet noise parameter
    epsilon = Game.epsilon  # Dirichlet noise parameter

    # Other constants, not game specific
    kl_ema = None
    EMA_ALPHA = 0.1
    ADJUST_RATE = 0.1   # Prevents instability
    net_promote_counts = 0  # Count promotions (evals where training net wins most of overall games)

    replay_warmup = int(replay_buffer_size * replay_warmup_ratio)
    
    model = Game.nn_model()
    if os.path.exists(Game.env.model_path):
        model.load_state_dict(torch.load(Game.env.model_path))
        print()
        print("Loaded existing network")

    net = NNetWrapper(
        model,
        forward_batch_size = net_forward_batch_size,
        training_batch_size = net_training_batch_size,
        epochs = net_epochs,
        lr = lr_max,
        grad_clip = 10.0,
        l2_reg = net_l2_reg,
        value_loss_weight = value_loss_weight,
        warmup_steps=(net_epochs * warmup_episodes)
    )

    # Keep list of 3 previous networks for reference
    reference_nets = []
    for _ in range(3):
        tmp_net = copy.deepcopy(net.nnet)
        tmp_net.eval()
        reference_nets.append(tmp_net)

    mcts = MonteCarlo(
        Game.env,
        simulations = simulations,
        num_games = num_games,
        determ_depth = determ_depth,
        rand_pos_depth = rand_pos_depth,
        alpha = alpha,
        epsilon = epsilon
    )

    REPLAY_BUFFER = Buffer(
        capacity = replay_buffer_size,
        state_shape = Game.env.FULL_BOARD_SHAPE,
        policy_shape = (Game.env.TOT_ACTIONS,)
    )

    # Store current net win rates for each evaluation
    nets_history = [[] for _ in reference_nets]
    eval_intervals_history = []

    # Make sure replay buffer is sufficiently full before starting
    print()
    print("⏳ Warming up replay buffer...")
    t1 = time.time()
    warmup_len = 0
    while warmup_len < replay_warmup:
        training_data = mcts.batch_self_play(net=net)
        REPLAY_BUFFER.push(training_data)
        warmup_len += len(training_data)
    print()
    print(f"Collected {warmup_len} samples")
    print(f"Filling replay buffer took {time.time()-t1:.1f} sec")


    # Begin training (maximum of 5000 episodes)
    for episode in range(1, 5000):

        print(f"\n=== Episode {episode} ===")

        # Collect training samples & train network
        t = time.time()
        training_data = mcts.batch_self_play(net=net)
        REPLAY_BUFFER.push(training_data)
        print(f"🔄 Self play took {time.time()-t:.1f} sec")
        print(f"Collected {len(training_data)} samples")
        
        batch = min(num_training_samples, len(REPLAY_BUFFER))
        batch_states, batch_policies, batch_values = REPLAY_BUFFER.sample(batch_size=batch)

        t = time.time()
        policy_loss, value_loss, net_KL = net.train_step(
            batch_states, batch_policies, batch_values
        )

        print(f"📈 Training took {time.time()-t:.1f} sec")

        print(
            "Network training results: "
            f" Policy Loss: {policy_loss:.4f} |"
            f" Value Loss: {value_loss:.4f} |"
            f" Learning: {net_KL if kl_ema is None else kl_ema:.4f}"
        )

        # Optional: stopping condition if losses low enough
        stop_for_low_losses = policy_loss < 0.5 and value_loss < 0.5


        if episode > warmup_episodes:
            
            # Self-play move temperature and dirichlet noise are decreased according to search entropy
            mcts.tau_move = np.clip(
                0.1 + (1.0 - 0.1) * (mcts.H_search / 2.0),
                0.1,
                1.0
            )
            mcts.epsilon = np.clip(
                0.05 + (epsilon - 0.05) * (mcts.H_search / 2.0),
                0.05, 
                epsilon
            )

            # Smoothly adjust saved KL tracker to account for learning of this batch
            if kl_ema is None:
                kl_ema = net_KL
            else:
                kl_ema = EMA_ALPHA * net_KL + (1 - EMA_ALPHA) * kl_ema

            if episode % 5 == 0:
                # Net learning rate is dependent on how much network learned on most recent training
                KL_TARGET = 0.002 + 0.02 * mcts.H_search
                ratio = kl_ema / KL_TARGET
                ratio = max(0.25, min(4.0, ratio))

                if 0.8 < ratio < 1.25:
                    factor = 1.0
                else:
                    factor = ratio ** (-ADJUST_RATE)
                    factor = max(0.8, min(1.25, factor))

                net.lr *= factor
                net.lr = max(lr_min, min(lr_max, net.lr))

                for g in net.optimizer.param_groups:
                    g["lr"] = net.lr

            print(
                "Training loop parameters: "
                f" MCTS Entropy: {mcts.H_search:.4f} |"
                f" Net LR: {net.lr:.3e} |"
                f" Simulations: {mcts.simulations} |"
                f" Move temp: {mcts.tau_move:.4f}"
            )

            # Stopping condition: Training ends when learning decreases sufficiently
            if kl_ema < 1e-3 or stop_for_low_losses:
                print()
                print("Training converged")
                break
            
            # Every interval, compare current training net with reference networks
            if episode % eval_interval == 0:
                
                print()
                print(f"So far there have been {net_promote_counts} network promotions")
                print("Evaluating networks...")
                eval_win_rates = np.zeros(len(reference_nets))
                for index, ref_net in enumerate(reference_nets):
                    net_wins, ref_wins, draws = mcts.compare_networks(
                        net=net,
                        other_net=ref_net,
                        tests=eval_games
                    )

                    eval_win_rates[index] = net_wins + 0.5 * draws  # Draws are half-credit

                    print(
                        f"Eval vs ref: "
                        f" Wins: {net_wins:.1%} |"
                        f" Losses: {ref_wins:.1%}, |"
                        f" Draws: {draws:.1%}"
                    )

                # Record minimum win rate, and maximum win rate index (worst opponent index)
                min_win_rate = np.min(eval_win_rates)
                worst_net = np.argmax(eval_win_rates)
                for index, winrate in enumerate(eval_win_rates):
                    nets_history[index].append(winrate)
                eval_intervals_history.append(episode // eval_interval)

                if min_win_rate > promote_thresh:
                    print(f"🏆 New network added to reference networks at index {worst_net}")
                    reference_nets[worst_net].load_state_dict(net.nnet.state_dict())
                    reference_nets[worst_net].eval()
                    torch.save(net.nnet.state_dict(), Game.env.model_path)
                    net_promote_counts += 1

        else:   # Network still being warmed up with gradually increasing lr
            if episode == warmup_episodes:
                net.warmup = False
            print("Training warmup; gradually increasing learning rate")

    # TRAINING LOOP ENDED
    # Start tournament to find best network among current net + ref networks
    print("Finding best network")
    all_nets = [net] + reference_nets
    net_scores = np.zeros(len(all_nets))
    for n1, net1 in enumerate(all_nets):
        for n2, net2 in enumerate(all_nets):
            if n1 == n2:
                continue

            n1_wins, n2_wins, draws = mcts.compare_networks(
                net=net1,
                other_net=net2,
                tests=eval_games
            )

            net_scores[n1] += (n1_wins + 0.5 * draws)
            net_scores[n2] += (n2_wins + 0.5 * draws)

    for idx, score in enumerate(net_scores):
        print(f"Network {idx} scored {score:.1%}")
    best_net = all_nets[np.argmax(net_scores)]
    torch.save(best_net.state_dict(), Game.env.model_path)

        
    # Display loss progress
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_ylim(0, 1)
    for history in nets_history:
        point_colors = ['red' if winrate > promote_thresh else 'blue' for winrate in history]
        ax.scatter(eval_intervals_history, history, c=point_colors, s=25)
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Win rate")
    ax.grid(True)
    ax.set_title("% New Network Wins vs. Each Reference Network")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # Control whether training loop runs
    # full_training()

    model = Game.nn_model()
    if os.path.exists(Game.env.model_path):
        model.load_state_dict(torch.load(Game.env.model_path))
        print("Loaded existing network")
    net = NNetWrapper(model)
    net.eval()
    mcts = MonteCarlo(Game.env)

    # Just to get network ready
    mcts.single_search(Game.env.START_BOARD, 1, net)

    AI_player = lambda board, player: mcts.single_search(board, player, net)
    Game.play_game(AI_player, human_player=1)