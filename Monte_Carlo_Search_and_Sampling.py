import numpy as np
from numba import njit
import torch

# -----------------------------
# Lookup / insert using linear probing
# -----------------------------

TABLE_SIZE = 2 ** 22  # Must be a power of 2
EMPTY_HASH = np.uint64(0xFFFFFFFFFFFFFFFF)
TABLE_MASK = np.uint64(TABLE_SIZE - 1)

@njit
def hash_idx(keys_arr, key):

    """
    Returns:
    "key_found" (whether key is already in table)
    "key_idx" (either index of key or empty index which key can be inserted into)
    """

    h = np.uint64(key ^ (key >> 32))
    start = np.int32(h & TABLE_MASK)

    for i in range(32):  # hard bound = no spikes
        idx = np.int32((start + i + i*i) & TABLE_MASK)  # Quadratic probing
        k = keys_arr[idx]

        if k == key:
            return True, idx
        if k == EMPTY_HASH:
            return False, idx

    return False, np.int32(-1)


# -----------------------------
# Numba functions for Monte Carlo Search
# -----------------------------

@njit
def compute_result(winner, player):
    """
    Compute terminal value from the perspective of root player.

    Returns:
        +1.0 if player wins
        -1.0 if player loses
         0.0 if draw
    """
    if winner == 0:
        return 0.0
    elif winner == player:
        return 1.0
    else:
        return -1.0


@njit
def backpropagate(paths, values, W, N):
    """
    Insert corresponding values at each state-action index, from root to leaf.
    """
    P, D, _ = paths.shape
    for p in range(P):
        value = values[p]
        for d in range(D):
            state_idx, action_idx = paths[p, d]
            if state_idx < 0 or action_idx < 0:
                # Stop once path is empty
                break
            W[state_idx, action_idx] += value
            N[state_idx, action_idx] += 1
            value = -value


@njit
def puct_score(W, N, P, tot_visits, c_puct):
    Q = W / max(N, 1)
    U = c_puct * P * np.sqrt(tot_visits) / (1 + N)
    return Q + U


def batch_select_wrapper(
    make_hash, legal_actions, action_mask, step,  # Environment functions
    MAX_DEPTH, c_puct, VIRTUAL_LOSS  # Constants
):
    
    """
    Return numba function for MCTS selection step, provided the parameters above.
    """
    
    @njit
    def batch_select(
        boards, players, # Root states
        leaf_boards, leaf_players, leaf_paths, leaf_hashes, # Arrays to collect non-terminal leaves
        term_paths, term_results,  # Arrays to collect terminal paths and values to backprop
        max_num_leaves, max_num_term, # Limits to impose on leaf/terminal states collections
        keys, wins, visits, priors, v_loss,  # MCTS table flat arrays
        tmp_board, tmp_actions, tmp_path  # Arrays to temporarily save board/actions/root-leaf path indices
    ):
        
        """
        Fill empty arrays leaf_boards, leaf_players, leaf_paths, term_paths, and term_results.
        Leaf_boards and leaf_players will contain leaf board/player states to be evaluated by network.
        Leaf_paths will contain lists of [hash_idx, action_idx] for each root-to-leaf path.
        Term_paths will contain lists of [hash_idx, action_idx] for each root-to-terminal state path.
        Term_results records results of terminal states, which can be immediately backpropagated after selection.
        Function returns total number of leaves to be evaluated and total terminal states encountered.
        """

        B = boards.shape[0]
        num_leaves = 0
        num_term = 0
        attempts = 0
        max_attempts = max_num_leaves + max_num_term

        while num_leaves < max_num_leaves and num_term < max_num_term and attempts < max_attempts:
            root = attempts % B
            attempts += 1

            for x in range(tmp_board.shape[0]):
                tmp_board[x] = boards[root, x]
            board = tmp_board
            player = players[root]
            done = False
            winner = 0
            root_player = player  # Save player at root for backpropagation

            h = np.uint64(make_hash(board, player))
            root_found, root_idx = hash_idx(keys, h)
            root_idx = np.int32(root_idx)
            if not root_found or root_idx < 0:
                continue
                    
            tmp_path[:] = -1
            node_idx = root_idx
            depth = 0

            # Find leaf, starting from this root.
            # Any "break" from this loop will end up rewriting the path from this root.
            while depth < MAX_DEPTH:
                
                n_actions = legal_actions(board, player, tmp_actions)
                if n_actions == 0:
                    # Terminal; no valid actions
                    # Assume this is a draw (this must be edited if not)
                    term_paths[num_term, :] = tmp_path[:]
                    term_results[num_term] = compute_result(winner, root_player)
                    num_term += 1
                    break

                # Select child of this node.
                # First determine if there are any unexpanded children (i.e. children not in table):
                has_unexpanded = False
                table_full = False
                for i in range(n_actions):
                    a = tmp_actions[i]
                    mask = np.uint64(action_mask(a, board, player))
                    child_hash = h ^ mask
                    child_found, child_idx = hash_idx(keys, child_hash)
                    child_idx = np.int32(child_idx)
                    if child_idx < 0:
                        table_full = True
                        break
                    if not child_found:
                        has_unexpanded = True
                        break

                # Failsafe if table is full (treat parent state as a leaf)
                if table_full:
                    # First save the path to leaf_paths
                    leaf_paths[num_leaves, :] = tmp_path[:]
                    # Now save board, player, index, hash to leaf arrays
                    leaf_boards[num_leaves, :] = board[:]
                    leaf_players[num_leaves] = player
                    leaf_hashes[num_leaves] = h
                    num_leaves += 1
                    break

                # Now select child (if any unexpanded children, selection must be from them):
                tot_visits = np.float32(0)
                for a in range(visits.shape[1]):
                    # Include virtual loss temporarily
                    eff_visits = visits[node_idx, a] + v_loss[node_idx, a]
                    tot_visits += eff_visits
                tot_visits = np.float32(max(tot_visits, 1.0))  # Guard against zero tot_visits
                best_score = -1e10
                best_child_idx = np.int32(-1)
                best_action_idx = np.int32(-1)
                best_mask = np.uint64(0)

                for i in range(n_actions):
                    a = tmp_actions[i]
                    mask = action_mask(a, board, player)
                    child_hash = h ^ mask
                    child_found, child_idx = hash_idx(keys, child_hash)

                    # If choosing from unexpanded children, assign random scores to those children.
                    # Else score according to PUCT formula. Select the highest score.
                    if has_unexpanded:
                        if child_found:
                            continue
                        score = np.float32(np.random.random())
                    else:
                        w = wins[node_idx, a]
                        n = visits[node_idx, a]
                        p = priors[node_idx, a]
                        v = v_loss[node_idx, a]
                        # Incorporate virtual loss into stats when calculating PUCT score
                        w_eff, n_eff = w - v, n + v
                        score = puct_score(w_eff, n_eff, p, tot_visits, c_puct)

                    if score > best_score:
                        best_score = score
                        best_child_idx = child_idx
                        best_action_idx = a
                        best_mask = mask

                # Failsafe if no valid child selected: remove virtual loss and break this selection loop
                if best_child_idx < 0:
                    for x in range(depth):
                        idx, a = tmp_path[x]
                        if idx >= 0 and v_loss[idx, a] >= VIRTUAL_LOSS:
                            v_loss[idx, a] -= VIRTUAL_LOSS
                    break
                
                # Valid child selected + table not full → we will step to this child.
                # Step to the selected child with best_action_idx
                player, done, winner = step(board, player, done, winner, best_action_idx, tmp_board)
                board = tmp_board
                h ^= best_mask  # Both board change + player switch are updated by ^mask

                # Any edge we choose must have virtual loss added
                v_loss[node_idx, best_action_idx] += VIRTUAL_LOSS

                # Record selected indices in path for later backpropagation
                tmp_path[depth, 0] = node_idx
                tmp_path[depth, 1] = best_action_idx
                node_idx = best_child_idx

                # Check if selected node is terminal
                if done:
                    term_paths[num_term, :] = tmp_path[:]
                    term_results[num_term] = compute_result(winner, root_player)
                    num_term += 1
                    break

                # Non-terminal leaf node handling
                # (If max depth is reached, this state is also treated as a leaf)
                if has_unexpanded or depth == MAX_DEPTH - 1:
                    # First save the path to leaf_paths
                    leaf_paths[num_leaves, :] =  tmp_path[:]
                    # Now save board, player, index, hash to leaf arrays
                    leaf_boards[num_leaves, :] = board[:]
                    leaf_players[num_leaves] = player
                    leaf_hashes[num_leaves] = h
                    num_leaves += 1
                    break

                depth += 1

        # Selection is complete; clear virtual loss from all visited indices
        for x in range(num_leaves):
            for y in range(leaf_paths.shape[1]):
                a, b = leaf_paths[x, y]
                if a >= 0 and b >= 0:
                    v_loss[a, b] -= VIRTUAL_LOSS
        for x in range(num_term):
            for y in range(term_paths.shape[1]):
                a, b = term_paths[x, y]
                if a >= 0 and b >= 0:
                    v_loss[a, b] -= VIRTUAL_LOSS

        return num_leaves, num_term
    

    return batch_select


def apply_temp(edge_visits, tau):
    """
    Convenient function to apply temperature to edge visit counts.
    """

    edge_visits = np.asarray(edge_visits, dtype=np.float32)

    # Deterministic: 100% chance of going with the best action
    if tau <= 0.0:
        out = np.zeros_like(edge_visits, dtype=np.float32)
        idx = np.argmax(edge_visits, axis=-1)
        if out.ndim == 1:
            out[idx] = 1.0
        else:
            out[np.arange(out.shape[0]), idx] = 1.0
        return out

    tau = np.float32(tau)

    # Log-space softmax
    x = np.log(edge_visits + 1e-8, dtype=np.float32)
    x /= tau
    x -= x.max(axis=-1, keepdims=True)

    out = np.exp(x, dtype=np.float32)
    out /= out.sum(axis=-1, keepdims=True)

    # Fast NaN failsafe
    if not np.isfinite(out).all():
        out.fill(0.0)
        idx = np.argmax(edge_visits, axis=-1)
        if out.ndim == 1:
            out[idx] = 1.0
        else:
            out[np.arange(out.shape[0]), idx] = 1.0

    return out


# -----------------------------
# Full Monte Carlo class for search and self play
# -----------------------------

class MonteCarlo:

    def __init__(
        self, env,
        simulations=100,
        num_games=128,
        determ_depth=6,
        rand_pos_depth=10,
        alpha=0.3,
        epsilon=0.25
    ):

        self.env = env

        self.num_games = num_games  # Number of self-play games (will be batch size in NN forward pass)
        self.simulations = simulations
        self.determ_depth = determ_depth  # Number of moves into each game after which temperature=0
        self.rand_pos_depth = rand_pos_depth # Max depth at which to find random starting positions 
        self.c_puct = 1.5   # Search exploration constant
        self.alpha = alpha
        self.epsilon = epsilon
        self.tau_move = 1.0   # Temperature applied to self-play move selection
        self.tau_target = 1.0  # Temperature applied to data saved as training samples
        self.max_num_leaves = self.num_games    # Maximum non-terminal leaves collected on each selection step
        self.max_num_term = 5 * self.max_num_leaves
        self.VIRTUAL_LOSS = 1.0
        self.H_search = 10.0  # Track MCTS search entropy
        self.use_dirichlet = True  # Easily control random search noise for self-play sessions

        # Preallocated arrays used in game operations 
        self.tmp_board = np.zeros(env.BOARD_LEN, dtype=np.int32)
        self.tmp_actions = np.zeros(env.TOT_ACTIONS, dtype=np.int32)
        self.tmp_path = np.empty((env.MAX_DEPTH, 2), dtype=np.int32)

        # MCTS search table
        self.keys = np.empty(TABLE_SIZE, dtype=np.uint64)
        self.wins = np.zeros((TABLE_SIZE, env.TOT_ACTIONS), dtype=np.float32)
        self.visits = np.zeros((TABLE_SIZE, env.TOT_ACTIONS), dtype=np.int32)
        self.priors = np.zeros((TABLE_SIZE, self.env.TOT_ACTIONS), dtype=np.float32)
        self.v_loss = np.zeros((TABLE_SIZE, env.TOT_ACTIONS), dtype=np.float32)

        # Preallocate arrays used during selection step
        self.leaf_boards = np.empty((self.num_games, self.env.BOARD_LEN), dtype=np.int32)
        self.leaf_players = np.empty(self.num_games, dtype=np.int32)
        self.leaf_paths = np.empty((self.max_num_leaves, self.env.MAX_DEPTH, 2), dtype=np.int32)
        self.leaf_hashes = np.empty(self.max_num_leaves, dtype=np.uint64)
        self.term_paths = np.empty((self.max_num_term, self.env.MAX_DEPTH, 2), dtype=np.int32)
        self.term_results = np.empty(self.max_num_term, dtype=np.float32)

        # Define selection function using environment functions + constants
        self.batch_select = batch_select_wrapper(
            self.env.make_hash, self.env.legal_actions, self.env.action_mask, self.env.step,
            np.int32(self.env.MAX_DEPTH), np.float32(self.c_puct), np.float32(self.VIRTUAL_LOSS)
        )


    def search(self, input_boards, input_players, net, simulations=100, use_dirichlet=True):

        # Reset keys table (other array elements will be zeroed as new keys are inserted)
        self.keys[:] = EMPTY_HASH

        # Inputs always of constant size num_games.
        input_len = input_boards.shape[0]
        root_idxs = np.full(input_len, -1, dtype=np.int32)  # Store unique root indices

        for sim in range(simulations):

            if sim == 0:

                # "Leaves" should just be input boards/players on first simulation
                self.leaf_boards[:input_len] = input_boards
                self.leaf_players[:input_len] = input_players
                for r in range(input_len):
                    root_hash = np.uint64(self.env.make_hash(input_boards[r], input_players[r]))
                    self.leaf_hashes[r] = root_hash
                num_leaves = input_len

            else:

                num_leaves, num_term = self.batch_select(
                    input_boards, input_players,
                    self.leaf_boards, self.leaf_players, self.leaf_paths, self.leaf_hashes,
                    self.term_paths, self.term_results,
                    self.max_num_leaves, self.max_num_term,
                    self.keys, self.wins, self.visits, self.priors, self.v_loss,
                    self.tmp_board, self.tmp_actions, self.tmp_path
                )

                # Immediately backpropagate along all terminal paths
                backpropagate(self.term_paths[:num_term], self.term_results[:num_term], self.wins, self.visits)

            if num_leaves == 0:
                continue
            
            # Convert leaf states into NN input version and get associated legal move masks
            states = np.empty((num_leaves, *self.env.FULL_BOARD_SHAPE), dtype=np.float32)
            masks = np.zeros((num_leaves, self.env.TOT_ACTIONS), dtype=np.bool_)
            for leaf in range(num_leaves):
                self.env.full_board(self.leaf_boards[leaf], self.leaf_players[leaf], states[leaf])
                n_legal = self.env.legal_actions(self.leaf_boards[leaf], self.leaf_players[leaf], self.tmp_actions)
                masks[leaf, self.tmp_actions[:n_legal]] = True
            states_tensor = torch.from_numpy(states)
            masks_tensor = torch.from_numpy(masks)
            with torch.no_grad():
                policy_batch_tensor, value_batch_tensor = net(states_tensor, masks_tensor)
            policies = policy_batch_tensor.cpu().numpy()
            values = value_batch_tensor.cpu().numpy().flatten()

            if sim == 0 and use_dirichlet:
                # On first simulation, leaves are just input boards/players.
                # This time only we mix policies (priors of root children) with dirichlet noise.
                gamma = np.random.gamma(self.alpha, 1.0, size=policies.shape) * masks
                row_sum = gamma.sum(axis=1, keepdims=True)
                row_sum[row_sum == 0] = 1.0
                dirichlet = gamma / row_sum
                policies[:] = (1 - self.epsilon) * policies + self.epsilon * dirichlet
                # Just for safety - renormalize over each row
                row_sum = policies.sum(axis=1, keepdims=True)
                row_sum[row_sum == 0] = 1.0
                policies /= row_sum

            for leaf in range(num_leaves):

                leaf_found, leaf_idx = hash_idx(self.keys, self.leaf_hashes[leaf])
                if leaf_found:
                    # Leaf already in table; no need to re-insert
                    continue
                if leaf_idx < 0:
                    # Table full
                    continue
                self.keys[leaf_idx] = self.leaf_hashes[leaf]
                self.wins[leaf_idx, :] = 0.0
                self.visits[leaf_idx, :] = 0
                self.priors[leaf_idx, :] = policies[leaf]

                # On the first simulation, save root indices
                if sim == 0:
                    root_idxs[leaf] = leaf_idx

            # Backpropagate all leaf evaluations after first simulation
            if sim > 0:
                backpropagate(self.leaf_paths[:num_leaves], values, self.wins, self.visits)

    
        # Return the final unnormalized visit count to each root edge
        root_visits = np.zeros((len(root_idxs), self.visits.shape[1]), dtype=np.float32)
        # Ignore root_idxs of -1 (visits to these index edges should be left as all zeros)
        valid = root_idxs >= 0
        root_visits[valid] = self.visits[root_idxs[valid]].astype(np.float32)

        return root_visits


    def batch_self_play(self, net):
        
        data = []   # List of data samples from self-play

        # Track MCTS search entropy for non-deterministic moves
        H_accum = 0.0     # Weighted entropy sum
        W_accum = 0.0     # Weight sum
        entropy_scale = max(1.0, self.determ_depth / 2)

        tau_move = self.tau_move
        game_boards = np.empty((self.num_games, self.env.BOARD_LEN), dtype=np.int32)
        game_players = np.empty(self.num_games, dtype=np.int32)
        # Start each game with random positions
        self.env.random_games(
            game_boards, game_players, self.tmp_board, self.tmp_actions, 
            self.env.START_BOARD, max_moves=self.rand_pos_depth
        )
        game_dones = np.zeros(self.num_games, dtype=np.bool_)
        game_winners = np.zeros(self.num_games, dtype=np.int32)

        for move in range(self.env.MAX_DEPTH + 1):

            # Switch to deterministic eval if deep enough
            if move >= self.determ_depth:
                tau_move = 0.0
            active_idx = np.where(~game_dones)[0]
            if active_idx.size == 0:  # No more unfinished games
                break
            
            # Repeat unfinished game indices if some games are completed
            if len(active_idx) < self.num_games:
                batch_idx = np.resize(active_idx, self.num_games)
            else:
                batch_idx = active_idx

            root_visits = self.search(
                game_boards[batch_idx], 
                game_players[batch_idx], 
                net=net, 
                simulations=self.simulations, 
                use_dirichlet=self.use_dirichlet
            )

            # Update entropy tracking
            if tau_move > 0.0:  # Ignore fully deterministic phase
                weight = np.exp(-move / entropy_scale)
                # Compute entropy for active games only
                for i in range(len(active_idx)):
                    visits = root_visits[i]
                    if visits.sum() > 0:
                        # Entropy formula
                        tmp_v = visits + 1e-8
                        tmp_v /= tmp_v.sum()
                        H = -np.sum(tmp_v * np.log(tmp_v))
                        H_accum += weight * H
                        W_accum += weight

            # Apply two different temperatures (for move selection & training sample)
            move_policies = apply_temp(root_visits, tau=tau_move)
            target_policies = apply_temp(root_visits, tau=self.tau_target)
            
            # Get the root policy for every active game
            for i, b in enumerate(active_idx):

                board = game_boards[b]
                player = game_players[b]
                done = game_dones[b]
                winner = game_winners[b]
                move_policy = move_policies[i]
                target_policy = target_policies[i]

                if np.sum(move_policy) == 0 or np.sum(target_policy) == 0:
                    # No valid policy; hold off and save for next round
                    # This may throw off move count for this particular game
                    continue

                state = np.empty(self.env.FULL_BOARD_SHAPE, dtype=np.float32)
                self.env.full_board(board, player, state)
                tmp_val = (b, player)  # Save final win index and current player
                data.append([state, target_policy.copy(), tmp_val])

                action = np.random.choice(np.arange(self.env.TOT_ACTIONS), p=move_policy)
                # Apply move (new board is written to tmp_board) 
                new_player, new_done, new_winner = self.env.step(
                    board, player, done, winner, action, self.tmp_board
                )

                # Reassign original array elements
                game_boards[b, :] = self.tmp_board[:]
                game_players[b] = new_player
                game_dones[b] = new_done
                game_winners[b] = new_winner

            #  All arrays have now been modified in-place; game states have progressed one move

        # Update final search entropy
        if W_accum > 0:
            self.H_search = H_accum / W_accum
        else:
            self.H_search = 0.0

        # Input final game results
        for d_idx in range(len(data)):
            state, policy, tmp_val = data[d_idx]
            win_idx, current_player = tmp_val
            val = compute_result(game_winners[win_idx], current_player)
            data[d_idx] = (
                torch.from_numpy(state), 
                torch.from_numpy(policy), 
                torch.tensor(val)
            )
        
        return data
    

    def compare_networks(self, net, other_net, tests=100):

        """
        Similar to batch_self_play, but specifically plays with another net to see which preforms better.
        Returns percent of self.net wins, other_net wins, and draws.
        """
        net_wins = 0
        other_net_wins = 0
        draws = 0

        game_boards = np.empty((tests, self.env.BOARD_LEN), dtype=np.int32)
        game_players = np.empty(tests, dtype=np.int32)
        # Start each game with random positions
        self.env.random_games(
            game_boards, game_players, self.tmp_board, self.tmp_actions, 
            self.env.START_BOARD, max_moves=self.rand_pos_depth
        )
        game_dones = np.zeros(tests, dtype=np.bool_)
        game_winners = np.zeros(tests, dtype=np.int32)

        # In each random position, self.net calculates first moves.
        # Save current game_players so that we can reference which players are self.net
        original_players = game_players.copy()

        for move in range(self.env.MAX_DEPTH + 1):

            current_net = net if (move % 2 == 0) else other_net

            active_idx = np.where(~game_dones)[0]
            if active_idx.size == 0:  # No more unfinished games
                break
            
            # Repeat unfinished game indices if some games are completed
            if len(active_idx) < self.num_games:
                batch_idx = np.resize(active_idx, self.num_games)
            else:
                batch_idx = active_idx

            root_visits = self.search(
                game_boards[batch_idx], 
                game_players[batch_idx], 
                net=current_net, 
                simulations=self.simulations,
                use_dirichlet=False
            )

            policies = apply_temp(root_visits, tau=0.0)  # Always deterministic
            
            # Get the root policy for every active game
            for i, b in enumerate(active_idx):

                board = game_boards[b]
                player = game_players[b]
                done = game_dones[b]
                winner = game_winners[b]
                policy = policies[i]

                if np.sum(policy) == 0:
                    # Hold off and save for next round
                    continue

                action = np.argmax(policy)
                # Apply move (new board is written to tmp_board) 
                new_player, new_done, new_winner = self.env.step(
                    board, player, done, winner, action, self.tmp_board
                )

                if new_done:
                    if new_winner == 0:
                        draws += 1
                    elif new_winner == original_players[b]:
                        # The original player in rand start position won; original net won.
                        net_wins += 1
                    else:
                        other_net_wins += 1

                # Reassign original array elements
                game_boards[b, :] = self.tmp_board[:]
                game_players[b] = new_player
                game_dones[b] = new_done
                game_winners[b] = new_winner

            #  All arrays have now been modified in-place; game states have progressed one move

        tot = net_wins + other_net_wins + draws
        return net_wins / tot, other_net_wins / tot, draws / tot
    

    def single_search(self, board, player, net, simulations=1000, tau=0.0):

        """
        Convenient function for doing MCTS on a single board/player state.
        """

        self.max_num_leaves = 1
        boards = np.expand_dims(board, axis=0)
        players = np.expand_dims(player, axis=0)
        root_visits = self.search(boards, players, net=net, simulations=simulations, use_dirichlet=False)
        policy = apply_temp(root_visits, tau=tau)[0]
        return policy
    

    def net_eval(self, net, board, player):

        """
        Use the final trained network to get move policy for a state.
        """

        tmp_state = np.zeros(self.env.FULL_BOARD_SHAPE, dtype=np.float32)
        tmp_mask = np.zeros(self.env.TOT_ACTIONS, dtype=np.bool_)
        self.env.full_board(board, player, tmp_state)
        n_legal = self.env.legal_actions(board, player, self.tmp_actions)
        tmp_mask[self.tmp_actions[:n_legal]] = True  # Mark legal action indices
        state = torch.from_numpy(np.expand_dims(tmp_state, axis=0))
        mask = torch.from_numpy(np.expand_dims(tmp_mask, axis=0))
        with torch.no_grad():
            policy_batch_tensor, _ = net(state, mask)
        policy = policy_batch_tensor.cpu().numpy()[0]
        return policy