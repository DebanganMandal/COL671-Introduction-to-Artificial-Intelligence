import time
import math
import random
import numpy as np
from helper import *

class AIPlayer:
    def __init__(self, player_number: int, timer):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer
        self.opponent_number = 1 if player_number == 2 else 2
        self.table = {}
        self.max_depth = 2

    def shuffle_with_probability(self, moves, probability=0.6):
        if random.random() < probability:
            random.shuffle(moves)
        return moves

    def minimax(self, state: np.array, depth: int, alpha: float, beta: float, is_maximizing: bool, start_time: float) -> float:
        flag = 1
        remaining_time = fetch_remaining_time(self.timer, self.player_number)
        if time.time() - start_time > remaining_time:
            raise TimeoutError

        state_tuple = tuple(map(tuple, state))
        if state_tuple in self.table:
            return self.table[state_tuple]

        valid_moves = get_valid_actions(state)
        if depth == 0 or not valid_moves or check_win(state, (0, 0), self.player_number)[0]:
            eval = self.evaluate_state(state)
            self.table[state_tuple] = eval
            return eval

        if is_maximizing:
            max_eval = -math.inf
            for move in valid_moves:
                new_state = state.copy()
                new_state[move[0], move[1]] = self.player_number
                eval = self.minimax(new_state, depth - 1, alpha, beta, False, start_time)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            self.table[state_tuple] = max_eval
            return max_eval
        else:
            min_eval = math.inf
            for move in valid_moves:
                new_state = state.copy()
                new_state[move[0], move[1]] = self.opponent_number
                eval = self.minimax(new_state, depth - 1, alpha, beta, True, start_time)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            self.table[state_tuple] = min_eval
            return min_eval

    def evaluate_state(self, state: np.array) -> float:
        player_score = np.sum(state == self.player_number)
        opponent_score = np.sum(state == self.opponent_number)
        player_win_potential = self.potential(state, self.player_number)
        opponent_win_potential = self.potential(state, self.opponent_number)
        return player_score - opponent_score + player_win_potential - opponent_win_potential

    def potential(self, state: np.array, player: int) -> float:
        potential_score = 0.0
        valid_moves = get_valid_actions(state)
        for move in valid_moves:
            new_state = state.copy()
            new_state[move[0], move[1]] = player
            if check_win(new_state, move, player)[0]:
                potential_score += 10.0
        return potential_score

    def get_move(self, state: np.array) -> Tuple[int, int]:
        best_move = None
        best_value = -math.inf

        valid_moves = get_valid_actions(state)
        valid_moves = self.shuffle_with_probability(valid_moves, 0.7)

        start_time = time.time()
        try:
            for depth in range(1, self.max_depth + 1):
                best_value = -math.inf
                for move in valid_moves:
                    new_state = state.copy()
                    new_state[move[0], move[1]] = self.player_number
                    move_value = self.minimax(new_state, depth, -math.inf, math.inf, False, start_time)
                    if move_value > best_value:
                        best_value = move_value
                        best_move = move
        except TimeoutError:
            pass

        if best_move is None:
            best_move = valid_moves[0] 
    
        return int(best_move[0]), int(best_move[1])