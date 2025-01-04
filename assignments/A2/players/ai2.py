import numpy as np
from helper import get_valid_actions, check_win

class AIPlayer:
    def _init_(self, player_number: int, timer):
        self.player_number = player_number
        self.type = 'ai3'
        self.player_string = f'Player {player_number}: ai'
        self.timer = timer
        self.opponent_num = 1 if player_number == 2 else 2

    def get_move(self, state: np.array) -> tuple:
        valid_moves = get_valid_actions(state)
        best_move = None
        best_score = float('-inf')
        valid_moves.sort(key=lambda move: self.center_proximity(move, state), reverse=True)

        for move in valid_moves:
            simulated_state = state.copy()
            simulated_state[move] = self.player_number

            if self.move_defeats_immediate_threat(simulated_state, move):
                return move  

            score = self.evaluate_move(simulated_state, move)

            simulated_state[move] = self.opponent_num
            if check_win(simulated_state, move, self.opponent_num)[0]:
                return move  

            simulated_state[move] = self.player_number

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def evaluate_move(self, state, move):
        score = 0
        if check_win(state, move, self.player_number)[0]:
            return float('inf')
        
        score += self.evaluate_position(state, move)
        score += self.secondary_heuristics(state, move)
        return score

    def evaluate_position(self, state, move):
        score = 0
        win, win_type = check_win(state, move, self.player_number)
        if win:
            if win_type == 'ring':
                score += 100
            elif win_type == 'bridge':
                score += 150
            elif win_type == 'fork':
                score += 130

        state[move] = self.opponent_num
        opponent_win, _ = check_win(state, move, self.opponent_num)
        state[move] = 0

        if opponent_win:
            score -= 200  

        state[move] = self.player_number
        score += self.connected_pieces_heuristic(state, move)
        return score

    def connected_pieces_heuristic(self, state, move):
        score = 0
        for direction in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            score += self.line_value(state, move, direction)
        return score

    def line_value(self, state, move, direction):
        player_score = 0
        opponent_score = 0
        for offset in range(-2, 3):
            pos = (move[0] + offset * direction[0], move[1] + offset * direction[1])
            if 0 <= pos[0] < state.shape[0] and 0 <= pos[1] < state.shape[1]:
                if state[pos] == self.player_number:
                    player_score += 1
                elif state[pos] == self.opponent_num:
                    opponent_score += 1

        if opponent_score > 1 and player_score == 0:
            return -(opponent_score ** 3)  
        if player_score > 1 and opponent_score == 0:
            return player_score ** 2  
        return 0

    def secondary_heuristics(self, state, move):
        score = 0
        score += self.center_proximity(move, state)
        score += self.defensive_heuristics(state, move)  
        return score

    def defensive_heuristics(self, state, move):
        score = 0
        state[move] = self.opponent_num
        valid_moves = get_valid_actions(state)

        for opponent_move in valid_moves:
            state[opponent_move] = self.opponent_num
            if check_win(state, opponent_move, self.opponent_num)[0]:
                score -= 100  
            state[opponent_move] = 0

        state[move] = self.player_number
        return score

    def center_proximity(self, move, state):
        center = (state.shape[0] // 2, state.shape[1] // 2)
        return -abs(move[0] - center[0]) - abs(move[1] - center[1])

    def move_defeats_immediate_threat(self, state, move):
        state[move] = self.opponent_num
        if check_win(state, move, self.opponent_num)[0]:
            state[move] = 0
            return True
        state[move] = self.player_number
        return False