def print_board(board):
    for row in board:
        print(' | '.join(row))
        print('-' * 5)

def check_winner(board, player):
    # Check rows, columns, and diagonals for a winning line
    for i in range(3):
        if all([board[i][j] == player for j in range(3)]) or all([board[j][i] == player for j in range(3)]):
            return True
    if all([board[i][i] == player for i in range(3)]) or all([board[i][2 - i] == player for i in range(3)]):
        return True
    
    # Check for a bridge (corners connected)
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    if all([board[r][c] == player for r, c in corners]):
        return True
    
    return False

def check_full(board):
    return all([cell != ' ' for row in board for cell in row])

def play_game():
    # Initialize empty board
    board = [[' ' for _ in range(3)] for _ in range(3)]
    players = ['X', 'O']
    current_player = 0

    print("Welcome to Tic-Tac-Havannah!")
    print_board(board)

    while True:
        print(f"Player {players[current_player]}'s turn")
        row = int(input("Enter row (0-2): "))
        col = int(input("Enter column (0-2): "))
        
        if board[row][col] == ' ':
            board[row][col] = players[current_player]
            print_board(board)
            
            # Check for a win
            if check_winner(board, players[current_player]):
                print(f"Player {players[current_player]} wins!")
                break
            # Check for a draw
            elif check_full(board):
                print("It's a draw!")
                break

            # Switch player
            current_player = 1 - current_player
        else:
            print("Cell already occupied! Try again.")

if __name__ == "__main__":
    play_game()
