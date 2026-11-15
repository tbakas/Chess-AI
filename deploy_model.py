import streamlit
from random import randint
from create_model import ChessModel
from train_model import ChessLightning
from transform import move_to_int, board_to_list
import chess
import chess.svg
import torch
from time import sleep


def get_int_to_move(m_to_i):
    out = dict()
    for x in m_to_i:
        i = m_to_i[x]
        out[i] = x
    return out


int_to_move = get_int_to_move(move_to_int)
checkpoint_path = "best_model.ckpt"


# Cache the model
@streamlit.cache_resource
def load_model():
    return ChessLightning.load_from_checkpoint(checkpoint_path, model=ChessModel()).model


model = load_model()

# Initialize board state
if "board" not in streamlit.session_state:
    streamlit.session_state.board = chess.Board()
if 'turn' not in streamlit.session_state:
    streamlit.session_state.turn = 0
if "game_started" not in streamlit.session_state:
    streamlit.session_state.game_started = False
if "player_choice" not in streamlit.session_state:
    streamlit.session_state.player_choice = 0
if 'move_input' not in streamlit.session_state:
    streamlit.session_state.move_input = None


def select_color():
    choice = streamlit.selectbox("Choose color:", ["White", "Black", "Random"])
    if streamlit.button("Play"):
        if choice == 'White':
            streamlit.session_state.player_choice = 0
        elif choice == 'Black':
            streamlit.session_state.player_choice = 1
        else:
            streamlit.session_state.player_choice = randint(0, 1)
        streamlit.session_state.game_started = True
        streamlit.rerun()


def check_game_over():
    out = False
    if streamlit.session_state.board.is_checkmate():
        streamlit.warning('Checkmate! Game Over!')
        out = True
    elif streamlit.session_state.board.is_checkmate():
        streamlit.warning('Stalemate! Draw.')
    return out


def player_move():
    with streamlit.form("move_form", clear_on_submit=True):
        move_input = streamlit.text_input("Enter your move in UCI format (e.g., e2e4):")
        submitted = streamlit.form_submit_button("Submit Move")
    if submitted and move_input:
        try:
            move = chess.Move.from_uci(move_input)
            if move in streamlit.session_state.board.legal_moves:
                streamlit.session_state.board.push(move)
                streamlit.session_state.turn = 1 ^ streamlit.session_state.turn
                streamlit.rerun()
            else:
                streamlit.error("Illegal move! Try again.")
        except ValueError:
            streamlit.error("Invalid move format. Use UCI like 'e2e4'.")


def ai_move():
    with torch.no_grad():
        position = torch.tensor(board_to_list(streamlit.session_state.board))
        t = torch.tensor(streamlit.session_state.turn, dtype=torch.long)

        mask = torch.zeros(size=(1, 1968), dtype=torch.bool)
        for m in streamlit.session_state.board.legal_moves:
            m = m.uci()
            mask[0][move_to_int[m]] = 1

        masked_logits = model(position.view((1, 64)), t).masked_fill(mask == 0, float('-inf'))

        n = torch.argmax(masked_logits)
        move = chess.Move.from_uci(int_to_move[n.item()])
    streamlit.session_state.board.push(move)
    streamlit.session_state.turn = 1 ^ streamlit.session_state.turn
    sleep(0.5)
    streamlit.rerun()


def reset_board():
    streamlit.session_state.board = chess.Board()
    streamlit.session_state.turn = 0
    streamlit.rerun()


def return_to_selection():
    streamlit.session_state.board = chess.Board()
    streamlit.session_state.turn = 0
    streamlit.session_state.game_started = False
    streamlit.session_state.player_choice = 0
    streamlit.rerun()


if not streamlit.session_state.game_started:
    streamlit.title("Game Setup")
    # Player selects color
    select_color()
else:
    # Check if game is still going
    game_over = check_game_over()
    if not game_over and streamlit.session_state.board.is_check():
        streamlit.warning("Check!")

    if not game_over:
        # Player's turn
        if streamlit.session_state.player_choice == streamlit.session_state.turn:
            player_move()
        # Model's turn
        else:
            ai_move()

    # Display board from player's perspective
    if streamlit.session_state.player_choice:
        board_svg = chess.svg.board(board=streamlit.session_state.board, size=400, orientation=chess.BLACK)
    else:
        board_svg = chess.svg.board(board=streamlit.session_state.board, size=400)
    streamlit.image(board_svg, use_container_width=False)

    # Reset game or return to color selection.
    if streamlit.button("Reset Game"):
        reset_board()
    if streamlit.button('Return to Color Selection'):
        return_to_selection()
