# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 20:45:38 2024

@author: bruno
"""

import ollama
import chess
from IPython.display import display
import re
import numpy as np
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
import chess.svg
import base64
import cairosvg

class OllamaAgent:
    def __init__(self, text, context):
        self.text = text         
        self.context = context    

    def get_text(self):
        return self.text

    def get_context(self):
        return self.context

    def set_text(self, new_text):
        self.text = new_text

    def set_context(self, new_context):
        self.context = new_context

    def add_context(self, new_context):
        self.context += new_context

    def get_response(self, image_path=None):
        if image_path:
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                # Replace this with the actual call to the `ollama.chat` function.
                result = ollama.chat(model='llama3.2-vision', messages=[
                    {'role': 'user', 'content': self.get_text(), 'images': [image_base64]}
                ])
        else:
            #llama3.2 or deepseek-r1 here
            result = ollama.chat(model='llama3.2', messages=[            
                {'role': 'user', 'content': self.get_text()}
            ])
        OllamaResponse = result['message']['content']
        return OllamaResponse

san_regex = re.compile(r"""
    ^(
        (?:O-O-O|O-O)                               
        |
        (?:[KQRNB])?                                 
        (?:[a-h]?[1-8]?)                            
        x?[a-h][1-8]                                 
        (?:=[QRNB])?                                 
    )$
""", re.VERBOSE)

chess_db = [
    {
        "fen": "r1bq1rk1/pp1nppbp/2pp1np1/8/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 8",
        "suggested_move": "e4",
    },
    {
        "fen": "r2q1rk1/ppp2ppp/3bp3/3n4/3P4/2N2N2/PPP2PPP/R1BQ1RK1 b - - 0 10",
        "suggested_move": "e4",
    },
    {
        "fen": "2rq1rk1/pp1n1ppp/2p1pn2/8/2PP4/2N2N2/PPQ1BPPP/R1B2RK1 w - - 1 12",
        "suggested_move": "d5",
    },
    {
        "fen": "3q1rk1/pp2bppp/2pr1n2/4p3/2PP4/2N2N2/PPQ1BPPP/R1B2RK1 b - - 0 14",
        "suggested_move": "d5",
    },
    {
        "fen": "r2q1rk1/pp2ppbp/2np1np1/2p5/2P5/2N1PN2/PP1B1PPP/R2QKB1R w KQ - 0 9",
        "suggested_move": "Nd5",
    },
    {
        "fen": "r2q1rk1/pp2bppp/3p1n2/3Pp3/2P5/2N2N2/PP3PPP/R1BQ1RK1 w - - 0 11",
        "suggested_move": "Ng4",
    },
    {
        "fen": "r1bq1rk1/pppp1ppp/2n2n2/3P4/2P5/2N1PN2/PP3PPP/R1BQKB1R b KQ - 0 8",
        "suggested_move": "d6",
    },
    {
        "fen": "2r1r1k1/pp2ppbp/3p1np1/2q5/3P4/2N2N2/PP1Q1PPP/R4RK1 w - - 0 15",
        "suggested_move": "Qb6",
    },
    {
        "fen": "r3kb1r/ppq2ppp/2ppbn2/3P4/3P1B2/2N2N2/PP2QPPP/R3KB1R w KQkq - 0 14",
        "suggested_move": "O-O-O",
    },
    {
        "fen": "r1bq1rk1/ppp2ppp/3p1n2/4p3/2PP4/2N2N2/PP2BPPP/R1BQ1RK1 w - - 0 10",
        "suggested_move": "Bg4",
    },
    {
        "fen": "3r1rk1/ppq2ppp/3b1n2/2p5/3P4/2N1BN2/PPQ2PPP/3R1RK1 w - - 0 16",
        "suggested_move": "Rfe8",
    },
    {
        "fen": "2rq1rk1/pp2bppp/3p1n2/3P4/2P5/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 13",
        "suggested_move": "Qc7",
    },
    {
        "fen": "r2q1rk1/1b1nbppp/p1pp1n2/8/1PP1P3/2N1BN2/P1BQ1PPP/R4RK1 w - - 0 12",
        "suggested_move": "dxc5",
    },
    {
        "fen": "r1bq1rk1/ppp2ppp/3p4/8/2Pn4/2N2N2/PP2BPPP/R1BQR1K1 b - - 0 13",
        "suggested_move": "f5",
    },
    {
        "fen": "r3r1k1/ppq2ppp/2pp1n2/3P4/2P5/2N2N2/PP1QBPPP/R4RK1 w - - 0 17",
        "suggested_move": "Rab8",
    },
    {
        "fen": "r3r1k1/pp2qppp/2pp1n2/4p3/2P2B2/2N2N2/PP2QPPP/R4RK1 w - - 0 15",
        "suggested_move": "h6",
    },
    {
        "fen": "2r2rk1/ppq2ppp/2ppbn2/3P4/2P5/2N2N2/PPQ2PPP/1RB1R1K1 w - - 0 17",
        "suggested_move": "cxd5",
    },
]

def get_last_word(text):
    text = text.rstrip('.')
    last_word = text.split()[-1]
    return last_word

def rag_openings(actual_context):
    debiuty = "1.e4, e5 2.Nf3, Nc6 3.Bb5, a6 4.Bxc6, dxc6; 1.e4, e5 2.Nf3, Nc6 3.Bb5, a6 4.Bxc6, bxc6; 1.e4, e5 2.Nf3, Nc6 3.Bb5, a6 4.Ba4, d6; 1.e4, e5 2.Nf3, Nc6 3.Bb5, a6 4.Ba4, f5; 1.e4, e5 2.Nf3, Nc6 3.Bb5, a6 4.Ba4, g5; 1.e4, e5 2.Nf3, Nc6 3.Bb5, a6 4.Ba4, b5; 1.e4, e5 2.Nf3, Nc6 3.Bb5, a6 4.Ba4, Nf6; 1.e4, e5 2.Nf3, Nc6 3.Bb5, a6 4.Ba4, Be7; 1.e4, c5 2.Nf3, e6 3.d4, cxd4 4.Nxd4, a6; 1.e4, c5 2.Nf3, e6 3.d4, cxd4 4.Nxd4, Nc6; 1.e4, c5 2.Nf3, e6 3.d4, cxd4 4.Nxd4, Nf6; 1.e4, c5 2.Nf3, d6 3.d4, cxd4 4.Nxd4, Nf6; 1.e4, c5 2.Nf3, d6 3.d4, cxd4 4.Nxd4, g6; 1.e4, c5 2.Nf3, d6 3.d4, cxd4 4.Nxd4, Nc6; 1.e4, c5 2.Nf3, Nc6 3.d4, cxd4 4.Nxd4, g6; 1.e4, c5 2.Nf3, Nc6 3.d4, cxd4 4.Nxd4, e6; 1.e4, c5 2.Nf3, Nc6 3.d4, cxd4 4.Nxd4, e5; 1.e4, c5 2.Nf3, Nc6 3.d4, cxd4 4.Nxd4, Nf6; 1.e4, c5 2.Nf3, Nc6 3.d4, cxd4 4.c3, dxc3; 1.e4, c5 2.Nf3, Nc6 3.d4, cxd4 4.Bd3, g6; 1.d4, Nf6 2.c4, c5 3.d5, e6 4.Nc3, exd5; 1.d4, Nf6 2.c4, c5 3.d5, e6 4.Nc3, d6; 1.d4, Nf6 2.c4, c5 3.d5, g6 4.Nc3, Bg7; 1.d4, Nf6 2.c4, c5 3.d5, d6 4.Nc3, g6; 1.d4, Nf6 2.c4, e6 3.Nf3, c5 4.d5, exd5; 1.d4, Nf6 2.c4, e6 3.Nf3, d5 4.g3, dxc4; 1.d4, Nf6 2.c4, e6 3.g3, d5 4.Bg2, dxc4; 1.d4, Nf6 2.c4, e6 3.Nc3, Bb4 4.Nf3, b6; 1.d4, Nf6 2.c4, c5 3.d5, e5 4.Nc3, d6; 1.d4, f5 2.c4, Nf6 3.g3, e6 4.Bg2, Be7; 1.d4, f5 2.c4, Nf6 3.g3, e6 4.Bg2, d5; 1.d4, f5 2.c4, Nf6 3.g3, e6 4.Bg2, c6; 1.d4, f5 2.c4, Nf6 3.g3, e6 4.Bg2, Bb4; 1.d4, f5 2.c4, Nf6 3.g3, e6 4.Bg2, d5; 1.d4, f5 2.c4, Nf6 3.Nf3, e6 4.g3, Bb4; 1.d4, f5 2.c4, Nf6 3.Nf3, d6 4.g3, g6; 1.d4, f5 2.c4, Nf6 3.Nf3, g6 4.g3, d6; 1.e4, d5 2.exd5, Hxd5 3.Nc3, Ha5 4.d4, Nf6; 1.e4, d5 2.exd5, Hxd5 3.Nc3, Ha5 4.d4, e5; 1.e4, d5 2.exd5, Hxd5 3.Nc3, Ha5 4.d4, Bf5; 1.e4, d5 2.exd5, Hxd5 3.Nc3, Ha5 4.d4, c6; 1.e4, d5 2.exd5, Hxd5 3.Nc3, Ha5 4.Nf3, c6; 1.e4, d5 2.exd5, Hxd5 3.Nc3, Ha5 4.a3, Nf6; 1.e4, Nf6 2.e5, Nd5 3.d4, d6 4.Nf3, Bg4; 1.e4, Nf6 2.e5, Nd5 3.d4, d6 4.Nf3, dxe5; 1.e4, Nf6 2.e5, Nd5 3.d4, d6; 4.Nf3, g6; 1.e4, Nf6 2.e5, Nd5 3.d4, d6 4.Nf3, Bf5; 1.e4, d6 2.d4, Nf6 3.Nc3, g6 4.Be3, Bg7; 1.e4, d6 2.d4, Nf6 3.Nc3, g6 4.Be3, c6; 1.e4, d6 2.d4, Nf6 3.Nc3, g6 4.Be3, a6; 1.e4, d6 2.d4, Nf6 3.Nc3, g6 4.Be2, Bg7; 1.e4, d6 2.d4, Nf6 3.Nc3, g6 4.Be2, a6; 1.e4, d6 2.d4, Nf6 3.Nc3, g6 4.Be2, Nc6; 1.e4, d6 2.d4, Nf6 3.Nc3, g6 4.Nf3, Bg7; 1.e4, d6 2.d4, Nf6; 3.Nc3, g6 4.Nf3, c6; 1.e4, d6 2.d4, Nf6 3.Nc3, g6 4.Nf3, a6; 1.e4, c6 2.d4, d5 3.e5, Bf5 4.Nd2, e6; 1.e4, c6 2.d4, d5 3.e5, a6 4.Bd3, c5; 1.e4, c6 2.d4, d5 3.e5, c5 4.Be3, Nd7; 1.e4, c6 2.d4, d5 3.e5, c5 4.Nd3, Nd7; 1.e4, c6 2.d4, d5 3.e5, c5 4.c4, Nd7; 1.e4, c6 2.d4, d5 3.e5, c5 4.Nf3, cxd4; 1.e4, c6 2.d4, d5 3.e5, c5 4.dxc5, e6; 1.e4, c6 2.d4, d5 3.e5, c5 4.c3, Nc6; 1.e4, c6 2.d4, d5 3.exd5, cxd5 4.c4, e6; 1.e4, c6 2.d4, d5 3.exd5, cxd5 4.Nf3, Nc6; 1.e4, c6 2.d4, d5 3.exd5, cxd5 4.Bd3, Nc6; 1.e4, c6 2.d4, d5 3.exd5, cxd5 4.Bf4, Nc6; 1.e4, c6 2.d4, d5 3.Nc3, dxe4 4.Nxe4, Nf6; 1.e4, c6 2.d4, d5 3.Nc3, dxe4 4.Nxe4, Nd7; 1.e4, c6 2.d4, d5 3.Nc3, dxe4 4.Nxe4, Bf5; 1.e4, c6 2.d4, d5 3.Nc3, dxe4 4.Nxe4, g6; 1.e4, c6 2.d4, d5 3.Nd2, dxe4 4.Nxe4, Nf6; 1.e4, c6 2.d4, d5 3.Nd2, dxe4 4.Nxe4, Nd7; 1.e4, c6 2.d4, d5 3.Nd2, dxe4 4.Nxe4, Bf5; 1.e4, c6 2.d4, d5 3.Nd2, dxe4 4.Nxe4, g6; 1.e4, e6 2.d4, d5 3.Nc3, Nf6 4.e5, Ne4; 1.e4, e6 2.d4, d5 3.Nc3, Nf6 4.e5, Ng8; 1.e4, e6 2.d4, d5 3.Nc3, Nf6 4.Bg5, Nc6; 1.e4, e6 2.d4, d5 3.Nc3, Nf6 4.Bg5, dxe4; 1.e4, e6 2.d4, d5 3.Nc3, Nf6 4.Bg5, Bb4; 1.e4, e6 2.d4, d5 3.Nc3, Nf6 4.Bg5, h6; 1.e4, e6 2.d4, d5 3.Nc3, Nf6 4.exd5, exd5; 1.e4, e5 2.d4, exd4 3.Hxd4, Nc6 4.He3, Nf6; 1.e4, e5 2.d4, exd4 3.Hxd4, Nc6 4.Hc4, Nf6; 1.e4, e5 2.d4, exd4 3.Hxd4, Nc6 4.Hd3, d5; 1.e4, e5 2.d4, exd4 3.Hxd4, Nf6 4.Nc3, Nc6; 1.e4, e5 2.d4, exd4 3.Hxd4, Nf6 4.Bg5, Nc6; 1.e4, e5 2.d4, exd4 3.Hxd4, Nf6 4.Bf4, Nc6; 1.e4, e5 2.f4, exf4 3.Nf3, g5 4.Nc3, d6; 1.e4, e5 2.f4, exf4 3.Nf3, g5 4.d4, g4; 1.e4, e5 2.f4, exf4 3.Nf3, g5 4.h4, g4; 1.e4, e5 2.f4, exf4 3.Nf3, d6 4.Nc3, g5; 1.e4, e5 2.f4, exf4 3.Nf3, d6 4.d4, g5; 1.e4, e5 2.f4, exf4 3.Nf3, d6 4.Bc4, h6; 1.e4, e5 2.Nf3, Nc6 3.Bc4, Bc5 4.c3, Nf6; 1.e4, e5 2.Nf3, Nc6 3.Bc4, Bc5 4.d3, Nf6; 1.e4, e5 2.Nf3, Nc6 3.Bc4, Bc5 4.Nc3, Nf6; 1.e4, e5 2.Nf3, Nc6 3.Bc4, Nf6 4.d3, Bc5; 1.e4, e5 2.Nf3, Nc6 3.Bc4, Nf6 4.d4, exd4;"
    aktualne_debiuty=""
    debiuty_iterator=0
    
    for x in range(100):
        debiut = ""
        debiut_iterator=0
        for x in debiuty[debiuty_iterator:]:
            if x=='1' and debiut_iterator>1:
                break;
            debiut += x
            debiut_iterator+=1
            debiuty_iterator+=1
        
        debiut_iterator=0
        flag = 1
        for x in actual_context:
            if debiut_iterator < len(debiut):
                if x != debiut[debiut_iterator]:
                    flag = 0
                debiut_iterator+=1;
        if flag == 1:
            aktualne_debiuty+=debiut
            
    return aktualne_debiuty

def compare_positions(current_fen, database):
    """
    Finds the best matching position in the database for the given FEN.
    """
    current_board = chess.Board(current_fen)
    best_match = None
    best_score = 0

    for entry in database:
        db_board = chess.Board(entry["fen"])
        score = sum(
            1 for sq in chess.SQUARES if current_board.piece_at(sq) == db_board.piece_at(sq)
        )

        if score > best_score:
            best_match = entry
            best_score = score

    return best_match

def fen_to_sentences(fen):
    # Split the FEN into its components
    board_fen, turn, castling, en_passant, halfmove, fullmove = fen.split()
    
    # Define the rows and columns
    rows = board_fen.split("/")
    columns = "abcdefgh"
    pieces = {
        'r': 'rook', 'n': 'knight', 'b': 'bishop', 
        'q': 'queen', 'k': 'king', 'p': 'pawn',
        'R': 'rook', 'N': 'knight', 'B': 'bishop', 
        'Q': 'queen', 'K': 'king', 'P': 'pawn'
    }

    sentences = []

    # Iterate through the rows to decode the board
    for rank_index, row in enumerate(rows):
        rank = 8 - rank_index  # Rank starts from 8 at the top
        file_index = 0  # File starts at 'a'
        
        for char in row:
            if char.isdigit():
                # Empty squares, increment file index by the number
                file_index += int(char)
            else:
                # There is a piece
                piece = pieces[char.lower()]
                color = "white" if char.isupper() else "black"
                file = columns[file_index]
                square = f"{file}{rank}"
                sentences.append(f"{color.capitalize()} {piece} is on {square}")
                file_index += 1  # Move to the next file
    
    return ". ".join(sentences) + "."

def average_sentence_embedding(sentence, model):
    embeddings = []
    for word in sentence:
        if word in model.wv:
            embeddings.append(model.wv[word])
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)

def cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

# Find the closest embedding
def find_closest_embedding(query, embeddings):
    similarities = [cosine_similarity(query, emb) for emb in embeddings]
    closest_index = np.argmax(similarities)
    return closest_index

def describe_position(fen_in):
    # Create a chessboard from the FEN string
    board = chess.Board(fen_in)
    
    # Generate the SVG data for the chessboard
    svg_data = chess.svg.board(board=board)

    # Save SVG to a file
    svg_path = "chess_position.svg"
    with open(svg_path, "w") as svg_file:
        svg_file.write(svg_data)

    png_file = "output.png"  # Output PNG file

    # Convert the SVG to PNG using cairosvg
    cairosvg.svg2png(url=svg_path, write_to=png_file)

    agent4 = OllamaAgent("Describe the following chess position", "")
    
    display(board)
    
    # Pass the correct PNG file path to OllamaAgent
    response = agent4.get_response(png_file)
    # Use OllamaAgent to describe the image
    agent4 = OllamaAgent("Describe the following chess position", "")
    display(board)
    response = agent4.get_response("test1.png")
    print(f"Ollama vision response: {response}")
    return response

def handle_conversation():
    board = chess.Board()
    board.set_fen("r3kb1r/ppq2ppp/2ppbn2/3P4/3P1B2/2N2N2/PP2QPPP/R3KB1R w KQkq - 0 14")
    display(board)
    agent1 = OllamaAgent("Im the player", "")
    agent2 = OllamaAgent("Im the verifier", "")
    agent3 = OllamaAgent("Im the expert", "")
    print("Welcome to the AI ChessBot! Format of data input is for example: e4. Type 'exit' to quit.")
    turn = 1
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        agent1.add_context(f"{turn}.{user_input}")
        board.push_san(user_input)
        display(board)
        if board.is_game_over() == 1:
            break
        legal_moves_lst = [
        board.san(move)
        for move in board.legal_moves
        ]
        legal_openings = rag_openings(agent1.get_context())
        
        current_position_fen = board.fen()
        best_match = compare_positions(current_position_fen, chess_db)
        sentences=[]
        for entry in chess_db:
            db_board = entry["fen"]
            sentence = fen_to_sentences(db_board)
            tokenized_sentence = sentence.split()
            sentences.append(tokenized_sentence)
        model = Word2Vec(
            sentences=sentences,
            vector_size=10,   
            window=3,         
            min_count=1,      
            workers=4,      
            sg=1
        )
        possible_matches=[]
        for x in sentences:
            possible_matches.append(average_sentence_embedding(x, model))
        current_position_embedings = average_sentence_embedding(fen_to_sentences(current_position_fen).split(), model)
        closest_index = find_closest_embedding(current_position_embedings, possible_matches)
        #llama_vision_description = describe_position(current_position_fen)
        agent3.set_text(f"There you have some chess tips: Occupy and influence central squares (d4, d5, e4, e5) to give your pieces more space.\
                        Place pieces on their optimal squares (e.g., knights in the center, rooks on open files).\
                        Prevent isolated, doubled, or backward pawns, and protect weak squares.\
                        Exchange pieces, not pawns, when you have a material or positional advantage.\
                        Maintain a solid, connected pawn structure to protect key squares.\
                        Place rooks on open or semi-open files, especially the 7th rank.\
                        Place knights on squares where they can't be attacked and control key areas.\
                        Don't push pawns too far without support to avoid weaknesses.\
                        Keep your king safe but consider activating it in the endgame.\
                        Ensure pieces defend and support each other to create strong threats.\
                        Choose the best move from the following list: {legal_moves_lst}\
                        The current position is: {agent1.get_context()}\
                        Give short responces")
        Ollama3Response = agent3.get_response()
        agent3.set_context(agent1.get_context())
        print(f"Ollama3response: {Ollama3Response}")
        
        agent1.set_text(f"You play chess as Black. You receive list of moves played and give your response on each turn.\
            Here you have most popular chess oppenings that matches current position:{legal_openings}\
            The current position is: {agent1.get_context()}\
            Give only short responses in this format: e5, Nf6, Bc5, Re8, Qe7 etc. Dont add any characters.\
            You can only choose your responce from the following list!!!: {legal_moves_lst}\
            There you have response by expert: {Ollama3Response}\
            There you have most matching move from the database: {chess_db[closest_index]['suggested_move']}\
            And there you have llama vision model description")
            
        Ollama1Response=agent1.get_response()
        print(f"Ollama1response: {Ollama1Response}")
        
        bot_move=""
        if not san_regex.match(Ollama1Response):
            agent2.set_text(f"Pull a chess move that black played out of this text:{Ollama1Response}\
                            respond with only this short move (about 3 characters)!")
            Ollama2Response=agent2.get_response()
            print(f"Ollama2response: {Ollama2Response}")
            agent2.add_context(f"{Ollama1Response} -> {Ollama2Response},")
            incorrectCounter=0
            while not san_regex.match(Ollama2Response):
                if incorrectCounter > 4:
                    break
                agent2.set_text(f"These werent correct responses: {agent2.get_context()}\
                                Pull a chess move that black played out of this text:{Ollama1Response}\
                                Remember you have to respond with only this short move (about 3 characters)!")
                Ollama2Response=agent2.get_response()
                print("Ollama2response: " + Ollama2Response)
                agent2.add_context(f"{Ollama1Response} - {Ollama2Response}")
                incorrectCounter+=1
            bot_move=Ollama2Response
        else:
            bot_move=Ollama1Response
            
                
        print(f"Bot: {bot_move}")
        botMoveFlag=0
        for move in board.legal_moves:
            if bot_move == board.san(move):
                botMoveFlag = 1
        
        if botMoveFlag == 1:
            board.push_san(bot_move)
            agent1.add_context(f", {bot_move} ")
        else:
            print("Illegal, making random legal move")
            firstMove = list(board.legal_moves)[0]
            firstMove = board.san(firstMove)
            board.push_san(firstMove)
            agent1.add_context(f", {firstMove} ")
        display(board)
        
        agent2.set_context("")
        turn += 1
        
if __name__ == "__main__":
    handle_conversation()