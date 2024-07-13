import experiment
import argparse
import torch
import numpy as np
np.random.seed(0)
torch.manual_seed(0)


def main():
    parser = argparse.ArgumentParser(description='Run an LLM vs LLM experiment using the Ultimatum game and save the results.')
    parser.add_argument('--google_api_key', type=str, required=True, help='Google API key')
    parser.add_argument('--hugging_face_token', type=str, required=True, help='Hugging Face token')
    parser.add_argument('--mode', type=int, required=True, help="""mode can be set to 1-8:
            1: LLM (gemini) vs LLM (llama)
            2: LLM (gemini) vs Human Male (llama)
            3: "Human Male" (gemini) vs LLM (llama)
            4: LLM (gemini) vs "Human Male" (llama)
            5: "Human Female" (gemini) vs LLM (llama)
            6: LLM (gemini) vs "Human Female" (llama)
            7: "Human Male" (gemini) vs "Human Female" (llama)
            8: "Human Female" (gemini) vs "Human Male" (llama)""")
    parser.add_argument('--num_of_games', type=int, default=30, help='Number of games in a single experiment.')
    parser.add_argument('--num_of_rounds', type=int, default=6, help='Number of rounds in a game.')
    parser.add_argument('--save_path', type=str, default='./', help='Path to save the game data')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity')

    args = parser.parse_args()

    e = experiment.Experiment(google_api_key=args.google_api_key, hugging_face_token=args.hugging_face_token, num_of_games=args.num_of_games, num_of_rounds=args.num_of_rounds, verbose=1)
    game = e.play(mode=args.mode)
    e.save_to_csv(game, save_path=args.save_path + "game_data.csv")
    print(f'Game saved as "game_data.csv" in {args.save_path}')

if __name__ == "__main__":
    main()