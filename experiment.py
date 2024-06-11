import models
import torch
import numpy as np
np.random.seed(0)
torch.manual_seed(0)


class Experiment():
    def __init__(self, mode, hugging_face_token: str, num_of_games: int = 30, num_of_rounds: int = 5):
        self.model_1 = models.initialize_llama(hugging_face_token=hugging_face_token)
        self.model_2 = models.initialize_gemma(hugging_face_token=hugging_face_token)
        self.games = []
        self.num_of_games = num_of_games
        self.num_of_rounds = num_of_rounds

    def play(self, mode) -> list[list]:
        for game_idx in np.arange(self.num_of_games)+1:
            game_result = self.one_game(mode)
            self.games.append(game_result)
        return self.games

    def one_game(self, mode) -> list[dict]:
        rounds = []
        for round_idx in np.arange(self.num_of_rounds)+1:
            if (round_idx % 2 == 1):
                proposer = self.model_1
                responder = self.model_2
            else:
                proposer = self.model_2
                responder = self.model_1
            round_result = self.one_round(proposer, responder, mode)
            rounds.append(round_result)
        return rounds

    def one_round(self, proposer, responder, mode) -> dict:
        proposal = self.get_proposal(proposer, mode)
        response = self.get_response(responder, proposal, mode)
        return {'proposer': proposer.__class__.__name__, 'proposal': proposal, 'responder': responder.__class__.__name__, 'response': response}

    def get_proposal(self, proposer, mode):
        user_prompt = "You are the proposer. Please propose a split of $10 between yourself and the responder."
        if proposer.__class__.__name__ == "Llama":
            proposal = get_response_meta_llama(user_prompt, model='llama-2', hugging_face_model=proposer, tokenizer=models.llama_tokenizer)
        else:
            proposal = get_response_google_gemma(user_prompt, model='gemma', hugging_face_model=proposer, tokenizer=models.gemma_tokenizer)
        return proposal

    def get_response(self, responder, proposal, mode):
        user_prompt = f"The proposer has proposed the following split: {proposal}. Do you accept or reject?"
        if responder.__class__.__name__ == "Llama":
            response = get_response_meta_llama(user_prompt, model='llama-2', hugging_face_model=responder, tokenizer=models.llama_tokenizer)
        else:
            response = get_response_google_gemma(user_prompt, model='gemma', hugging_face_model=responder, tokenizer=models.gemma_tokenizer)
        return response