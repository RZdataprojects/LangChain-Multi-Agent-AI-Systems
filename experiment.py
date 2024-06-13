import models
import pandas as pd
import re
import torch
import numpy as np
np.random.seed(0)
torch.manual_seed(0)


class Experiment():
    def __init__(self, google_api_key: str, hugging_face_token: str, num_of_games: int = 30, num_of_rounds: int = 6):
        """
        
        """
        self.gemini = models.initialize_gemini(google_api_key=google_api_key)
        self.mistral = models.initialize_mistral(hugging_face_token=hugging_face_token)
        self.games = []
        self.num_of_games = num_of_games
        self.num_of_rounds = num_of_rounds

    def play(self, mode: int) -> list[list]:
    """
        
    """
        for game_idx in np.arange(self.num_of_games)+1:
            history = []
            game_result = self.one_game(mode, history)
            self.games.append(game_result)
        return self.games

    def one_game(self, mode) -> list[dict]:
        rounds = []
        gemini_sum = 0
        mistral_sum = 0
        for round_idx in np.arange(self.num_of_rounds)+1:
            if (round_idx % 2 == 1):
                proposer = "gemini"
                responder = "mistral"
            else:
                proposer = "mistral"
                responder = "gemini"
            round_result = self.one_round(proposer, responder, mode)
            gemini_sum += round_result.gemini_sum
            mistral_sum += round_result.mistral_sum
            rounds.append(round_result)
        return rounds

    def one_round(self, proposer, responder, mode, history) -> dict:
        """
        
        """
        proposer_role = assign_roles(mode, proposer)
        responder_role = assign_roles(mode, responder)
        proposal, reasoning = self.get_proposal(proposer=proposer,
                                                proposer_role=proposer_role,
                                                responder=responder,
                                                responder_role=responder_role,
                                                mode=mode,
                                                history=history)
        response, reasoning = self.get_response(responder=responder,
                                                responder_role=responder_role,
                                                proposer=proposer,
                                                proposer_role=proposer_role,
                                                mode=mode,
                                                history=history)
        if response == 'accept':
            proposal = proposal.split('/')
            gemini_sum = float(re.findall(r'\d+\.?\d*', proposal[0]))
            mistral_sum = float(re.findall(r'\d+\.?\d*', proposal[1]))
        else:
            gemini_sum = 0
            mistral_sum = 0
        return {'proposer': proposer, 'proposer_role': proposer_role, 'proposal': proposal,
                'responder': responder, 'responder_role': responder_role, 'response': response,
                'gemini_sum': gemini_sum,
                'mistral_sum': mistral_sum
               }

    def get_proposal(*, self, proposer, proposer_role, responder, responder_role, mode, history):
        """
        
        """
        message = HumanMessage(
                                content=[
                                    {
                                        "type": "text",
                                        "text": f"""You are a {proposer_role} and are given insturctions by the user. 
                                        Answer the insturctions with under 10 words and ONLY in the format:
                                            response:
                                            reasoning:
                                            YOU MAY NOT ANSWER WITHOUT USING THE FORMAT UNDER 10 WORDS."""
                                    },  
                                    {
                                        "type": "text", "text": """
                                        You are playing the ultimatum game against 'gemma'.
                                        In stage 1 of the ultimatum game,
                                        the first player (Proposer) proposes a specific split of a fixed amount of money of $100,
                                        to the second player.
                                        In stage 2 the second player (Responder) can either accept the proposed split or reject it.
                                        If they accept the proposal, 
                                        the fixed amount is divided according to the first mover's proposal.
                                        If they choose to reject the proposal, the players do not receive anything."""},
                                    {
                                        "type": "text", "text": """You are the proposer.
                                        Please propose a split of $100 between yourself and the responder.
                                        You may use information from previous games if any is available."""
                                    },
                                    {
                                        "type": "text", "text": f"information from previous games: {history}"
                                    }
                                ]
                            )
             
        if proposer == 'gemini':
            proposal = self.gemini.invoke([message]).content
        else:
            proposal = self.mistral.invoke([message]).content                              
        return proposal.split()[1], proposal.split()[-1]

    def get_response(*, self, responder, responder_role, proposer, proposer_role, mode, history):
        """
        
        """
        message = HumanMessage(
                                content=[
                                    {
                                        "type": "text",
                                        "text": """You are given insturctions by the user. 
                                        Answer the insturctions with under 10 words and ONLY in the format:
                                            response:
                                            reasoning:
                                            YOU MAY NOT ANSWER WITHOUT USING THE FORMAT UNDER 10 WORDS."""
                                    },  
                                    {
                                        "type": "text", "text": """
                                        You are playing the ultimatum game against 'gemma'.
                                        In stage 1 of the ultimatum game,
                                        the first player (Proposer) proposes a specific split of a fixed amount of money of $100,
                                        to the second player.
                                        In stage 2 the second player (Responder) can either accept the proposed split or reject it.
                                        If they accept the proposal, 
                                        the fixed amount is divided according to the first mover's proposal.
                                        If they choose to reject the proposal, the players do not receive anything."""},
                                    {
                                        "type": "text", "text": f"""You are the responder.
                                        Choose whether to 'accept' or 'reject' the proposal.
                                        The proposal is {99} for the Proposer and {1} for you.
                                        You may use information from previous games if any is available."""
                                    },
                                    {
                                        "type": "text", "text": f"information from previous games: Gemma: {history}"
                                    }
                                ]
                            )
        return proposal.split()[1], proposal.split()[-1]

    @staticmethod
    def assign_roles(mode, player):
        assert mode in np.arange(8)+1, /
        """
        mode can be set to 1-8:
            1: LLM (gemini) vs LLM (mistral)
            2: LLM (gemini) vs Human Male (mistral)
            3: "Human Male" (gemini) vs LLM (mistral)
            4: LLM (gemini) vs "Human Male" (mistral)
            5: "Human Female" (gemini) vs LLM (mistral)
            6: LLM (gemini) vs "Human Female" (mistral)
            7: "Human Male" (gemini) vs "Human Female" (mistral)
            8: "Human Female" (gemini) vs "Human Male" (mistral)
        """
        roles = {
            1: {"gemini": "Large Language Model", "mistral": "Large Language Model"},
            2: {"gemini": "Large Language Model", "mistral": "Human Male"},
            3: {"gemini": "Human Male", "mistral": "Large Language Model"},
            4: {"gemini": "Large Language Model", "mistral": "Human Male"},
            5: {"gemini": "Human Female", "mistral": "Large Language Model"},
            6: {"gemini": "Large Language Model", "mistral": "Human Female"},
            7: {"gemini": "Human Male", "mistral": "Human Female"},
            8: {"gemini": "Human Female", "mistral": "Human Male"},
        }

        return roles[mode][player]
        