import models
import re
from langchain_core.messages import AIMessage, HumanMessage
import torch
import numpy as np
np.random.seed(0)
torch.manual_seed(0)


class Experiment():
    def __init__(self, google_api_key: str, hugging_face_token: str, num_of_games: int = 30, num_of_rounds: int = 6, verbose: int = 1):
        """
        :param google_api_key:
        :param hugging_face_token:
        """
        self.gemini = models.initialize_gemini(google_api_key=google_api_key)
        self.mistral = models.initialize_gemini(google_api_key=google_api_key)
        self.games = []
        self.num_of_games = num_of_games
        self.num_of_rounds = num_of_rounds
        self.verbose = verbose

    def play(self, mode: int) -> list[list]:
        """

        """
        gemini_sum, mistral_sum = 0, 0
        for game_idx in np.arange(self.num_of_games)+1:
            if self.verbose:
                print(f"running game {game_idx}")
            results = self.one_game(mode)
            gemini_sum += results[0]
            mistral_sum += results[1]
            self.games.append(results)
        return {'gemini sum': gemini_sum,
                'mistral sum': mistral_sum,
                'history': self.games}

    def one_game(self, mode) -> tuple[float, float, dict]:
        history = []
        gemini_sum = 0
        mistral_sum = 0
        for round_idx in np.arange(self.num_of_rounds)+1:
            if (round_idx % 2 == 1):
                proposer = "gemini"
                responder = "mistral"
            else:
                proposer = "mistral"
                responder = "gemini"
            if self.verbose:
                print(f"round {round_idx}", end=" ")
            round_result = self.one_round(proposer, responder, mode, history)
            round_result['round'] = round_idx
            # print(round_result)
            history.append(round_result)
            gemini_sum += round_result['gemini_sum']
            mistral_sum += round_result['mistral_sum']
        return gemini_sum, mistral_sum, history

    def one_round(self, proposer, responder, mode, history) -> dict:
        """
        
        """
        proposer_role = self.assign_roles(mode, proposer)
        responder_role = self.assign_roles(mode, responder)
        if self.verbose:
            print(f"Proposer: {proposer} as {proposer_role}, Responder: {responder} as {responder_role}", end=" ")
        proposer_sum, responder_sum, proposal_reasoning = self.get_proposal(proposer=proposer,
                                                proposer_role=proposer_role,
                                                responder=responder,
                                                responder_role=responder_role,
                                                mode=mode,
                                                history=history)

        response, response_reasoning = self.get_response(responder=responder,
                                                responder_role=responder_role,
                                                proposer=proposer,
                                                proposer_role=proposer_role,
                                                proposer_sum=proposer_sum,
                                                responder_sum=responder_sum,
                                                mode=mode,
                                                history=history)
        if response == 'accept':
            if proposer == 'gemini':
                gemini_sum = proposer_sum
                mistral_sum = responder_sum
            else:
                gemini_sum = responder_sum
                mistral_sum = proposer_sum
        else:
            gemini_sum = 0
            mistral_sum = 0
        if ~self.verbose:
            print(f"Proposal: {proposer_sum}|{responder_sum} - Reasoning: {proposal_reasoning}, Response: {response} - Reasoning: {response_reasoning}") 
        return {'proposer': proposer, 'proposer_role': proposer_role, 
                'proposal': f"{proposer_sum}|{responder_sum}", 'proposal reasoning': proposal_reasoning,
                'responder': responder, 'responder_role': responder_role, 
                'response': response, 'response reasoning': response_reasoning,
                'gemini_sum': gemini_sum,
                'mistral_sum': mistral_sum
               }

    def get_proposal(self, *, proposer, proposer_role, responder, responder_role, mode, history):
        """
        
        """
        message = HumanMessage(content=[
                                    {
                                        "type": "text",
                                        "text": f"""You are a {proposer_role} and are given insturctions by the user. 
                                        Answer the insturctions with under 10 words and ONLY in the format:
                                            proposer_sum:
                                            responder_sum:
                                            reasoning:
                                            YOU MAY NOT ANSWER WITHOUT USING THE FORMAT AND UNDER 10 WORDS."""
                                    },  
                                    {
                                        "type": "text", "text": f"""
                                        You are playing the ultimatum game against a {responder_role}.
                                        In stage 1 of the ultimatum game,
                                        the first player (Proposer) proposes a specific split of a fixed
                                        amount of money of $100 to the second player.
                                        In stage 2 the second player (Responder) can either accept 
                                        the proposed split or reject it.
                                        If they accept the proposal, 
                                        the fixed amount is divided according to the first mover's proposal.
                                        If they choose to reject the proposal, the players do not receive anything."""},
                                    {
                                        "type": "text", "text": """
                                        You are the proposer.
                                        Please propose a split of $100 between yourself and the responder.
                                        You may use information from previous games if any is available."""
                                    },
                                    {
                                        "type": "text", "text": f"information from previous games: {history}"
                                    }
                                ]
                            )
             
        if proposer == 'gemini':
            proposal = self.gemini.invoke([message])
        else:
            proposal = self.mistral.invoke([message])
        if isinstance(proposal, str):
            proposal = proposal.lower()
        else:
            proposal = proposal.content.lower()
        sums = re.findall(r'\d+\.?\d*', proposal)
        proposal_sum, responder_sum = float(sums[0]), float(sums[1])
        return proposal_sum, responder_sum, proposal.split('reasoning: ')[-1]

    def get_response(self, *, responder, responder_role, proposer, proposer_role, proposer_sum, responder_sum, mode, history):
        """
        
        """
        message = HumanMessage(content=[
                                    {
                                        "type": "text",
                                        "text": f"""You are a {responder_role} and are given insturctions by the user. 
                                        Answer the insturctions with under 10 words and ONLY in the format:
                                            response:
                                            reasoning:
                                            YOU MAY NOT ANSWER WITHOUT USING THE FORMAT AND UNDER 10 WORDS."""
                                    },  
                                    {
                                        "type": "text", "text": f"""
                                        You are playing the ultimatum game against {proposer_role}.
                                        In stage 1 of the ultimatum game,
                                        the first player (Proposer) proposes a specific split of a fixed
                                        amount of money of $100 to the second player.
                                        In stage 2 the second player (Responder) can either accept 
                                        the proposed split or reject it.
                                        If they accept the proposal, 
                                        the fixed amount is divided according to the first mover's proposal.
                                        If they choose to reject the proposal, the players do not receive anything."""},
                                    {
                                        "type": "text", "text": f"""You are the responder.
                                        Choose whether to 'accept' or 'reject' the proposal.
                                        The proposal is {proposer_sum} for the Proposer and {responder_sum} for you.
                                        You may use information from previous games if any is available."""
                                    },
                                    {
                                        "type": "text", "text": f"information from previous games: {history}"
                                    }
                                ]
                             )
        if responder == 'gemini':
            response = self.gemini.invoke([message])
        else:
            response = self.mistral.invoke([message])
        if isinstance(response, str):
            response = response.lower()
        else:
            response = response.content.lower()
        return response.split()[1], response.split('reasoning: ')[-1]

    @staticmethod
    def assign_roles(mode, player):
        assert mode in np.arange(8) + 1, \
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
        