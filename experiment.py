import models
import re
import csv
import torch
import numpy as np
np.random.seed(0)
torch.manual_seed(0)


class Experiment():
    def __init__(self, google_api_key: str, hugging_face_token: str, num_of_games: int = 30, num_of_rounds: int = 6, verbose: int = 1):
        """
        This method instantiate a game of Ultimatum.
        mode can be set to 1-8:
            1: LLM (gemini) vs LLM (llama)
            2: LLM (gemini) vs Human Male (llama)
            3: "Human Male" (gemini) vs LLM (llama)
            4: LLM (gemini) vs "Human Male" (llama)
            5: "Human Female" (gemini) vs LLM (llama)
            6: LLM (gemini) vs "Human Female" (llama)
            7: "Human Male" (gemini) vs "Human Female" (llama)
            8: "Human Female" (gemini) vs "Human Male" (llama)
        :param google_api_key:
        :param hugging_face_token:
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gemini = models.initialize_gemini(google_api_key=google_api_key)
        self.llama, self.llama_tokenizer = models.initialize_llama(hugging_face_token=hugging_face_token, device=self.device)
        self.games = []
        self.num_of_games = num_of_games
        self.num_of_rounds = num_of_rounds
        self.verbose = verbose
        """
        mode can be set to 1-8:
            1: LLM (gemini) vs LLM (llama)
            2: LLM (gemini) vs Human Male (llama)
            3: "Human Male" (gemini) vs LLM (llama)
            4: LLM (gemini) vs "Human Male" (llama)
            5: "Human Female" (gemini) vs LLM (llama)
            6: LLM (gemini) vs "Human Female" (llama)
            7: "Human Male" (gemini) vs "Human Female" (llama)
            8: "Human Female" (gemini) vs "Human Male" (llama)
        """
        self.role = {
            1: {"gemini": "Large Language Model", "llama": "Large Language Model"},
            2: {"gemini": "Large Language Model", "llama": "Human Male"},
            3: {"gemini": "Human Male", "llama": "Large Language Model"},
            4: {"gemini": "Large Language Model", "llama": "Human Male"},
            5: {"gemini": "Human Female", "llama": "Large Language Model"},
            6: {"gemini": "Large Language Model", "llama": "Human Female"},
            7: {"gemini": "Human Male", "llama": "Human Female"},
            8: {"gemini": "Human Female", "llama": "Human Male"},
        }

    def play(self, mode: int) -> list[list]:
        """

        """
        gemini_sum, llama_sum = 0, 0
        for game_idx in np.arange(self.num_of_games)+1:
            if self.verbose:
                print(f"running game {game_idx}")
            results = self.one_game(mode)
            gemini_sum += results[0]
            llama_sum += results[1]
            self.games.append(results)
        return {'gemini sum': gemini_sum,
                'llama sum': llama_sum,
                'history': self.games}

    def one_game(self, mode) -> tuple[float, float, dict]:
        history = []
        gemini_sum = 0
        llama_sum = 0
        for round_idx in np.arange(self.num_of_rounds)+1:
            if (round_idx % 2) == 1:
                proposer = "gemini"
                responder = "llama"
            else:
                proposer = "llama"
                responder = "gemini"
            if self.verbose:
                print(f"round {round_idx}", end=" ")
            round_result = self.one_round(proposer, responder, mode, history)
            round_result['round'] = round_idx
            # print(round_result)
            history.append(round_result)
            gemini_sum += round_result['gemini_sum']
            llama_sum += round_result['llama_sum']
        return gemini_sum, llama_sum, history

    def one_round(self, proposer, responder, mode, history) -> dict:
        """
        
        """
        proposer_role = self.role[mode][proposer]
        responder_role = self.role[mode][responder]
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
                llama_sum = responder_sum
            else:
                gemini_sum = responder_sum
                llama_sum = proposer_sum
        else:
            gemini_sum = 0
            llama_sum = 0
        if self.verbose:
            print(f"Proposal: {proposer_sum}|{responder_sum} - Reasoning: {proposal_reasoning}, Response: {response} - Reasoning: {response_reasoning}") 
        return {'proposer': proposer, 'proposer_role': proposer_role, 
                'proposal': f"{proposer_sum}|{responder_sum}", 'proposal reasoning': proposal_reasoning,
                'responder': responder, 'responder_role': responder_role, 
                'response': response, 'response reasoning': response_reasoning,
                'gemini_sum': gemini_sum,
                'llama_sum': llama_sum
               }

    def get_proposal(self, *, proposer, proposer_role, responder, responder_role, mode, history):
        """
        
        """
        string_A = f"""You are a {proposer_role} and are given insturctions by the user. 
                    Answer the insturctions with under 50 words and ONLY in the format:
                    '''
                    "proposer_sum":
                    "responder_sum":
                    "reasoning":
                    '''
                    YOU MAY NOT ANSWER WITHOUT USING THE FORMAT AND UNDER 50 WORDS.
                    Please explain clearly what is your reasoning.\n\n
                    You are playing the ultimatum game against a {responder_role}.
                    In stage 1 of the ultimatum game,
                    the first player (Proposer) proposes a specific split of a fixed
                    amount of money of $100 to the second player.
                    In stage 2 the second player (Responder) can either accept 
                    the proposed split or reject it.
                    If they accept the proposal, 
                    the fixed amount is divided according to the first mover's proposal.
                    If they choose to reject the proposal, the players do not receive anything.\n\n"""
        string_B = f"""You are the Proposer.
                    Please propose a split of $100 between yourself and the responder.
                    You may use information from previous games if any is available.
                    information from previous games: {history}\n\nIN YOUR RESPONSE, USE THE FORMAT:
                    '''
                    "proposer_sum":
                    "responder_sum":
                    "reasoning":
                    '''
                    AND UNDER 50 WORDS. 
                    FAILURE TO FOLLOW THESE INSTRUCTIONS WILL RESULT IN AN INVALID RESPONSE."""
        if proposer == 'gemini':
            raw_response = self.gemini.generate_content(string_A + string_B)
            proposal = raw_response.candidates[0].content.parts[0].text

        else:
            message = [{"role": "text", "system": string_A}, {"role": "user", "content": string_B}]
            proposal = models.call_llama(message, self.llama, self.llama_tokenizer, self.device)
            
        proposal = proposal.lower()
        return Experiment.parse_output(proposal)


    def get_response(self, *, responder, responder_role, proposer, proposer_role, proposer_sum, responder_sum, mode, history):
        """
        
        """
        string_A = f"""You are a {responder_role} and are given instructions by the user. Answer the instructions with under 50 words and ONLY in the format:

'''
"response": "accept"/"reject"
"reasoning": 
'''

YOU MAY NOT ANSWER WITHOUT USING THE FORMAT AND UNDER 50 WORDS. FAILURE TO FOLLOW THESE INSTRUCTIONS WILL RESULT IN AN INVALID RESPONSE.

Explain your reasoning clearly.

You are playing the ultimatum game against {proposer_role}. In stage 1 of the ultimatum game, the first player (Proposer) proposes a specific split of a fixed amount of money ($100) to the second player. In stage 2, the second player (Responder) can either accept the proposed split or reject it. If they accept the proposal, the fixed amount is divided according to the first mover's proposal. If they choose to reject the proposal, the players do not receive anything."""

        string_B = f"""You are the Responder. Choose whether to 'accept' or 'reject' the proposal. The proposal is {proposer_sum} for the Proposer and {responder_sum} for you. You may use information from previous games if any is available.

Information from previous games: {history}

IN YOUR RESPONSE, USE THE FORMAT 
'''
"response": "accept"/"reject"
"reasoning":
'''
AND UNDER 50 WORDS. FAILURE TO FOLLOW THESE INSTRUCTIONS WILL RESULT IN AN INVALID RESPONSE."""
                    
        if responder == 'gemini':
            raw_response = self.gemini.generate_content(string_A + '\n\n' + string_B)
            response = raw_response.candidates[0].content.parts[0].text
        else:
            message = [{"role": "text", "system": string_A}, {"role": "user", "content": string_B}]
            response = models.call_llama(message, self.llama, self.llama_tokenizer, self.device)

        response = response.lower()
        if self.verbose:
            print(responder, ":", response)
        return self.parse_response(response)

    def save_to_csv(self, game_data, save_path="game_data.csv"):
        # Define the column names
        columns = [
            "GAME", "ROUND", "PROPOSER", "PROPOSAL ROLE", "PROPOSAL", "PROPOSAL REASONING", 
            "RESPONDER", "RESPONDER ROLE", "RESPONSE", "RESPONSE REASONING", "PROPOSAL SCORE IN ROUND", 
            "RESPONDER SCORE IN ROUND", "GEMINI SCORE", "LLAMA SCORE", "PROPOSAL RUNNING SUM", "RESPONDER RUNNING SUM",
            "GEMINI AS PROPOSER RUNNING SCORE", "LLAMA AS PROPOSER RUNNING SCORE",
            "GEMINI AS RESPONDER RUNNING SCORE", "LLAMA AS RESPONDER RUNNING SCORE"
        ]

        # Open a CSV file for writing
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(columns)  # Write the column headers

            # Initialize running sums
            gemini_running_sum = 0
            llama_running_sum = 0
            gemini_proposer_running_score, llama_proposer_running_score = 0, 0
            gemini_responder_running_score, llama_responder_running_score = 0, 0
            
            # Iterate through the games and rounds to collect the data
            for game_idx, game in enumerate(game_data['history']):
                for round_data in game[2]:
                    round_idx = round_data['round']
                    proposer = round_data['proposer']
                    proposer_role = round_data['proposer_role']
                    proposal = round_data['proposal']
                    proposal_reasoning = round_data['proposal reasoning']
                    responder = round_data['responder']
                    responder_role = round_data['responder_role']
                    response = round_data['response']
                    response_reasoning = round_data['response reasoning']
                    proposal_score = float(proposal.split('|')[0])
                    responder_score = float(proposal.split('|')[1])
                    gemini_score = round_data['gemini_sum']
                    llama_score = round_data['llama_sum']

                    # Update running sums based on proposer and responder
                    if proposer == 'gemini':
                        gemini_running_sum += gemini_score
                        llama_running_sum += llama_score
                        gemini_proposer_running_score += gemini_score
                        llama_responder_running_score += llama_score
                    else:
                        gemini_running_sum += llama_score
                        llama_running_sum += gemini_score
                        llama_proposer_running_score += llama_score 
                        gemini_responder_running_score += gemini_score

                    # Write the row data
                    writer.writerow([
                        game_idx + 1, round_idx, proposer, proposer_role, proposal, proposal_reasoning,
                        responder, responder_role, response, response_reasoning, proposal_score, responder_score,
                        gemini_score, llama_score, gemini_running_sum, llama_running_sum,
                        gemini_proposer_running_score, llama_proposer_running_score,
                        gemini_responder_running_score, llama_responder_running_score
                    ])


    @staticmethod
    def parse_output(proposal):
        proposal_cleaned = re.sub(r'["\n{}]', '', proposal.split('reasoning: ')[-1]).strip()
        try:
            proposer_sum = float(re.search(r'proposer_sum\"?\:\s?\"?\$?(\d+\.?\d?\d?)', proposal.replace(" ", "")).group(1))
            responder_sum = float(re.search(r'responder_sum\"?\:\s?\"?\$?(\d+\.?\d*)', proposal.replace(" ", "")).group(1))
            if (proposer_sum + responder_sum) != 100:
                print(proposal)
                while (proposer_sum + responder_sum) != 100:
                    try:
                        proposer_sum = float(input("Please enter the proposal sum: "))
                        responder_sum = float(input("Please enter the responder sum: "))
                    except ValueError:
                        print("Invalid input. Please enter numerical values.")
            return proposer_sum, responder_sum, proposal_cleaned
        except Exception as e:
            print(e, proposal)    
            while True:
                try:
                    proposer_sum = float(input("Please enter the proposal sum: "))
                    responder_sum = float(input("Please enter the responder sum: "))
                    if (proposer_sum + responder_sum) == 100:
                        break
                except ValueError:
                    print("Invalid input. Please enter numerical values.")
            return proposer_sum, responder_sum, proposal_cleaned

    @staticmethod
    def parse_response(response: str) -> list[str]:
        try:
            decision = re.search(r'accept|reject', response.split(":")[1]).group(0)
            explaination = response[re.search(r'\"?reasoning\"?:', response).end():]
        except Exception as e:
            print(e, '\n\n', response)
            while True:
                try:
                    decision = str(input("Please enter the decision [accept/reject]: "))
                    explaination = str(input("Please enter the explaination: "))
                    if decision in ['accept', 'reject']:
                        return decision, explaination
                    else:
                        print("Invalid input. Please enter input as requested.")
                except ValueError:
                    print("Invalid input. Please enter input as requested.")
        if decision not in ['accept', 'reject']:
            print(response)
            while True:
                try:
                    decision = str(input("Please enter the decision [accept/reject]: "))
                    explaination = str(input("Please enter the explaination: "))
                    if decision not in ['accept', 'reject']:
                        print("Invalid input. Please enter input as requested.")
                    else:
                        break
                except ValueError:
                    print("Invalid input. Please enter input as requested.")
        return decision, explaination