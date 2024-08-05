# Ultimatum Arena 
### Multi-Agent AI Systems in Game-Theory 2024

This project implements an experiment framework for studying interactions between large language models (LLMs) in the context of the [Ultimatum Game](https://en.wikipedia.org/wiki/Ultimatum_game).
Utilizes: Pytorch - CUDA, hugging face Transformers, Google's generative-ai, pandas, numpy.

## Key Features

- Supports experiments with different LLM combinations (Gemini and LLaMA)
- Configurable roles (LLM, Human Male, Human Female)
- Customizable number of games and rounds
- Detailed logging and CSV output
- LLMs are not notified if there are further rounds, but they can view outputs of previous rounds
- If there is more than a single round, the players switch roles (Proposer/Responder), with Gemini starting each game by default

## Setup

1. Clone the repository
2. Install dependencies:
```
conda env create -f environment.yaml
conda activate llms-arena
```

3. Run an experiment:
```
python main.py --google_api_key YOUR_GOOGLE_API_KEY --hugging_face_token YOUR_HUGGING_FACE_TOKEN --mode 1-8 --num_of_games 30 --num_of_rounds 6 --save_path ./PATH/
```

## Ethical Considerations & Disclaimer

This project involves simulating human behavior using AI models. Care should be taken when interpreting results, and findings should not be generalized to actual human behavior without further validation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:

@software{LLMs_Arena,

author = {Roei Zaady},

title = {Ultimatum Arena - Multi-Agent AI Systems},

year = {2024},

url = {https://github.com/RZdataprojects/Multi-Agent-AI-Systems}
}
