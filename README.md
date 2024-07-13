# Multi-Agent AI Systems - LLMs Arena - Game-Theory 

This project implements an experiment framework for studying interactions between large language models (LLMs) in the context of the Ultimatum game.
Uses Pytorch - CUDA, hugging face Transformers, Google's generative-ai, pandas, numpy.

## Key Features

- Supports experiments with different LLM combinations (Gemini and LLaMA)
- Configurable roles (LLM, Human Male, Human Female)
- Customizable number of games and rounds
- Detailed logging and CSV output

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

title = {Multi-Agent AI Systems - LLMs Arena},

year = {2024},

url = {https://github.com/RZdataprojects/llms-arena}
}
