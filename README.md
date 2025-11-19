# Causal-Informed Recommendation System

This project implements a recommendation engine that moves from reactive prediction to proactive intervention, optimizing for long-term interest diversity using Causal Inference and Reinforcement Learning.

## Project Structure

- `src/data_loader.py`: Loads and processes the MIND dataset, extracting User (U), Item (I), Action (A), and Outcome (Y) tensors.
- `src/causal_discovery.py`: Learns the Structural Causal Model (DAG) using Hill-Climbing with BIC and tiered constraints.
- `src/counterfactual_engine.py`: Estimates the diversity reward using a neural network (simulating counterfactual outcomes).
- `src/rl_agent.py`: Implements an Actor-Critic RL agent with a composite reward function (Engagement + Diversity).
- `src/evaluation.py`: Calculates metrics like NDCG, Precision, ILD, and Homogeneity Score.
- `main.py`: Orchestrates the entire pipeline.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the project:
   ```bash
   python main.py
   ```

## Data

The project is configured to generate mock data for testing purposes if the MIND dataset is not found in the `data/` directory. To use the real MIND dataset, place `news.tsv` and `behaviors.tsv` in the `data/` folder.
