# Causal-Informed Recommendation System for Mitigating Echo Chambers

This project implements a recommendation engine that moves from reactive prediction to proactive intervention, optimizing for long-term interest diversity using Causal Inference and Reinforcement Learning. It aims to mitigate echo chambers by balancing user engagement with content diversity.

## Project Structure

- **`app.py`**: Streamlit dashboard for interactive visualization of the entire pipeline.
- **`src/data_loader.py`**: Loads and processes the MIND dataset, extracting User (U), Item (I), Action (A), and Outcome (Y) tensors.
- **`src/causal_discovery.py`**: Learns the Structural Causal Model (DAG) using the GES algorithm with tiered constraints.
- **`src/counterfactual_engine.py`**: Estimates the diversity reward using a neural network (simulating counterfactual outcomes).
- **`src/rl_agent.py`**: Implements an Actor-Critic RL agent with a composite reward function (Engagement + Diversity).
- **`src/evaluation.py`**: Calculates metrics like NDCG, Precision, ILD (Intra-List Diversity), and Homogeneity Score.
- **`src/mock_data.py`**: Generates realistic mock data with causal patterns and variance to simulate user behaviors when the full MIND dataset is unavailable.
- **`main.py`**: CLI entry point for the pipeline.

## Setup & Installation

1. **Create a Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For the best graph visualization experience, install [Graphviz](https://graphviz.org/download/) and add it to your system PATH. If not installed, the app will fallback to a NetworkX-based visualization.*

## Running the Application

### Interactive Dashboard (Recommended)
Launch the Streamlit app to visualize the data, causal graph, and training progress:
```bash
streamlit run app.py
```

### Command Line Interface
Run the full pipeline in the terminal:
```bash
python main.py
```

## Data

The project is configured to automatically generate **mock data** (`data/news.tsv`, `data/behaviors.tsv`) if the MIND dataset is not found. 
- The mock data generator creates diverse user preferences and news categories to ensure statistical validity for Causal Discovery.
- To use the real [MIND Dataset](https://msnews.github.io/), download and place `news.tsv` and `behaviors.tsv` in the `data/` folder.

## Key Features
- **Causal Discovery**: Uses the GES algorithm to discover causal relationships between User features, Item features, and Outcomes (Click, Diversity).
- **Counterfactual Reasoning**: Estimates the potential diversity impact of recommending specific items.
- **Reinforcement Learning**: Optimizes a policy that balances immediate clicks (Engagement) with long-term diversity (Echo Chamber Mitigation).
- **Visualization**: Interactive graphs and training curves via Streamlit.
