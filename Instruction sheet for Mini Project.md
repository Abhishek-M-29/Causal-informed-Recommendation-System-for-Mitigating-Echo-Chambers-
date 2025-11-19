# **Master Construction Blueprint: Causal-Informed Recommendation System**

Project Title: Causal-Informed Recommendation System for Mitigating Echo Chambers  
Goal: Build a recommendation engine that moves from reactive prediction (predicting clicks) to proactive intervention (optimizing for long-term interest diversity).

## **1\. System Architecture: The "Causal-RL Loop"**

The system architecture is a closed loop where causal inference informs the reward signal for a reinforcement learning agent.

### **High-Level Data Flow**

1. **Input:** User interaction logs (e.g., MIND Dataset).  
2. **Module 1: Causal Discovery ("The Brain"):** Learns a Structural Causal Model (DAG) to understand how user features, item features, and recommendations interact causally, not just correlationally.  
3. **Module 2: Counterfactual Engine ("The Simulator"):** Queries the SCM to ask: *"What would happen to the user's diversity if I showed Item X instead of Item Y?"*  
4. **Module 3: RL Policy ("The Decision Maker"):** An Actor-Critic agent uses the signal from Module 2 to balance short-term clicks with long-term diversity.  
5. **Output:** A ranked list of news articles that breaks filter bubbles.

## **2\. Step-by-Step Build Instructions**

### **Phase 1: Environment & Data Setup**

**Objective:** Prepare the workspace and the MIND dataset for causal analysis.

* **Tech Stack:** Python 3.9+, Pandas, NumPy, Scikit-learn.  
* **Dataset:** Microsoft News Dataset (MIND).

**Instruction Details:**

1. **Data Acquisition:** Load the MIND dataset (users, news, behaviors).  
2. **Variable Extraction:** Create a processing pipeline to extract four specific tensor sets:  
   * $U$ **(User Features):** History embedding, dwell time, click counts.  
   * $I$ **(Item Features):** Category, sentiment score, entity embeddings.  
   * $A$ **(Action/Treatment):** The recommended item ID.  
   * $Y$ **(Outcome):** Binary click label AND a computed Interest\_Diversity\_Score.  
3. **Diversity Metric:** Implement a function to calculate Interest\_Diversity\_Score (e.g., 1 \- Cosine Similarity between candidate item and user history).

**Prompt for LLM/Coder:** "Write a Python script using Pandas to load the MIND dataset. Extract user features (history), item features (category, sentiment), and interaction labels. Define a 'Homogeneity Score' function that calculates the cosine similarity between a candidate news item and the user's history vector."

### **Phase 2: Causal Discovery (The SCM)**

**Objective:** Build a Directed Acyclic Graph (DAG) that maps cause-and-effect.

* **Tech Stack:** causal-learn or dowhy.  
* **Algorithm:** Hill-Climbing (HC) with Bayesian Information Criterion (BIC).

**Instruction Details:**

1. **Define Domain Constraints (Tiers):** Enforce a hierarchy to prevent illogical edges (e.g., a click cannot cause a user's age).  
   * *Tier 1:* Static User Features ($U$)  
   * *Tier 2:* Item Features ($I$)  
   * *Tier 3:* Action ($A$)  
   * *Tier 4:* Outcome ($Y$)  
2. **Structure Learning:** Run the Hill-Climbing algorithm on the observational data to learn the DAG structure.  
3. **Validation:** Output the Adjacency Matrix and visualize the graph.

**Prompt for LLM/Coder:** "Using the causal-learn library, implement a constraint-based Hill-Climbing algorithm. Enforce a tiered structure where User Features are parents of Actions, and Actions are parents of Outcomes. Output the final Adjacency Matrix representing the Causal Graph."

### **Phase 3: The Counterfactual Engine**

**Objective:** Create a distinct module that estimates the "diversity reward" for the RL agent.

* **Tech Stack:** PyTorch or TensorFlow.

**Instruction Details:**

1. **Input Interface:** The engine must accept a User Vector $u$ and a Candidate Item Vector $i'$.  
2. **Intervention Logic:** It must answer the counterfactual question: $E\[Y\_{diversity} | do(A=i'), U=u\]$.  
3. **Estimator:** Implement a Propensity Score Matching or a Deep Learning estimator (like TARNet or Dragonnet) to predict the potential diversity outcome if the intervention were applied.  
4. **Output:** A scalar value representing the predicted impact on user diversity.

**Prompt for LLM/Coder:** "Create a Python class CounterfactualEngine. It should load the learned Causal Graph. Implement a method estimate\_effect(user\_vector, item\_vector) that uses Inverse Probability Weighting to predict the user's future diversity score if that item is recommended."

### **Phase 4: Reinforcement Learning (The Agent)**

**Objective:** Train the policy that actually picks the recommendations.

* **Tech Stack:** PyTorch (Stable Baselines3 or custom implementation).  
* **Architecture:** Actor-Critic (A2C or PPO).

**Instruction Details:**

1. **MDP Setup:**  
   * **State (**$s$**):** User profile vector \+ current diversity score.  
   * **Action (**$a$**):** Selection of a news article from the candidate pool.  
2. Composite Reward Function: This is the critical component. Implement:  
   $$r \= w \\cdot r\_{engagement} \+ (1-w) \\cdot r\_{diversity}$$  
   * $r\_{engagement}$: Binary click (1 or 0\) or click probability.  
   * $r\_{diversity}$: The output from the **Counterfactual Engine** (Phase 3).  
   * $w$: Hyperparameter (e.g., 0.7) balancing the trade-off.  
3. **Training:** Train the agent to maximize cumulative $r$ over a session trajectory.

**Prompt for LLM/Coder:** "Build a PyTorch Actor-Critic RL agent. The get\_reward function must combine a binary click signal with a continuous diversity signal. The loss function should optimize the policy $\\pi(a|s)$ to maximize cumulative reward over a session."

### **Phase 5: Evaluation & Metrics**

**Objective:** Validate the system offline.

**Instruction Details:**

1. **Accuracy Metrics:** Calculate NDCG@10 and Precision@10 (standard engagement).  
2. **Diversity Metrics:**  
   * **Intra-list Diversity (ILD):** Average dissimilarity between recommended items.  
   * **Homogeneity Score:** Similarity of recommendations to user history.  
3. **Benchmark:** Compare results against a standard Collaborative Filtering baseline. The goal is to show higher Diversity metrics with comparable Accuracy.

## **3\. Developer Checklist**

| Component | Tool/Algorithm | Input | Output |
| :---- | :---- | :---- | :---- |
| **Data Prep** | Pandas/MIND | Raw Logs | Tensors ($U, I, A, Y$) |
| **Discovery** | Hill-Climbing \+ BIC | Observational Data | Causal DAG Structure |
| **Estimator** | Causal Inference | DAG \+ User \+ Item | Predicted Diversity Impact |
| **RL Policy** | Actor-Critic (PPO/A2C) | User State | Recommended Item List |
| **Reward** | Weighted Formula | Click \+ Diversity Est. | Scalar Reward Signal |

