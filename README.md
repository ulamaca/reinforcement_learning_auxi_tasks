# Reinforcement Learning with Auxiliary Tasks:
- This repo contains all stuff in my lab rotation project at Autonomous Learning Group at MPI Intelligent Systems at Tuebingen
- Rotation duration: 2017.05 - 2017.08
- In the project, we built a DQN model with 2 auxiliary networks, state prediction and reward prediction networks, based on OpenAI baselines [1] library. The 2 auxiliary networks are aimed at representation learning. The models are tested in 2 Atari games, Q*bert and Alien. 
- The follow-up project:
  - Utilization of the learned model: 
    - a) Informed exploration with the state-prediction model
      - a.1) choosing the action smartly 
    - b) Planning with the two models
    - c) Successor representation using Monte Carlo estimate from the state-prediction model
- The final report of the project will be uploaded soon.

reference:

[1] OpenAI Baselines: https://github.com/openai/baselines
