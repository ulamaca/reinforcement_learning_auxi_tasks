# Reinforcement Learning with Auxiliary Tasks:
- This repo contains all stuff in my lab rotation project at Autonomous Learning Group at MPI Intelligent Systems at Tuebingen
- Rotation duration: 2017.05 - 2017.08
- Our main idea is to ask agents to perform auxiliary tasks to facilitate the learning process. Auxiliary tasks can help agent gain knowledge in the environment and contribute to the main task solving. Such knowledge gain can be instantiated as representation learning in neural network. Following the idea, we built a DQN model with 2 auxiliary networks: state prediction and reward prediction networks, through which we aim at building efficient and generalization representation during the RL learning loop.  
- We implemented the models based on OpenAI baselines [1] library. The models are tested in 2 Atari games, Q*bert and Alien. 
- The usage of the program can be found in ./custom
- The follow-up project:
  - Utilization of the learned model: 
    - a) Informed exploration with the state-prediction model
      - a.1) choosing the action smartly 
    - b) Planning with the two models
    - c) Successor representation using Monte Carlo estimate from the state-prediction model
- The final report of the project will be uploaded soon.

# Reference:

[1] OpenAI Baselines: https://github.com/openai/baselines
