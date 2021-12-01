# Reinforcement Learning (RL)

This is a self-learning project of Deep Reinforcement Learning. The highlight features of this project are:

* Picklable agent that able to resume training and apply trained agent to different environment easily.
* Generate graphs and video for evaluating performance of the agents.
* Examples of training a reinforcement learning agent that able to train agents with simple command in command prompt.
	* Customize parameters of agents by simply passing arguments in command prompt.
* Designed base class of agent, network, layer that able to implement subclass object for further customization.

## Prerequisites
* Python 3.6.7
* Numpy 1.19.5
* Tensorflow 2.6.0
* Tensorflow-probability 0.12.2
* OpenAI gym 0.18.0
* Matplotlib 3.1.1
Please note that you may need to install extra package or files for running some gym environment (such as Atari environment). For more details, please refer to [OpenAI Gym](https://github.com/openai/gym).

## Examples
To run the examples, open command prompt at the base directory (same directory as this README file), and then try the following lines for different examples:
1. Initiate a new DDQN agent for Atari Breakout environment:
```cmd
python examples/atari_ddqn.py  --gym-id BreakoutNoFrameskip-v4
```
2. Resume to training for agent stored as rl-main/trained/training_agent:
```cmd
python examples/train_folder.py --folder-name training_agent
```
3. Evaluate the performance of agent stored as rl-main/trained/training_agent by ploting graph of rewards over time steps, and save video of performance of agent with fps of 15.
```cmd
python examples/evaluate.py --folder-name training_agent --fps 15
```
4. Plot the reward over time steps for two agents, rl-main/trained/DDQN and rl-main/trained/PPO, on the same graph for comparing their performance.
```cmd
python examples/plot_metrics.py --folder-names DDQN PPO
```
Please refer to the files or --help flag for more details about arguments available for each examples.

## Algorithms Implemented
* Deep Reinforcement Learning
	* Double Deep Q-Learning (DQN)
	* Proximal Policy Gradient (PPO)
* Neural Network
	* Convolutional Neural Network (CNN)
	* Long Short-Term Memory (LSTM)
	* Multi-head Self Attention Mechanism
	* Transformer

## Agent file structure
Agent is saved as a folder under rl-main/trained with given folder name or auto-generated default name based on environment name and reinforcement learning algorithm name. rl-main/trained folder will be auto-generated if it does not exist.
The agent folder including:
* Metrics folder including:
	* Average episode rewards
	* Time steps
	* Any other customized metrics for agent
* Models folder including:
	* Files generated by Tensorflow CheckpointManager for storing weights of the neural network(s)
		* Please refer [Tensorflow document](https://www.tensorflow.org/api_docs/python/tf/train/CheckpointManager) for more details about the files
* agent.pkl for storing the structure of whole reinforcement learning agent
* summary.txt with brief description about attributes and neural network(s) of the agent

Here is a brief description of the file structure of the project:
```
rl-main
├── examples
│    └─ ...
├── rl
│    └─ ...
├── trained
│      ├─ agent
│      │      ├ metrics
│      │      │     ├ Average Episode Reward.pkl
│      │      │     └ Time Steps.pkl
│      │      ├ models
│      │      │     │ checkpoint
│      │      │     ├ ckpt-xx.data-00000-of-00001
│      │      │     ├ ckpt-xx.index
│      │      │     └ ...
│      │      ├ agents.pkl
│      │      └ summary.txt
│      └─ ...
└── README.md
```