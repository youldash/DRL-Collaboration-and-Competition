# Deep Reinforcement Learning: Collaboration and Competition

[![Twitter Follow](https://img.shields.io/twitter/follow/youldash.svg?style=social?style=plastic)](https://twitter.com/youldash)

[banana]: misc/reacher.gif "Unity ML-Agents Reacher Environment."

## License

By using this site, you agree to the **Terms of Use** that are defined in [LICENSE](https://github.com/youldash/DRL-Continuous-Control/blob/master/LICENSE).

## About

The goal of this project is to train a so-called [intelligent agent](https://en.wikipedia.org/wiki/Intelligent_agent) (in the form of a double-jointed arm) to reach target locations in a virtual world (or environment).

<div align="center">
	<img src="misc/reacher.gif" width="100%" />
</div>

This project was developed in partial fulfillment of the requirements for Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.

## Reinforcement Learning

According to **skymind.ai**, the term [Reinforcement Learning (RL)](https://skymind.ai/wiki/deep-reinforcement-learning) refers to:

> Goal-oriented algorithms, which learn how to attain a complex objective (goal) or how to maximize along a particular dimension over many steps; for example, they can maximize the points won in a game over many moves. RL algorithms can start from a blank slate, and under the right conditions, they achieve superhuman performance. Like a pet incentivized by scolding and treats, these algorithms are penalized when they make the wrong decisions and rewarded when they make the right ones – this is reinforcement.

### Deep Reinforcement Learning

[Deep Reinforcement Learning (DRL)](https://skymind.ai/wiki/deep-reinforcement-learning) combines [Artificial Neural Networks (ANNs)](https://en.wikipedia.org/wiki/Artificial_neural_network) with an RL architecture that enables software-defined agents to learn the best actions possible in virtual environment in order to attain their goals. That is, it unites function approximation and target optimization, mapping state-action pairs to expected rewards.

## Project Environment

The project development environment is based on [Unity](https://unity.com)'s [Machine Learning Agents Toolkit (ML-Agents)](https://github.com/Unity-Technologies/ml-agents). The [toolkit](https://unity3d.ai/) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents.

The project environment is referred to the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment, which can be found on the Unity ML-Agents GitHub page.

### The Reacher

In this environment, a double-jointed arm can move to target locations. A reward of `+0.1` is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

### The State Space

The observation/state space consists of `33` variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between `-1` and `1`.

### Distributed Training

For this project, two separate versions of the Unity environment are provided:

- The first version contains a **single** agent.
- The second version contains **20 identical** agents, each with its own copy of the environment.

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.

## Solving the Environment

Note that the project submission need only solve one of the two versions of the environment.

### Option 1: Solve the First Version

The task is episodic, and in order to solve the environment, the agent must get an average score of `+30` over `100` consecutive episodes.

### Option 2: Solve the Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, the agents must get an average score of `+30` (over `100` consecutive episodes, and over all agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields `20` (potentially different) scores. We then take the average of these `20` scores.
* This yields an average score for each episode (where the average is over all `20` agents).

__Example:__ Consider the plot below, which depicts the **average score** (over all `20` agents) obtained with each episode.

[average score]: misc/plot.png "Plot of average scores (over all agents) with each episode."

<div align="center">
	<img src="misc/plot.png" width="50%" />
</div>

The environment is considered **solved**, when the average (over **100** episodes) of those average scores is at least `+30`. In the case of the plot above, the environment was solved at **episode 63**, since the average of the average scores from **episodes 64** to **163** (inclusive) was greater than `+30`.

Follow the instructions below to explore the environment on your own machine! You will also learn how to use the Python API to control your agent.

### Step 1: Clone the DRLND Repository

At first, you need to follow the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your [Python](https://www.python.org) environment. These instructions can be found in `README.md` at the root of the repository. By following these instructions, you will install [PyTorch](https://pytorch.org), the ML-Agents toolkit, and a few more Python packages required to complete the project.

__For Windows users:__ The ML-Agents toolkit supports **Windows 10**. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

### Step 2: Download the Unity Environment

For this project, you will **not** need to install Unity - this is because Udacity has already built the environment for you (for convenience), and you can download it from one of the links below. You need only select the environment that matches your operating system:

#### Version 1: One (1) Agent

- __Linux__ [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip).
- __macOS__ [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip).
- __Windows (32-bit)__ [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip).
- __Windows (64-bit)__ [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip).

#### Version 2: Twenty (20) Agents

- __Linux__ [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip).
- __macOS__ [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip).
- __Windows (32-bit)__ [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip).
- __Windows (64-bit)__ [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip).

Then, place the file in the `p2_continuous-control/` folder in the DRLND GitHub repository (provided that you have already completed Step 1), and unzip (or decompress) the file.

##### For Windows Users

Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

##### For AWS Users

If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment. You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.

> To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.

### Step 3: Explore the Environment

After you have followed the instructions above, open `Continuous_Control.ipynb` (located in the `p2_continuous-control/` folder in the DRLND GitHub repository) and follow the instructions to learn how to use the Python API to control the agent.

In the last code cell of the notebook, you'll learn how to design and observe an agent that always selects random actions at each time step. The goal of this project is to create an agent that performs much better!

## Results

See [`REPORT.md`](https://github.com/youldash/DRL-Continuous-Control/blob/master/REPORT.md) for a detailed attempts made to solve the challenges presented. Feel free to use the code and notebooks in this repo for learning purposes, and perhaps to beat the attempts mentioned in the report.
