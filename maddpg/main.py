# Imports.
import matplotlib.pyplot as plt
import numpy as np
import progressbar as pb

# Other imports.
from collections import deque

# MADDPG algorithm imports.
from maddpg.hyperparams import *


def train(wrapper, env, n_episodes=int(2.7e3), autosave_every=int(1e2)):
    """ Train agents currently running the environment,
        using a Multi-agent Deep Deterministic Policy Gradient (MADDPG) algorithm wrapper.
        See <https://github.com/openai/maddpg> and <https://arxiv.org/pdf/1706.02275.pdf>.
    
    Params
    ======
        wrapper (Wrapper): Instance of the MADDPG Wrapper class
        env (UnityEnvironment): Instance of the Unity environment for DDPG agent training
        n_episodes (int): Number of episodes to train an agent (or agents)
        autosave_every (int): Threshold (or frequency) for auto-saving model weights to disk
    """
    
    # Define the widgets used in a progress bar (useful for logging).
    widgets = [
        "EPISODE: ", pb.Counter(), '/', str(n_episodes), ' ',
        pb.Percentage(), ' ',
        pb.ETA(), ' ',
        pb.Bar(marker=pb.RotatingMarker()), ' ',
        'ROLLING AVG: ', pb.FormatLabel(''), ' ',
        'HIGHEST: ', pb.FormatLabel('')]

        
    # Define a progress bar timer.
    timer = pb.ProgressBar(widgets=widgets, maxval=n_episodes).start()
    
    # Define other "utility" variables.
    total_scores = []
    scores_deque = deque(maxlen=int(1e2))
    rolling_avgs = []
    highest_score = 0.
    solved = False
    
    # Get the environment information (or brain name).
    brain_name = env.brain_names[0]
    
    for i_episode in range(1, n_episodes+1):
        current_avg = 0. if i_episode == 1 else rolling_avgs[-1]

        # Current avg.
        widgets[12] = pb.FormatLabel(str(current_avg)[:6])

        # Highest score.
        widgets[15] = pb.FormatLabel(str(highest_score)[:6])

        # Update the timer.
        timer.update(i_episode)
        
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations[:, -STATE_SIZE:]
        scores = np.zeros(NUM_AGENTS)
        wrapper.reset()
        
        while True:
            actions = wrapper.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations[:, -STATE_SIZE:]
            rewards = env_info.rewards
            dones = env_info.local_done
            wrapper.step(states, actions, rewards, next_states, dones)
            scores += rewards
            states = next_states

            if np.any(dones):
                break
                
        # Save the current maximum score.
        max_score = np.max(scores)

        scores_deque.append(max_score)
        total_scores.append(max_score)
        
        # Save the current average score.
        avg_score = np.mean(scores_deque)
        rolling_avgs.append(avg_score)
        
        # Environment is solved.
        if avg_score >= .5 and not solved:
            print('\nEnvironment solved in {:d} episodes...\tAverage score: {:.2f}'.format(
                i_episode, avg_score))
            solved = True
            wrapper.save()
            print("\rModel saved successfully.")
            highest_score = avg_score
            
        # Environment is solved (at this stage).
        if i_episode % autosave_every == 0 and solved:
            
            # Only save these weights if they are better than the ones previously saved.
            if avg_score > highest_score:
                highest_score = avg_score
                wrapper.save()
                print("\rModel updated successfully.")

    # Return both total scores and rolling averages.
    return total_scores, rolling_avgs


def test(wrapper, env, num_games=10, load_from_file=True):
    """ Test the training results by having both DDPG agents play a match.
    
    Params
    ======
        wrapper (Wrapper): Instance of the MADDPG Wrapper class
        env (UnityEnvironment): instance of Unity environment for testing
        num_games (int): Number of games to be played in the environment
        load_from_file (bool): Toggle for loading model weights from their saved location (or working directory)
    """
    
    # Load the saved models from disk.
    if load_from_file:
        wrapper.load()

    print("Agent [0]: Red Racket.")
    print("Agent [1]: Blue Racket.", end="\n\n")
    
    # Set the scores for each playable game.
    game_scores = [0 for _ in range(NUM_AGENTS)]

    # Get the environment information (or brain name).
    brain_name = env.brain_names[0]

    for i in range(1, num_games+1):
        env_info = env.reset(train_mode=False)[brain_name]   
        states = env_info.vector_observations
        scores = np.zeros(NUM_AGENTS)

        while True:
            actions = wrapper.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            scores += rewards
            dones = env_info.local_done

            if np.any(dones):
                winner = np.argmax(scores)
                game_scores[winner] += 1
                print("\rPartial gameplay score: {}".format(game_scores))
                break

            states = next_states

    print("\n\rAgent #{} Won!".format(np.argmax(game_scores)))    


def plot(scores, rolling_avgs):
    """ Plot the training results obtained from the training loop.
    
    Params
    ======
        scores (list): Maximum list among all agents in a given episode
        rolling_avgs (list): Maximum agent scores averages in the last number of episodes
    """
    
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)

    # Plot the scores using matplotlib.
    plt.style.use('ggplot')    

    plt.title('Rewards - Using MADDPG')
    plt.plot(np.arange(1, len(scores)+1), scores, label="Maximum Score", color="g")
    plt.plot(np.arange(1, len(rolling_avgs)+1), rolling_avgs, label="Rolling Average")
    
    # This line indicates the score at which the environment is considered solved.
    plt.axhline(y=.5, linestyle="-", label="Solved (0.5)")

    plt.legend(fontsize='xx-large', loc=2, borderaxespad=0., bbox_to_anchor=(1.025,1.))
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    
    # Reveal the plot.
    plt.show()
