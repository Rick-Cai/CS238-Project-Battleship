import numpy as np
import random
from battleship_env import BattleshipEnv

DIFFUSION_STEPS = 10

def get_current_util(bombed, known_rewards):
    util = known_rewards
    for _ in range(DIFFUSION_STEPS):
        new_util = np.zeros(util.shape)
        for i in range(util.shape[0]):
            for j in range(util.shape[1]):
                if bombed[i][j] == 0:
                    neighbors = []
                    if i-1>=0:
                        neighbors.append(util[i-1][j])
                    if i+1<=util.shape[0]-1:
                        neighbors.append(util[i+1][j])
                    if j-1>=0:
                        neighbors.append(util[i][j-1])
                    if j+1<=util.shape[1]-1:
                        neighbors.append(util[i][j+1])
                    new_util[i][j] = sum(neighbors)/len(neighbors)
        util = new_util
    for i in range(util.shape[0]):
        for j in range(util.shape[1]):
            if known_rewards[i][j] != 0:
                util[i][j] = -1e9
    return util

def approx_trial():

    env = BattleshipEnv()

    current_board = env.agent_board
    bombed = env.agent_board
    known_rewards = env.agent_board

    valid_targets = np.where(current_board == 0)
    idx = random.randint(0,len(valid_targets[0])-1)
    action_i = valid_targets[0][idx]
    action_j = valid_targets[1][idx]
    action = action_i * env.grid_size + action_j

    current_board, reward, done, info = env.step(action)

    count_steps = 1

    while not done:
        #print("Performing Step", count_steps)

        bombed[action_i][action_j] = 1
        known_rewards[action_i][action_j] = reward

        util = get_current_util(bombed, known_rewards)

        max_util = util.max()
        potential_targets = np.where(util == max_util)
        idx = random.randint(0,len(potential_targets[0])-1)
        action_i = potential_targets[0][idx]
        action_j = potential_targets[1][idx]
        action = action_i * env.grid_size + action_j

        current_board, reward, done, info = env.step(action)
        count_steps += 1

    print("Finished in", count_steps, "steps.")

    return count_steps

NO_TRIALS = 50

total_steps = 0
for _ in range(NO_TRIALS):
    total_steps += approx_trial()
print("Performed", NO_TRIALS, "trials. Average finishing steps is", total_steps/NO_TRIALS)

