import numpy as np
import random
from battleship_env import BattleshipEnv

def random_trial():

    env = BattleshipEnv()

    current_board = env.agent_board

    valid_targets = np.where(current_board == 0)
    idx = random.randint(0,len(valid_targets[0])-1)
    action = valid_targets[0][idx] * env.grid_size + valid_targets[1][idx]

    current_board, reward, done, info = env.step(action)

    count_steps = 1

    while not done:
        valid_targets = np.where(current_board == 0)
        idx = random.randint(0,len(valid_targets[0])-1)
        action = valid_targets[0][idx] * env.grid_size + valid_targets[1][idx]
        current_board, reward, done, info = env.step(action)
        count_steps += 1

    print("Finished in", count_steps, "steps.")

    return count_steps

NO_TRIALS = 50

total_steps = 0
for _ in range(NO_TRIALS):
    total_steps += random_trial()
print("Performed", NO_TRIALS, "trials. Average finishing steps is", total_steps/NO_TRIALS)