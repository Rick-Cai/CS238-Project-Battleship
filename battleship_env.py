import gym
from gym import spaces
import numpy as np

class Ship:
    def __init__(self, size, grids):
        self.size = size
        self.grids = {grid: False for grid in grids}  # Dictionary of (row, col) to hit status
        self.is_sunk = False     # Track whether the ship has been sunk
        self.cnt_hit = 0         # Count of hits taken by this ship

    def covers_grid(self, row, col):
        # Check if this ship covers the given grid cell
        return (row, col) in self.grids

    def register_hit(self, row, col):
        # Only register the hit if this grid has not been hit before
        if not self.grids[(row, col)]:
            self.grids[(row, col)] = True
            self.cnt_hit += 1
            if self.cnt_hit == self.size:
                self.is_sunk = True  # Update is_sunk if all grids are hit

class BattleshipEnv(gym.Env):
    def __init__(self, reward_fn=None):
        super(BattleshipEnv, self).__init__()
        self.reward_fn = reward_fn
        self.grid_size = 10
        self.action_space = spaces.Discrete(self.grid_size * self.grid_size)
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(self.grid_size, self.grid_size), dtype=int
        )
        self.ships = []  # List of ships placed on the board
        self.cnt_ships_sunk = 0  # Counter to track the number of ships sunk
        self.all_ships_sunk = False  # Track whether all ships are sunk
        self._setup_game()

    def _default_reward_fn(self, agent_board, row, col, state):
        # Default reward function: Penalizes each action to encourage efficient gameplay
        return -1

    def _setup_game(self):
        # Initialize the opponent's board with ships placed
        self.opponent_board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.ships = []
        self.cnt_ships_sunk = 0  # Reset the sunk ship counter
        self.all_ships_sunk = False  # Reset all ships sunk status
        self._place_ships()
        # Initialize the agent's observation board
        self.agent_board = np.zeros((self.grid_size, self.grid_size), dtype=int)

    def _place_ships(self):
        ship_sizes = [5, 4, 3, 3, 2]
        for size in ship_sizes:
            placed = False
            while not placed:
                orientation = np.random.choice(['horizontal', 'vertical'])
                grids = []
                if orientation == 'horizontal':
                    row = np.random.randint(0, self.grid_size)
                    col_start = np.random.randint(0, self.grid_size - size + 1)
                    grids = [(row, col_start + i) for i in range(size)]
                else:
                    row_start = np.random.randint(0, self.grid_size - size + 1)
                    col = np.random.randint(0, self.grid_size)
                    grids = [(row_start + i, col) for i in range(size)]

                # Check if grids are already occupied by another ship
                if all(self.opponent_board[row, col] == 0 for row, col in grids):
                    for row, col in grids:
                        self.opponent_board[row, col] = 1  # Place the ship
                    self.ships.append(Ship(size, grids))  # Create a Ship instance
                    placed = True

    def step(self, action, reward_fn=None):
        # Use default reward function if none is provided
        if reward_fn is None:
            if self.reward_fn is None:
                reward_fn = self._default_reward_fn
            else:
                reward_fn = self.reward_fn

        if self.all_ships_sunk:
            # If all ships are sunk, the game is over and no further actions should be taken.
            return self.agent_board.copy(), 0, True, {"message": "Game over. All ships have been sunk."}

        row = action // self.grid_size
        col = action % self.grid_size
        done = False
        info = {}

        if self.agent_board[row, col] != 0:
            # Already targeted this cell, passing the state to the reward function
            reward = reward_fn(self.agent_board, row, col, 'redundant')
        else:
            if self.opponent_board[row, col] == 1:
                self.agent_board[row, col] = 2  # Mark as hit
                ship_sunk = self._register_hit_and_check_sunk(row, col)
                if ship_sunk:
                    reward = reward_fn(self.agent_board, row, col, 'sunk')
                else:
                    reward = reward_fn(self.agent_board, row, col, 'hit')
            else:
                self.agent_board[row, col] = 1  # Mark as miss
                reward = reward_fn(self.agent_board, row, col, 'miss')

        # Check if all ships are sunk by comparing cnt_ships_sunk to 5
        if self.cnt_ships_sunk == 5:
            self.all_ships_sunk = True
            done = True
            reward += reward_fn(self.agent_board, row, col, 'all_sunk')
            info = {"message": "Congratulations! All ships have been sunk."}

        return self.agent_board.copy(), reward, done, info

    def reset(self):
        self._setup_game()
        return self.agent_board.copy()

    def render(self, mode='human'):
        # Optional: Implement visualization of the board
        pass

    def close(self):
        pass

    # This function assumes that it is called only when there is a valid new hit on the grid.
    # It registers a hit on the ship covering the cell (row, col) and checks if the ship is sunk.
    def _register_hit_and_check_sunk(self, row, col):
        for ship in self.ships:
            if ship.covers_grid(row, col):
                ship.register_hit(row, col)
                if ship.is_sunk:
                    self.cnt_ships_sunk += 1  # Increment cnt_ships_sunk if the ship is sunk
                    # Mark the ship as sunk on the agent's board
                    for r, c in ship.grids:
                        self.agent_board[r, c] = 3  # Mark each grid of the ship as sunk
                return ship.is_sunk  # Return whether this specific ship has been sunk
        return False