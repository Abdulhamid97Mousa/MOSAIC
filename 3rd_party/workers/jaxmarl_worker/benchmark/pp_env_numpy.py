"""
NumPy-only Predator-Prey environment — a faithful reimplementation of:
  MAGIC/envs/ic3net-envs/ic3net_envs/predator_prey_env.py

No gym/gym.Env dependency; uses plain numpy + Python only.

Config baked in:
  n_predators=3, n_prey=1, dim=5, vision=1, max_steps=20,
  mode='cooperative', no_stay=False  (5 actions: 0=UP,1=RIGHT,2=DOWN,3=LEFT,4=STAY)

Notable deviation from the original:
  - Cooperative mode terminates the episode when ALL predators reach the prey.
    The original only terminates in 'mixed' mode; we add it here for a fair
    comparison baseline (JAX MAGIC port mirrors this behaviour).

Spec:
  BASE         = dim*dim = 25
  OUTSIDE_CLASS = 1 + BASE = 26     (padded border cells)
  PREY_CLASS    = 2 + BASE = 27
  PREDATOR_CLASS= 3 + BASE = 28
  vocab_size    = 1 + 1 + BASE + 1 + 1 = 29
  obs shape per agent: (vocab_size, 2*vision+1, 2*vision+1) = (29,3,3)
  flattened obs dim  : 29*3*3 = 261
"""

import numpy as np


class PredatorPreyEnvNumpy:
    """Gym-free predator-prey environment using plain NumPy."""

    # ---------- class-level constants ----------
    TIMESTEP_PENALTY = -0.05
    POS_PREY_REWARD  =  0.05

    def __init__(
        self,
        n_predators: int = 3,
        n_prey:      int = 1,
        dim:         int = 5,
        vision:      int = 1,
        max_steps:   int = 20,
        mode:        str = "cooperative",
        no_stay:     bool = False,
    ):
        self.npredator = n_predators
        self.nprey     = n_prey
        self.dim       = dim
        self.vision    = vision
        self.max_steps = max_steps
        self.mode      = mode
        self.stay      = not no_stay          # stay=True → 5 actions

        self.dims = (dim, dim)

        # action space
        self.naction = 5 if self.stay else 4

        # vocabulary / grid encoding
        self.BASE            = dim * dim          # 25
        # Note: original starts OUTSIDE/PREY/PREDATOR at 1/2/3 then adds BASE
        self.OUTSIDE_CLASS   = 1 + self.BASE      # 26
        self.PREY_CLASS      = 2 + self.BASE      # 27
        self.PREDATOR_CLASS  = 3 + self.BASE      # 28
        self.vocab_size      = 1 + 1 + self.BASE + 1 + 1   # 29

        self.observation_dim = self.vocab_size * (2 * vision + 1) ** 2   # 261
        self.num_actions     = [self.naction]      # list, like MAGIC expects
        self.dim_actions     = 1
        self.num_agents      = n_predators

        self.episode_over    = False
        self.reached_prey    = np.zeros(self.npredator)
        self.predator_loc    = np.zeros((n_predators, 2), dtype=int)
        self.prey_loc        = np.zeros((n_prey,      2), dtype=int)
        self.stat            = {}

        # grid and obs buffers (populated on reset)
        self.grid                 = None
        self.empty_bool_base_grid = None
        self.bool_base_grid       = None
        self.obs                  = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset and return initial observations, shape (n_predators, 261)."""
        self.episode_over = False
        self.reached_prey = np.zeros(self.npredator)
        self.stat         = {}

        locs = self._get_coordinates()
        self.predator_loc = locs[: self.npredator].copy()
        self.prey_loc     = locs[self.npredator :].copy()

        self._set_grid()
        self.obs = self._get_obs()
        return self.obs

    def reset_epoch(self, epoch: int) -> np.ndarray:
        """Alias for reset() (trainer.py checks for this signature)."""
        return self.reset()

    def step(self, action_list):
        """
        Parameters
        ----------
        action_list : array-like of length n_predators, values in [0, naction)

        Returns
        -------
        obs          : np.ndarray (n_predators, 261)
        reward       : np.ndarray (n_predators,)
        episode_over : bool
        info         : dict
        """
        if self.episode_over:
            raise RuntimeError("Episode is done — call reset() first.")

        action = np.array(action_list, dtype=int).squeeze()
        action = np.atleast_1d(action)

        assert np.all(action < self.naction), (
            f"Actions must be in [0, {self.naction}), got {action}"
        )

        for i, a in enumerate(action):
            self._take_action(i, int(a))

        self.obs = self._get_obs()
        reward   = self._get_reward()          # may set self.episode_over

        info = {
            "predator_locs": self.predator_loc.copy(),
            "prey_locs":     self.prey_loc.copy(),
        }
        return self.obs, reward, self.episode_over, info

    def reward_terminal(self) -> np.ndarray:
        """Called by trainer after episode ends; return zero rewards."""
        return np.zeros(self.npredator)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_coordinates(self) -> np.ndarray:
        """Sample (npredator + nprey) distinct grid positions."""
        n_total = self.npredator + self.nprey
        idx = np.random.choice(np.prod(self.dims), n_total, replace=False)
        return np.vstack(np.unravel_index(idx, self.dims)).T  # shape (n_total, 2)

    def _set_grid(self):
        """Build the padded base grid and its one-hot encoding."""
        # Inner grid: cells labelled 0..BASE-1 (their linear index)
        self.grid = np.arange(self.BASE).reshape(self.dims)
        # Pad with OUTSIDE_CLASS
        self.grid = np.pad(
            self.grid, self.vision,
            mode="constant", constant_values=self.OUTSIDE_CLASS
        )
        # empty_bool_base_grid: shape (padded_dim, padded_dim, vocab_size)
        self.empty_bool_base_grid = self._onehot_initialization(self.grid)

    def _get_obs(self) -> np.ndarray:
        """
        Build current observations by stamping agent positions onto the
        base grid, then extracting (vision)-patches per predator.

        Returns shape (n_predators, 261).
        """
        self.bool_base_grid = self.empty_bool_base_grid.copy()

        for p in self.predator_loc:
            self.bool_base_grid[
                p[0] + self.vision, p[1] + self.vision, self.PREDATOR_CLASS
            ] += 1

        for p in self.prey_loc:
            self.bool_base_grid[
                p[0] + self.vision, p[1] + self.vision, self.PREY_CLASS
            ] += 1

        obs = []
        v = self.vision
        for p in self.predator_loc:
            sy = slice(p[0], p[0] + 2 * v + 1)
            sx = slice(p[1], p[1] + 2 * v + 1)
            patch = self.bool_base_grid[sy, sx]   # (3, 3, 29)
            # Transpose to (vocab_size, H, W) then flatten → 261
            obs.append(patch.transpose(2, 0, 1).reshape(-1))

        return np.stack(obs)   # (n_predators, 261)

    def _take_action(self, idx: int, act: int):
        """Apply action `act` for predator `idx`."""
        # Predators that already reached prey cannot move
        if self.reached_prey[idx] == 1:
            return

        r, c = self.predator_loc[idx]
        v     = self.vision

        # 0: UP    — row decreases
        if act == 0:
            if self.grid[max(0, r + v - 1), c + v] != self.OUTSIDE_CLASS:
                self.predator_loc[idx][0] = max(0, r - 1)

        # 1: RIGHT — col increases
        elif act == 1:
            if self.grid[r + v, min(self.dims[1] - 1, c + v + 1)] != self.OUTSIDE_CLASS:
                self.predator_loc[idx][1] = min(self.dims[1] - 1, c + 1)

        # 2: DOWN  — row increases
        elif act == 2:
            if self.grid[min(self.dims[0] - 1, r + v + 1), c + v] != self.OUTSIDE_CLASS:
                self.predator_loc[idx][0] = min(self.dims[0] - 1, r + 1)

        # 3: LEFT  — col decreases
        elif act == 3:
            if self.grid[r + v, max(0, c + v - 1)] != self.OUTSIDE_CLASS:
                self.predator_loc[idx][1] = max(0, c - 1)

        # 4: STAY  — fall-through, do nothing (matches original intent)

    def _get_reward(self) -> np.ndarray:
        """
        Compute per-agent rewards (cooperative mode):
          - Every predator gets TIMESTEP_PENALTY = -0.05
          - Predators co-located with prey get POS_PREY_REWARD * n_on_prey = 0.05 * n
          - Episode ends when ALL predators have reached the prey (cooperative termination)
        """
        reward = np.full(self.npredator, self.TIMESTEP_PENALTY)

        # For each prey, find which predators are on it
        for prey_pos in self.prey_loc:
            on_prey = np.where(
                np.all(self.predator_loc == prey_pos, axis=1)
            )[0]
            nb_on_prey = on_prey.size

            if self.mode == "cooperative":
                reward[on_prey] = self.POS_PREY_REWARD * nb_on_prey
            elif self.mode == "competitive":
                if nb_on_prey:
                    reward[on_prey] = self.POS_PREY_REWARD / nb_on_prey
            elif self.mode == "mixed":
                reward[on_prey] = 0.0   # PREY_REWARD = 0
            else:
                raise ValueError(
                    f"Unknown mode '{self.mode}'. "
                    "Choose from cooperative|competitive|mixed."
                )

            self.reached_prey[on_prey] = 1

        # Cooperative: terminate when all predators have reached the prey
        # (deviation from upstream which only terminates in 'mixed' mode)
        if self.mode == "cooperative" and np.all(self.reached_prey == 1):
            self.episode_over = True

        # Track success rate
        if np.all(self.reached_prey == 1):
            self.stat["success"] = 1
        else:
            self.stat["success"] = 0

        return reward

    def get_stat(self) -> dict:
        return self.stat

    # ------------------------------------------------------------------
    # One-hot encoding helpers (verbatim from original)
    # ------------------------------------------------------------------

    def _onehot_initialization(self, a: np.ndarray) -> np.ndarray:
        """
        Return a 3-D array where out[i, j, a[i,j]] = 1 and 0 elsewhere.
        Shape: a.shape + (vocab_size,)

        dtype=np.float64 so observations can be passed directly to
        nn.Linear (float64 weights under torch.set_default_dtype(float64))
        without an explicit cast.
        """
        ncols = self.vocab_size
        out   = np.zeros(a.shape + (ncols,), dtype=np.float64)
        out[self._all_idx(a, axis=2)] = 1.0
        return out

    def _all_idx(self, idx: np.ndarray, axis: int):
        """Build index tuple for advanced indexing on given axis."""
        grid = list(np.ogrid[tuple(map(slice, idx.shape))])  # list so .insert works on NumPy 2
        grid.insert(axis, idx)
        return tuple(grid)
