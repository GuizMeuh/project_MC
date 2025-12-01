from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from mcmc.chain import MCMCChain  # Assuming MCMCChain is available
import numpy as np


@dataclass
class AnnealingSchedule:
    """
    Manages different cooling schedules for Simulated Annealing.
    Supports either a discrete list of temperatures or continuous geometric cooling.
    """

    # --- Parameters for discrete/constant schedule ---
    temperatures: List[float] = field(default_factory=list)
    num_steps_per_temp: int = 10000

    # --- Parameters for geometric (continuous) schedule ---
    T_initial: Optional[float] = None
    alpha: Optional[float] = None
    max_steps: Optional[int] = None

    # --- Internal State ---
    _current_step: int = field(init=False, default=0)
    _current_T: float = field(init=False, default=0.0)
    is_geometric: bool = field(init=False, default=False)

    # --- Verbose parameter ---
    verbose_every: int = 1000

    def __post_init__(self):
        self.is_geometric = (
            self.T_initial is not None
            and self.alpha is not None
            and self.max_steps is not None
        )

        if self.is_geometric:
            if not (0 < self.alpha < 1):
                raise ValueError(
                    "The cooling factor 'alpha' must be strictly between 0 and 1."
                )
            if self.T_initial <= 0:
                raise ValueError("The initial temperature must be positive.")
            if self.max_steps <= 0:
                raise ValueError("The maximum number of steps must be positive.")
            self._current_T = self.T_initial
        elif not self.temperatures:
            raise ValueError(
                "You must provide either a list of temperatures or parameters for geometric cooling (T_initial, alpha, max_steps)."
            )

    def _next_temperature_geometric(self) -> float:
        """Calculates the next temperature T(k+1) using T(k+1) = alpha * T(k)."""
        if self._current_step >= self.max_steps:
            return 0.0

        if self._current_step > 0:
            self._current_T *= self.alpha

        self._current_step += 1
        return self._current_T

    def run(self, mcmc_chain: MCMCChain, rng: np.random.Generator) -> None:
        """
        Runs the annealing schedule on the given MCMC chain.
        """
        if self.is_geometric:
            self._run_geometric(mcmc_chain, rng)
        else:
            self._run_discrete(mcmc_chain, rng)

    def _run_discrete(self, mcmc_chain: MCMCChain, rng: np.random.Generator) -> None:
        """Executes the chain using a predefined list of temperatures."""

        for temp_idx, T in enumerate(self.temperatures):
            for step in range(self.num_steps_per_temp):
                mcmc_chain.step(rng, T)
                if step % self.verbose_every == 0:
                    attacked = mcmc_chain.energy_model.count_attacked_queens(mcmc_chain.state)
                    print(
                        f"Step {self._current_step+1}/{self.max_steps}, "
                        f"T={T:.4f}, "
                        f"Energy={mcmc_chain.energy_model.current_energy:.4f}, "
                        f"Attacked Queens={attacked}"
                    )

    def _run_geometric(self, mcmc_chain: MCMCChain, rng: np.random.Generator) -> None:
        """Executes the chain using the geometric cooling scheme."""

        T = self.T_initial
        while T > 0.0 and self._current_step < self.max_steps:
            mcmc_chain.step(rng, T)

            if self._current_step % self.verbose_every == 0:
                attacked = mcmc_chain.energy_model.count_attacked_queens(mcmc_chain.state)
                print(
                    f"Step {self._current_step+1}/{self.max_steps}, "
                    f"T={T:.4f}, "
                    f"Energy={mcmc_chain.energy_model.current_energy:.4f}, "
                    f"Attacked Queens={attacked}"
                )

            T = self._next_temperature_geometric()

        print("Geometric Simulated Annealing complete.")
