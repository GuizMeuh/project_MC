#! Mcmc chain
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List
from state_space.states import StackState, ConstraintStackState
from energy.energy_model import EnergyModel
from mcmc.proposals import Proposal, SingleStackMove, SingleConstraintStackMove


@dataclass
class MCMCChain:
    """
    MCMC chain container
    """

    state: StackState | ConstraintStackState
    energy_model: EnergyModel
    proposal: Proposal

    def acceptance_probability(self, delta_E: int, T: float) -> float:
        """
        Compute acceptance probability for a given energy change at temperature T
        """

        if delta_E <= 0:
            return 1.0
        else:
            return np.exp(-delta_E / T)

    def step(self, rng: np.random.Generator, T: float) -> None:
        """
        Perform a single MCMC step at temperature T
        """

        #! Propose a move
        move, delta_E = self.proposal.propose(self.state, self.energy_model, rng)

        #! Compute acceptance probability
        acc_prob = self.acceptance_probability(delta_E, T)

        #! Accept or reject
        if rng.uniform(0, 1) < acc_prob:
            # ? Accept the move
            if isinstance(move, SingleStackMove):
                # ? Update state
                self.energy_model.apply_move(
                    self.state, move.i, move.j, move.k_new, delta_E
                )
            elif isinstance(move, SingleConstraintStackMove):
                # ? Swap heights in constrained stacks
                self.energy_model.apply_move(
                    self.state,
                    move.i1,
                    move.i2,
                    move.j,
                    move.k1,
                    move.k2,
                    delta_E,
                )
