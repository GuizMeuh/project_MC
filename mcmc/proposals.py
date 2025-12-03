from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
from state_space.states import StackState
from energy.energy_model import EnergyModel
#! Proposal (moves)

@dataclass
class SingleStackMove:
    """
    Container for move description
    """
    i: int #! [1, ..., N]
    j: int #! [1, ..., N]
    k_old: int #! [1, ..., N] (current height)
    k_new: int #! [1, ..., N] (proposed new height)


class Proposal(ABC):
    """
    Abstract base class for move generators
    """

    @abstractmethod
    def propose(
        self,
        state: StackState,
        energy_model: EnergyModel,
        rng: np.random.Generator,
    ) -> tuple[SingleStackMove, int]:
        """
        Propose a move starting from the given state

        Returns:
        (move, delta_E)
        - move : a description of the local change
        - delta_E: energy(state_after_move) - energy(state)
        """
        pass

class SingleStackRandomHeightProposal(Proposal):
    """
    Baseline proposal:
    - choose a random stack (i,j)
    - choose a random new height k_new != current k_old
    - compute delta_E via energy_model
    """

    def __init__(self, N: int):
        self.N = N

    def propose(
        self,
        state: StackState,
        energy_model: EnergyModel,
        rng: np.random.Generator,
    ) -> tuple[SingleStackMove, int]:

        #! pick a random stack (i,j)
        i = rng.integers(1, self.N + 1)
        j = rng.integers(1, self.N + 1)

        #! get current height
        k_old = state.get_height(i, j)

        #! sample a new height != k_old
        #* simplest: sample from [1...N] (we assume N small)
        while True:
            k_new = rng.integers(1, self.N + 1)
            if k_new != k_old:
                break

        #! compute delta_E using energy model
        delta_E = energy_model.delta_energy_single_move(state, i, j, k_new)

        move = SingleStackMove(i=i, j=j, k_old=k_old, k_new=k_new)
        return move, delta_E
<<<<<<< Updated upstream
=======


class SingleConstraintStackSwapProposal(Proposal):
    """
    Proposal for constraint stack state:
    - choose two random stacks (i1,j) and (i2,j)
    - swap their heights k1 and k2
    - compute delta_E via energy_model
    """

    def __init__(self, N: int):
        self.N = N

    def propose(
        self,
        state: ConstraintStackState,
        energy_model: EnergyModel,
        rng: np.random.Generator,
    ) -> tuple[SingleConstraintStackMove, int]:

        #! pick two random stacks (i1,j) and (i2,j)
        j = rng.integers(1, self.N + 1)
        i1 = rng.integers(1, self.N + 1)
        i2 = rng.integers(1, self.N + 1)
        while i2 == i1:
            i2 = rng.integers(1, self.N + 1)

        #! get current heights
        k1 = state.get_height(i1, j)
        k2 = state.get_height(i2, j)

        #! compute delta_E using energy model
        delta_E = energy_model.delta_energy(state, i1, i2, j, k1, k2)
        move = SingleConstraintStackMove(i1=i1, i2=i2, j=j, k1=k1, k2=k2)
        return move, delta_E



class SmartSingleStackProposal(Proposal):
    """
    Smart proposal (Min-Conflicts heuristics):
    - choose a random stack (i,j)
    - INSTEAD of random k, scan ALL possible k for this stack
    - propose the k that minimizes the energy (greediest move)
    or sample from the best candidates.
    """

    def __init__(self, N: int, random_noise: float = 0.1):
        self.N = N
        self.random_noise = random_noise

    def propose(
        self,
        state: StackState,
        energy_model: EnergyModel,
        rng: np.random.Generator,
    ) -> tuple[SingleStackMove, int]:

        #! 1. Pick a random stack (i,j)
        # Optimization: In a more advanced version, we would track 
        # which queens are attacked and pick one of them specifically.
        i = rng.integers(1, self.N + 1)
        j = rng.integers(1, self.N + 1)
        k_old = state.get_height(i, j)

        #! 2. Scan candidates to find the best k
        # We look for the move that gives the minimal Delta E
        best_k = k_old
        min_delta = float('inf')
        
        # We can test a subset or all positions. Testing all N is usually fast enough.
        candidates = []
        
        # Check all possible heights
        for k_test in range(1, self.N + 1):
            if k_test == k_old:
                continue
                
            # Calculate what the energy change WOULD be
            d_e = energy_model.delta_energy(state, i, j, k_test)
            
            if d_e < min_delta:
                min_delta = d_e
                candidates = [k_test]
            elif d_e == min_delta:
                candidates.append(k_test)

        #! 3. Select new height
        # With a small probability, pick purely random to avoid getting stuck 
        # (if you want to maintain detailed balance strictly, logic is more complex, 
        # but for optimization, this is fine).
        if rng.random() < self.random_noise or not candidates:
             # Fallback to random if noise triggers or no better move found
            k_new = rng.integers(1, self.N + 1)
            while k_new == k_old:
                k_new = rng.integers(1, self.N + 1)
            final_delta = energy_model.delta_energy(state, i, j, k_new)
        else:
            # Pick one of the best candidates
            k_new = rng.choice(candidates)
            final_delta = min_delta

        move = SingleStackMove(i=i, j=j, k_old=k_old, k_new=k_new)
        return move, final_delta
>>>>>>> Stashed changes
