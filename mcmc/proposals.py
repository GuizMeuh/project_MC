#! Mcmc chain
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
from state_space.states import StackState, ConstraintStackState
from energy.energy_model import EnergyModel

#! Proposal (moves)


@dataclass
class BlockShuffleMove:
    """
    Stocke un mouvement complexe impliquant plusieurs reines.
    """
    indices: list[tuple[int, int]] 
    old_heights: list[int]
    new_heights: list[int]
    
    # --- Attributs de compatibilité (CRITIQUES POUR EVITER L'ERREUR) ---
    # Ces valeurs par défaut permettent au reste du code (logs, main.py)
    # de lire .j ou .i sans planter avec un AttributeError.
    j: int = -1  
    i: int = -1
    k_new: int = -1
    k_old: int = -1

@dataclass
class SingleStackMove:
    """
    Container for move description
    """
    i: int  # [1, ..., N]
    j: int  # [1, ..., N]
    k_old: int  # [1, ..., N] (current height)
    k_new: int  # [1, ..., N] (proposed new height)


@dataclass
class SingleConstraintStackMove:
    """
    Container for move description with constraint
    """
    i1: int  # [1, ..., N]
    i2: int  # [1, ..., N]
    j: int   # [1, ..., N]
    k1: int  # [1, ..., N] (height of stack i1)
    k2: int  # [1, ..., N] (height of stack i2)


class Proposal(ABC):
    """
    Abstract base class for move generators
    """
    @abstractmethod
    def propose(
        self,
        state: StackState | ConstraintStackState,
        energy_model: EnergyModel,
        rng: np.random.Generator,
    ) -> tuple[SingleStackMove | SingleConstraintStackMove, int]:
        """
        Returns: (move, delta_E)
        """
        pass


class SingleStackRandomHeightProposal(Proposal):
    """
    Baseline proposal: Single column move.
    """
    def __init__(self, N: int):
        self.N = N

    def propose(
        self,
        state: StackState,
        energy_model: EnergyModel,
        rng: np.random.Generator,
    ) -> tuple[SingleStackMove, int]:

        i = rng.integers(1, self.N + 1)
        j = rng.integers(1, self.N + 1)
        k_old = state.get_height(i, j)

        while True:
            k_new = rng.integers(1, self.N + 1)
            if k_new != k_old:
                break

        delta_E = energy_model.delta_energy(state, i, j, k_new)
        move = SingleStackMove(i=i, j=j, k_old=k_old, k_new=k_new)
        return move, delta_E


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
        delta_E = energy_model.delta_energy(state, i1=i1, i2=i2, j=j, k1=k1, k2=k2)
        move = SingleConstraintStackMove(i1=i1, i2=i2, j=j, k1=k1, k2=k2)
        return move, delta_E
    

class GlobalSubcubeShuffleProposal(Proposal):
    """
    "The Blender": Sélectionne un sous-carré de la grille et mélange (shuffle)
    aléatoirement les positions des reines à l'intérieur de ce carré.
    Idéal pour casser les structures locales figées (minima locaux).
    """
    def __init__(self, N: int, radius_ratio: float = 0.08):
        self.N = N
        # Rayon du cube (au minimum 1, sinon ça ne fait rien)
        self.radius = max(1, int(N * radius_ratio))

    def propose(
        self,
        state: StackState | ConstraintStackState,
        energy_model: EnergyModel,
        rng: np.random.Generator,
    ) -> tuple[BlockShuffleMove, int]:

        # 1. Choisir un centre aléatoire pour le cube
        c_i = rng.integers(1, self.N + 1)
        c_j = rng.integers(1, self.N + 1)

        # 2. Identifier les cellules affectées (dans le carré autour du centre)
        # On gère les bords pour ne pas dépasser 1..N
        i_min = max(1, c_i - self.radius)
        i_max = min(self.N, c_i + self.radius)
        j_min = max(1, c_j - self.radius)
        j_max = min(self.N, c_j + self.radius)

        indices = []
        old_values = []

        # Collecte des données actuelles
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                indices.append((i, j))
                old_values.append(state.get_height(i, j))

        # 3. Création de la perturbation (Shuffle)
        # On copie les valeurs et on les mélange
        new_values = list(old_values)
        rng.shuffle(new_values) 
        # Note: Si tu préfères des nouvelles valeurs aléatoires (pas un swap), 
        # utilise: new_values = [rng.integers(1, self.N + 1) for _ in indices]

        # 4. Calcul de l'énergie (Méthode Robuste "Apply-Recalc-Revert")
        # Nécessaire car on change N reines d'un coup.
        
        # Sauvegarde état initial
        current_E = energy_model.current_energy
        if current_E is None:
             energy_model.initialize(state)
             current_E = energy_model.current_energy

        try:
            # A. Appliquer le Shuffle
            for (i, j), k in zip(indices, new_values):
                self._apply_height(state, i, j, k)
            
            # B. Recalculer l'énergie
            energy_model.initialize(state)
            new_E = energy_model.current_energy
            
            # C. Restaurer l'état (Backtrack) pour que la chaîne MCMC décide d'accepter ou non
            # La chaîne rappliquera le mouvement si elle l'accepte.
            for (i, j), k in zip(indices, old_values):
                self._apply_height(state, i, j, k)
            energy_model.initialize(state) # Remise au propre du modèle

            delta_E = new_E - current_E

        except Exception as e:
            print(f"Shuffle failed: {e}")
            # Fallback neutre
            return BlockShuffleMove([], [], []), 0

        move = BlockShuffleMove(indices=indices, old_heights=old_values, new_heights=new_values)
        return move, delta_E

    def _apply_height(self, state, i, j, k):
        """Helper pour modifier l'état"""
        if hasattr(state, 'stacks'):
            state.stacks[i-1][j-1] = k
        elif hasattr(state, 'set_height'):
            state.set_height(i, j, k)
        elif hasattr(state, 'queens'):
             state.queens[i-1][j-1] = k


# Dans proposals.py, classe MixedProposal
class MixedProposal(Proposal):
    def __init__(self, N, p_shuffle=0.05, p_swap=0.2): # 5% de chance de Shuffle
        self.simple = SingleStackRandomHeightProposal(N)
        self.shuffle = GlobalSubcubeShuffleProposal(N, radius_ratio=0.15)
        self.constraint_swap = SingleConstraintStackSwapProposal(N)
        self.p_shuffle = p_shuffle
        self.p_swap = p_swap  # 20% de chance de swap si applicable

    def propose(self, state, model, rng):
        r = rng.uniform(0, 1)
        if r < self.p_shuffle:
            return self.shuffle.propose(state, model, rng) # LA BOMBE
        elif(r < self.p_swap):
            return self.constraint_swap.propose(state, model, rng)  # LE SWAP
        else:
            return self.simple.propose(state, model, rng)  # L'AJUSTEMENT
        