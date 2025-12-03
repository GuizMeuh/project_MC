from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import List

from state_space.states import StackState
from state_space.geometry import Board, LineIndex

def nchoose2(n: int) -> int:
    return (n * (n - 1)) // 2

@dataclass
class EnergyModel:
    geometry: Board
    line_index: LineIndex

    def __post_init__(self):
        
        #! Nb of lines
        L = len(self.line_index.lines)
        
        #! Count the nb of queens on each line
        self.line_counts = np.zeros(L, dtype=int)
        
        #! Tot energy
        self.current_energy = 0

    def initialize(self, state: StackState) -> None:
        """
        Compute line_counts and current_energy from scratch for this state
        """
        #? zero counts
        self.line_counts[:] = 0

        #? count queens on each line
        for (i, j, k) in state.iter_queens():

            cell_id = self.geometry.coord_to_id(i, j, k)
            for line_id in self.line_index.cell_to_lines[cell_id]:

                self.line_counts[line_id] += 1

        #? computes energy
        energy = 0
        for c in self.line_counts:
            if c > 1:
                energy += nchoose2(c)

        self.current_energy = energy

    def get_energy(self) -> int:
        return int(self.current_energy)
    
    def _line_delta_energy(self, line_id, in_old, in_new):
        c = self.line_counts[line_id]

        if in_old and not in_new:
            dc = -1
        elif in_new and not in_old:
            dc = +1

        return nchoose2(c + dc) - nchoose2(c)

    def delta_energy(self, state: StackState, i: int, j: int, k_new: int) -> int:
        """Energy change if queen at (i,j) moves to new height k_new."""
        old_k = state.get_height(i, j)
        if k_new == old_k:
            return 0

        board = self.geometry
        lid = self.line_index

        cell_old = board.coord_to_id(i, j, old_k)
        cell_new = board.coord_to_id(i, j, k_new)

        old_lines = lid.cell_to_lines[cell_old]
        new_lines = lid.cell_to_lines[cell_new]

        old_set = set(old_lines)
        new_set = set(new_lines)

        delta_E = 0

        #! for every line in the union of sets
        for line_id in old_set.union(new_set):

            #! net diff count
            in_old = line_id in old_set
            in_new = line_id in new_set

            delta_E += self._line_delta_energy(line_id, in_old, in_new)

        return delta_E

<<<<<<< Updated upstream
    def apply_move(self, state: StackState, i: int, j: int, k_new: int, delta_E) -> None:
        """
        Apply the move (i,j,k_old)->(i,j,k_new), updating counts and energy
        """

        old_k = state.get_height(i, j)
        if k_new == old_k:
            return

        board = self.geometry
=======
    def delta_energy(
        self,
        state: StackState | ConstraintStackState,
        i: int = None,
        j: int = None,
        k_new: int = None,
        i1: int = None,
        i2: int = None,
        k1: int = None,
        k2: int = None,
    ) -> int:
        """
        Energy change for a proposed move.
        OPTIMIZED for StackState to avoid set creation overhead.
        """
        # --- OPTIMISATION CRITIQUE POUR STACKSTATE ---
        if isinstance(state, StackState):
            old_k = state.get_height(i, j)
            k_new_val = k_new if k_new is not None else old_k
            
            if k_new_val == old_k:
                return 0
            
            board = self.geometry
            # ATTENTION: Vérifie ici si tes i, j, k doivent être décalés (-1) ou non
            # selon ta classe Board. Si Board attend 0..N-1 et que i est 1..N :
            # cell_old = board.coord_to_id(i, j, old_k)  <-- Vérifie ça !
            
            cell_old = board.coord_to_id(i, j, old_k)
            cell_new = board.coord_to_id(i, j, k_new_val)
            
            # Accès direct aux listes (beaucoup plus rapide que les Sets)
            lines_old = self.line_index.cell_to_lines[cell_old]
            lines_new = self.line_index.cell_to_lines[cell_new]
            
            delta_E = 0
            
            # 1. Lignes quittées (on retire une reine)
            for line_id in lines_old:
                # Est-ce que cette ligne existe aussi dans la nouvelle position ?
                # On vérifie manuellement pour éviter de créer un set
                is_shared = False
                for l_new in lines_new:
                    if l_new == line_id:
                        is_shared = True
                        break
                
                if not is_shared:
                    c = self.line_counts[line_id]
                    # La formule mathématique simplifiée : Energy(c-1) - Energy(c)
                    delta_E -= (c - 1)

            # 2. Lignes rejointes (on ajoute une reine)
            for line_id in lines_new:
                is_shared = False
                for l_old in lines_old:
                    if l_old == line_id:
                        is_shared = True
                        break
                
                if not is_shared:
                    c = self.line_counts[line_id]
                    # La formule mathématique simplifiée : Energy(c+1) - Energy(c)
                    delta_E += c
            
            return delta_E

        # --- CAS COMPLEXE (Swap) : On garde l'ancienne méthode lente ---
        elif isinstance(state, ConstraintStackState):
            k1_val = k1 if k1 is not None else state.get_height(i1, j)
            k2_val = k2 if k2 is not None else state.get_height(i2, j)
            if k1_val == k2_val:
                return 0
            board = self.geometry
            cell_1_old = board.coord_to_id(i1, j, k1_val)
            cell_2_old = board.coord_to_id(i2, j, k2_val)
            cell_1_new = board.coord_to_id(i1, j, k2_val)
            cell_2_new = board.coord_to_id(i2, j, k1_val)
            return self._delta_energy_generic(
                [cell_1_old, cell_2_old], [cell_1_new, cell_2_new]
            )
        return 0
    
    def _apply_move_generic(
        self,
        affected_cells_old: list[int],
        affected_cells_new: list[int],
        delta_E: int = None,
    ) -> None:
>>>>>>> Stashed changes
        lid = self.line_index

        cell_old = board.coord_to_id(i, j, old_k)
        cell_new = board.coord_to_id(i, j, k_new)

        old_lines = lid.cell_to_lines[cell_old]
        new_lines = lid.cell_to_lines[cell_new]

        old_set = set(old_lines)
        new_set = set(new_lines)

        #! if the energy is not given, recompute it
        if delta_E == None:
            delta_E = 0

            for line_id in old_set.union(new_set):
                in_old = line_id in old_set
                in_new = line_id in new_set
                
                #old : delta_E += nchoose2(new_c) - nchoose2(c)
                delta_E += self._line_delta_energy(line_id, in_old, in_new)


        #! Update line count
        for line_id in old_set.union(new_set):
            in_old = line_id in old_set
            in_new = line_id in new_set
            if in_old and not in_new:
                self.line_counts[line_id] -= 1
            elif in_new and not in_old:
                self.line_counts[line_id] += 1


        #! Update energy and state
        self.current_energy += delta_E
        state.set_height(i, j, k_new)

