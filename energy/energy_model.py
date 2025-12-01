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
        # ? zero counts
        self.line_counts[:] = 0

        # ? count queens on each line
        for i, j, k in state.iter_queens():

            cell_id = self.geometry.coord_to_id(i, j, k)
            for line_id in self.line_index.cell_to_lines[cell_id]:

                self.line_counts[line_id] += 1

        # ? computes energy
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
        else:
            dc = 0

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

    def apply_move(
        self, state: StackState, i: int, j: int, k_new: int, delta_E
    ) -> None:
        """
        Apply the move (i,j,k_old)->(i,j,k_new), updating counts and energy
        """

        old_k = state.get_height(i, j)
        if k_new == old_k:
            return

        board = self.geometry
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

                # old : delta_E += nchoose2(new_c) - nchoose2(c)
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

    def count_attacked_queens(self, state: StackState) -> int:
        """
        Returns the number of queens that lie on at least one line with >= 2 queens
        """
        attacked = 0
        board = self.geometry
        l_id = self.line_index

        for (i, j, k) in state.iter_queens():
            cell_id = board.coord_to_id(i, j, k)
            lines = l_id.cell_to_lines[cell_id]

            #! is this queen on any conflicting line?
            if any(self.line_counts[line_id] > 1 for line_id in lines):
                attacked += 1

        return attacked

