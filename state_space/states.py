from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional
import numpy as np


Coord3D = Tuple[int, int, int]  # ? (i, j, k)


class State(ABC):
    """
    Abstract interface for a 3D board configuration
    """

    def __init__(self, N: int):
        self.N = N

    @abstractmethod
    def copy(self) -> "State":
        """
        Deep copy of the state
        """
        pass

    @abstractmethod
    def iter_queens(self) -> Iterable[Coord3D]:
        """
        Iterate over all queen positions (i, j, k)
        """
        pass

    @abstractmethod
    def as_array(self):
        """
        Return an array-like representation (for logging / plotting)
        """
        pass


@dataclass
class StackState(State):
    """
    One queen per "vertical stack" (i,j,*)
    Internally stored as heights[i, j] = k in {1,...,N}
    """

    heights: np.ndarray  #! shape (N, N), values in [1..N]

    def __init__(self, heights: np.ndarray):

        N = heights.shape[0]
        super().__init__(N)

        # ? Heights as a (N x N) matrix
        assert heights.shape == (N, N)
        self.heights = heights

    @classmethod
    def random(cls, N: int):
        """
        Uniformly random state
        """

        rng = np.random.default_rng()
        heights = rng.integers(1, N + 1, size=(N, N))

        return cls(heights)

    def copy(self):
        """
        Deep copy
        """
        return StackState(self.heights.copy())

    def iter_queens(self) -> Iterable[Coord3D]:
        """
        Provides the full set of queen coordinates [do better]
        Interface for the energy model to counts conflicts, initialize line counts, ...
        """
        for i in range(self.N):
            for j in range(self.N):
                k = self.heights[i, j]
                yield (i + 1, j + 1, int(k))

    def as_array(self) -> np.ndarray:
        return self.heights

    #! Helpers for proposal

    def get_height(self, i: int, j: int) -> int:
        """
        Get queen's height for stack (i,j)
        """
        return int(self.heights[i - 1, j - 1])

    def set_height(self, i: int, j: int, k: int) -> None:
        """
        Set the queen's height at stack (i,j) to k
        """
        assert 1 <= k <= self.N
        self.heights[i - 1, j - 1] = k


@dataclass
class ConstraintStackState(State):
    """
    One queen per "vertical stack" (i,j,*)
    Internally stored as heights[i, j] = k in {1,...,N}
    """

    heights: np.ndarray  #! shape (N, N), values in [1..N]

    def __init__(self, heights: np.ndarray):

        N = heights.shape[0]
        super().__init__(N)

        # ? Heights as a (N x N) matrix
        assert heights.shape == (N, N)
        self.heights = heights

    @classmethod
    def random(cls, N: int):
        """
        Uniformly random state
        """
        # TODO : implement constraints

        rng = np.random.default_rng()
        heights = rng.integers(1, N + 1, size=(N, N))

        return cls(heights)

    def copy(self):
        """
        Deep copy
        """
        return StackState(self.heights.copy())

    def iter_queens(self) -> Iterable[Coord3D]:
        """
        Provides the full set of queen coordinates [do better]
        Interface for the energy model to counts conflicts, initialize line counts, ...
        """
        for i in range(self.N):
            for j in range(self.N):
                k = self.heights[i, j]
                yield (i + 1, j + 1, int(k))

    def as_array(self) -> np.ndarray:
        return self.heights

    #! Helpers for proposal

    def get_height(self, i: int, j: int) -> int:
        """
        Get queen's height for stack (i,j)
        """
        return int(self.heights[i - 1, j - 1])

    def set_height(self, i: int, j: int, k: int) -> None:
        """
        Set the queen's height at stack (i,j) to k
        """
        assert 1 <= k <= self.N
        self.heights[i - 1, j - 1] = k
