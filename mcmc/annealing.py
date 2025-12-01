from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from mcmc.chain import MCMCChain  # Assuming MCMCChain is available
from abc import ABC, abstractmethod
import numpy as np




class AnnealingSchedule(ABC):
    """
    Classe abstraite définissant l'interface pour tout calendrier de refroidissement.
    """
    
    @abstractmethod
    def get_temperature(self) -> float:
        """Retourne la température actuelle."""
        pass

    @abstractmethod
    def step(self) -> None:
        """Met à jour la température pour l'itération suivante."""
        pass

    @abstractmethod
    def is_finished(self) -> bool:
        """Indique si le processus de recuit doit s'arrêter."""
        pass


@dataclass
class GeometricSchedule(AnnealingSchedule):
    """
    T(k+1) = alpha * T(k)
    """
    T_initial: float
    alpha: float
    max_steps: int
    
    _current_step: int = field(init=False, default=0)
    _current_T: float = field(init=False)

    # --- Verbose parameter ---
    verbose_every: int = 1000

    def __post_init__(self):
        if not (0 < self.alpha < 1):
            raise ValueError("Alpha doit être entre 0 et 1.")
        self._current_T = self.T_initial

    def get_temperature(self) -> float:
        return self._current_T

    def step(self) -> None:
        if self._current_step < self.max_steps:
            self._current_T *= self.alpha
            self._current_step += 1

    def is_finished(self) -> bool:
        return self._current_step >= self.max_steps or self._current_T <= 1e-9
    


@dataclass
class LinearSchedule(AnnealingSchedule):
    """
    T(k+1) = T(k) - decremenet
    """
    T_initial: float
    decrement: float
    min_temp: float = 0.001

    _current_T: float = field(init=False)

    def __post_init__(self):
        self._current_T = self.T_initial

    def get_temperature(self) -> float:
        return self._current_T

    def step(self) -> None:
        self._current_T -= self.decrement

    def is_finished(self) -> bool:
        return self._current_T <= self.min_temp
    


def run_simulated_annealing(
    mcmc_chain,  
    schedule: 'AnnealingSchedule',  # The abstract cooling schedule object
    rng: np.random.Generator,
    verbose_every
) -> None:
    """
    Run the simulated annealing process using the provided MCMC chain and cooling schedule.
    """
    iteration = 0
    
    # The loop continues as long as the cooling schedule dictates (T > T_min or steps < max_steps)
    while not schedule.is_finished():
        # 1. Retrieve the current temperature T
        T = schedule.get_temperature()
        
        # 2. Perform one MCMC step (State transition with Metropolis acceptance)
        mcmc_chain.step(rng, T) 
        
        # 3. Update the temperature for the next iteration
        schedule.step()
        
        # 4. Monitoring and Logging (every n steps)
        if iteration % verbose_every == 0:
                    attacked = mcmc_chain.energy_model.count_attacked_queens(mcmc_chain.state)
                    print(
                        f"Step {self._current_step+1}/{self.max_steps}, "
                        f"T={T:.4f}, "
                        f"Energy={mcmc_chain.energy_model.current_energy:.4f}, "
                        f"Attacked Queens={attacked}"
                    )
            
            # Early stopping condition if the perfect solution (E=0) is found
            if current_energy == 0:
                print(f"Solution found at iteration {iteration}!")
                break
                
        iteration += 1
        
    print("Simulated Annealing complete.")