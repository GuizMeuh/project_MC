from dataclasses import dataclass, field
from typing import List, Optional
from mcmc.chain import MCMCChain
from mcmc.proposals import SingleStackMove, SingleConstraintStackMove
from abc import ABC, abstractmethod
import numpy as np

# --- 1. L'interface enrichie ---

class AnnealingSchedule(ABC):
    @abstractmethod
    def get_temperature(self) -> float:
        pass

    @abstractmethod
    def step(self) -> None:
        """Avance d'un pas dans le temps."""
        pass
    
    @abstractmethod
    def update_metrics(self, accepted: bool, current_energy: float, best_energy: float) -> None:
        """
        Nouveau: Permet au Schedule de réagir à la performance de la chaîne.
        """
        pass

    @abstractmethod
    def is_finished(self) -> bool:
        pass

# --- 2. Le Schedule Adaptatif (La solution à ton problème) ---

@dataclass
class AdaptiveSchedule(AnnealingSchedule):
    """
    Un schedule intelligent qui:
    1. Ralentit le refroidissement si on trouve de nouvelles meilleures solutions (Cruising).
    2. Réchauffe (Reheating) si on stagne trop longtemps.
    """
    T_initial: float
    alpha: float
    min_temp: float = 0.001
    
    # Paramètres d'adaptation
    stagnation_limit: int = 3000   # Nb itérations sans amélioration avant reheat
    reheat_factor: float = 1.5     # Facteur de multiplication de T lors du reheat
    
    _current_T: float = field(init=False)
    _stagnation_counter: int = field(init=False, default=0)
    _improving_streak: bool = field(init=False, default=False)

    def __post_init__(self):
        self._current_T = self.T_initial

    def get_temperature(self) -> float:
        return self._current_T

    def update_metrics(self, accepted: bool, current_energy: float, best_energy: float) -> None:
        """
        Logique de contrôle:
        - Si on bat le record (current < best), on reset le compteur de stagnation et on active le 'streak'.
        - Sinon, on incrémente la stagnation.
        """
        if current_energy < best_energy:
            self._stagnation_counter = 0
            self._improving_streak = True
        else:
            self._improving_streak = False
            self._stagnation_counter += 1

    def step(self) -> None:
        # CAS 1: On est sur une bonne lancée (Improving Streak)
        # -> ON NE REFROIDIT PAS ! On laisse le système exploiter le filon.
        if self._improving_streak:
            return 

        # CAS 2: On est bloqué (Stagnation)
        # -> ON RÉCHAUFFE (Reheating) pour sortir du puits local.
        if self._stagnation_counter > self.stagnation_limit:
            self._current_T = min(self._current_T * self.reheat_factor, self.T_initial)
            self._stagnation_counter = 0 # Reset pour éviter de chauffer en boucle
            # Optionnel: on peut print ici pour debugger
            # print(f"  >>> REHEATING to {self._current_T:.2f}")
            return

        # CAS 3: Normal
        # -> Refroidissement géométrique classique
        self._current_T *= self.alpha

    def is_finished(self) -> bool:
        # On ajoute une sécurité: si T est très bas, on arrête
        return self._current_T <= self.min_temp


# --- 3. Compatibilité pour tes anciennes classes ---

@dataclass
class GeometricSchedule(AnnealingSchedule):
    # ... (Ton code existant) ...
    T_initial: float
    alpha: float
    max_steps: int
    max_steps_max_temp : int
    _current_step: int = field(init=False, default=0)
    _current_T: float = field(init=False)

    def __post_init__(self):
        self._current_T = self.T_initial

    def get_temperature(self) -> float:
        return self._current_T
    
    def update_metrics(self, accepted: bool, current_energy: float, best_energy: float) -> None:
        pass # Le géométrique simple s'en fiche

    def step(self) -> None:
        # ... (Ton code existant) ...
        if( self._current_step < self.max_steps_max_temp ) :
            self._current_step += 1
            return
        elif self._current_step < self.max_steps:
            self._current_T *= self.alpha
            self._current_step += 1

    def is_finished(self) -> bool:
        return self._current_step >= self.max_steps


# --- 4. La boucle principale mise à jour ---

def run_simulated_annealing(
    mcmc_chain,  
    schedule: AnnealingSchedule, 
    rng: np.random.Generator,
    verbose_every: int = 1000,
    detailed_stats : bool = False
) -> None:
    
    iteration = 0
    best_energy = float('inf') # On garde une trace du meilleur absolu pour le schedule
    
    while not schedule.is_finished():
        T = schedule.get_temperature()
        
        # 1. Capture l'énergie AVANT
        energy_before = mcmc_chain.energy_model.current_energy
        
        # 2. Step
        # Idéalement, mcmc_chain.step devrait retourner 'accepted' (bool).
        # Si ta méthode step() retourne None, on le déduit de l'énergie.
        result = mcmc_chain.step(rng, T) 
        
        # 3. Capture l'énergie APRÈS
        energy_after = mcmc_chain.energy_model.current_energy
        
        # Déduction de l'acceptation si step() ne le renvoie pas explicitement
        accepted = (energy_after != energy_before) 
        if isinstance(result, bool): 
            accepted = result
        
        # Mise à jour du meilleur global
        if energy_after < best_energy:
            best_energy = energy_after

        # 4. FEEDBACK au schedule (C'est la clé !)
        schedule.update_metrics(accepted, energy_after, best_energy)
        
        # 5. Mise à jour de la température
        schedule.step()
        
        # Logging
        if iteration % verbose_every == 0:
            if not detailed_stats : 
                attacked = mcmc_chain.energy_model.count_attacked_queens(mcmc_chain.state)
                # Ajout d'un marqueur visuel si on reheat
                status_symbol = "^" if getattr(schedule, '_stagnation_counter', 0) == 0 and energy_after < best_energy else ""
                print(f"Iter {iteration} {status_symbol}, "
                        f"T={T:.4f}, "
                        f"Energy={energy_after:.4f}, "
                        f"Attacked={attacked}")
            else : 
                attacked_stats = mcmc_chain.energy_model.attacked_stats(mcmc_chain.state)
                print(
                    f'Iter {iteration}, '
                    f'T={T:.4f}, '
                    f'Energy={energy_after:.4f}, '
                    f'Attacked={attacked_stats["attacked_queens"]}'
                )

            if energy_after == 0:
                print(f"Solution found at iteration {iteration}!")
                break
                
        iteration += 1
        
    print("Simulated Annealing complete.")


def calibrate_initial_temperature(
    mcmc_chain: 'MCMCChain',
    target_acceptance_rate: float = 0.8,
    n_samples: int = 1000,
    quantile: float = 0.9,
    rng: np.random.Generator = np.random.default_rng()
) -> float:
    """
    Calibrates the initial temperature T0 for Simulated Annealing (SA)
    using a pilot random walk.
    """
    energy_increases = []

    # Copy the initial state (deep copy if needed)
    current_state = mcmc_chain.state.copy()
    energy_model = mcmc_chain.energy_model
    proposal = mcmc_chain.proposal

    for _ in range(n_samples):

        move, delta_E = proposal.propose(current_state, energy_model, rng)

        # Store ΔE > 0
        if delta_E > 0:
            energy_increases.append(delta_E)

        # Apply move unconditionally (pilot random walk)
        energy_model.apply_move(
            current_state,
            *(move.i, move.j, move.k_new) if isinstance(move, SingleStackMove)
            else (move.i1, move.i2, move.j, move.k1, move.k2),
            delta_E
        )



    if not energy_increases:
        print("Warning: no positive ΔE found. Returning default T0 = 10.")
        return 10.0

    # Use quantile, not mean
    delta_E_star = np.quantile(energy_increases, quantile)

    T0 = -delta_E_star / np.log(target_acceptance_rate)

    print(f"ΔE_{quantile*100:.0f} percentile = {delta_E_star:.3f}")
    print(f"Initial temperature T0 = {T0:.3f}")

    return T0

