# ! Main script to run MCMC annealing
import numpy as np
import argparse
from state_space.states import StackState, ConstraintStackState
from state_space.geometry import Board, LineIndex
from energy.energy_model import EnergyModel
from mcmc.proposals import (
    SingleStackMove,
    SingleConstraintStackMove,
    SingleStackRandomHeightProposal,
)
from mcmc.chain import MCMCChain
from mcmc.annealing import AnnealingSchedule
from mcmc.proposals import SingleConstraintStackSwapProposal


def main(
    N: int,
    number_of_steps: int,
    rng_seed: int,
    temperatures: list,
    T_initial: float = None,
    alpha: float = None,
    max_steps: int = None,
    verbose_every: int = 1000,
) -> None:
    # ? Initialize random number generator
    rng = np.random.default_rng(rng_seed)

    # ? Initialize state (example: empty stacks)
    """ Ici il pourrait être intéressant d'ajouter une logique pour initialiser des états différents respectant certaines contraintes """
    initial_heights = np.ones((N, N), dtype=int)
    state = StackState(heights=initial_heights)

    # ? Initialize energy model (example: dummy geometry and line index)
    geometry = Board(N=N)  # ? Initialize Board object
    line_index = LineIndex(
        geometry=geometry, include_vertical=False
    )  # ? Initialize LineIndex object
    energy_model = EnergyModel(geometry=geometry, line_index=line_index)
    energy_model.initialize(state)

    # ? Initialize proposal mechanism
    proposal = SingleStackRandomHeightProposal(
        N
    )  # ? Replace with actual proposal class

    # ? Initialize MCMC chain
    mcmc_chain = MCMCChain(state=state, energy_model=energy_model, proposal=proposal)

    # ? Define annealing schedule
    if T_initial is not None and alpha is not None and max_steps is not None:
        annealing_schedule = AnnealingSchedule(
            T_initial=T_initial,
            alpha=alpha,
            max_steps=max_steps,
            num_steps_per_temp=number_of_steps,  # optionnel, ignoré en mode géométrique
            verbose_every=verbose_every,
        )
    else:
        annealing_schedule = AnnealingSchedule(
            temperatures=temperatures,
            num_steps_per_temp=number_of_steps,
            verbose_every=verbose_every,
        )

    # ? Run annealing
    annealing_schedule.run(mcmc_chain, rng)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCMC Annealing for Stack States")
    parser.add_argument("--N", type=int, default=8, help="Size of the board (N x N)")
    parser.add_argument("--steps", type=int, default=10000, help="Number of MCMC steps")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random number generator seed"
    )
    parser.add_argument(
        "--temperatures",
        type=str,
        default="100",
        help="Comma-separated list of temperatures (for discrete annealing)",
    )
    parser.add_argument(
        "--T_initial",
        type=float,
        default=None,
        help="Initial temperature for geometric annealing",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Cooling factor for geometric annealing",
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Max steps for geometric annealing"
    )
    parser.add_argument(
        "--verbose_every",
        type=int,
        default=1000,
        help="Print energy/diagnostics every n steps"
    )

    args = parser.parse_args()

    # Parse temperature list
    temperatures = [float(t) for t in args.temperatures.split(",") if t]

    main(
        N=args.N,
        number_of_steps=args.steps,
        rng_seed=args.seed,
        temperatures=temperatures,
        T_initial=args.T_initial,
        alpha=args.alpha,
        max_steps=args.max_steps,
        verbose_every=args.verbose_every,
    )
