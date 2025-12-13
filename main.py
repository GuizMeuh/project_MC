# ! Main script to run MCMC annealing
import os
import numpy as np
import argparse
import concurrent.futures
import time
import matplotlib.pyplot as plt
from state_space.states import StackState, ConstraintStackState
from state_space.geometry import Board, LineIndex
from energy.energy_model import EnergyModel
from mcmc.proposals import (
    SingleStackMove,
    SingleConstraintStackMove,
    SingleStackRandomHeightProposal,
    GlobalSubcubeShuffleProposal,
    MixedProposal
)
from mcmc.chain import MCMCChain
from mcmc.annealing import (
    AnnealingSchedule,
    LinearSchedule,
    AdaptiveSchedule,
    run_simulated_annealing,
    GeometricSchedule,
    calibrate_initial_temperature,
)
from mcmc.proposals import SingleConstraintStackSwapProposal


def main(
    N: int,
    number_of_steps: int,
    rng_seed: int,
    state: StackState | ConstraintStackState = StackState,
    mode_init: str = "noisy_latin_square",
    T_initial: float = None,
    alpha: float = None,
    max_steps: int = None,
    verbose_every: int = 1000,
    detailed_stats: bool = False,
    is_watched: bool = False,
    re_heat: bool = False,
) -> None:
    # ? Initialize random number generator
    rng = np.random.default_rng(rng_seed)

    # ? Initialize state (example: empty stacks)
    """ Ici il pourrait être intéressant d'ajouter une logique pour initialiser des états différents respectant certaines contraintes """
    state = state.init_state(N=N, rng=rng, mode=mode_init)

    # ? Initialize energy model (example: dummy geometry and line index)
    geometry = Board(N=N)  # ? Initialize Board object
    line_index = LineIndex(
        geometry=geometry, include_vertical=False
    )  # ? Initialize LineIndex object
    energy_model = EnergyModel(geometry=geometry, line_index=line_index)
    energy_model.initialize(state)

    # ? Initialize proposal mechanism
    #proposal = MixedProposal(N, p_shuffle=0.0001, p_swap=-1) seed 42 to have 0
    proposal = MixedProposal(N, p_shuffle=0.0001, p_swap=-1)
        
    # ? Initialize MCMC chain
    mcmc_chain = MCMCChain(state=state, energy_model=energy_model, proposal=proposal)
    mcmc_chain_calibration = MCMCChain(state=state, energy_model=energy_model, proposal=proposal)
    T_initial = calibrate_initial_temperature(
        mcmc_chain_calibration, target_acceptance_rate=0.85, n_samples=5000, rng=rng
    )
    #T_initial=100
    print(f"Calibrated initial temperature: T0 = {T_initial:.4f}")
    # ? Define annealing schedule
    if T_initial is not None and alpha is not None and max_steps is not None:
        if(re_heat == True):
            annealing_schedule = AdaptiveSchedule(
                T_initial=T_initial,
                alpha=alpha,
                reheat_ratio=0.2,
                stagnation_limit=20000,
                max_steps=max_steps
            )
        else:
            annealing_schedule = GeometricSchedule(
                T_initial=T_initial,
                alpha=alpha,
                max_steps=max_steps
            )
    else:
        print(
            "You must provide T_initial, alpha, and max_steps for geometric annealing !"
        )

    # ? Run annealing
    try:
        history = run_simulated_annealing(
            mcmc_chain,
            annealing_schedule,
            rng,
            verbose_every,
            detailed_stats,
            is_watched,
            re_heat
        )
    except Exception as e:
        print(f"Simulated Annealing failed with error: {e}")
        return {"error": str(e)}
    return history


def run_single_simulation(args_dict: dict) -> dict:
    """
    Wrapper to run a single simulation with given arguments.
    """
    seed = args_dict["rng_seed"]
    try:
        # We can override verbose_every here if needed to not flood the output
        args_dict["verbose_every"] = (
            10000000
            if args_dict.get("is_watched", False)
            else args_dict["verbose_every"]
        )
        print(f"--> Start simulation Seed {seed} on process {os.getpid()}")

        result = main(**args_dict)
        result["seed"] = seed
        print(
            f"<-- End simulation Seed {seed}. Energie: {result['energy'][-1]:.4f}, Queen attacked : {result['attacked_queens'][-1]:.4f} on process {os.getpid()} in {len(result['energy'])} steps"
        )
        return result
    except AssertionError as e:
        print(f"!! Simulation Seed {seed} failed with error: {e}")
        return {"seed": seed, "error": str(e)}


def plot_results(N: int, mode_init: str, state_type: str, results: list) -> None:
    """
    Plots the results from multiple Simulated Annealing runs.
    Highlights the Mean, the Best Run (global minimum final energy), and the Worst Run.

    Args:
        results: A list of dictionaries, where each dict must contain
                "energy" and "attacked_queens" (lists of values per step).
    """

    # 1. Set Plot Style
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Aggregate Calculation and Winner Identification ---
    final_energies = np.array([r["energy"][-1] for r in results])

    # 🏆 Identify the Global Winner and Loser Indices
    winner_index = np.argmin(final_energies)
    loser_index = np.argmax(final_energies)

    # Check if simulation lengths are consistent for MEAN/aggregate calculations
    has_aggregate = False
    try:
        all_energies = np.array([r["energy"] for r in results])
        all_queens = np.array([r["attacked_queens"] for r in results])

        mean_energy = np.mean(all_energies, axis=0)
        mean_queens = np.mean(all_queens, axis=0)

        has_aggregate = True
    except ValueError:
        print(
            "Warning: Simulation lengths are inconsistent; Mean trace will be skipped."
        )

    # --- Subplot 1: Energy over Steps ---
    ax_energy = axes[0]

    # Plot ALL individual runs (faint and transparent)
    for res in results:
        ax_energy.plot(res["energy"], color="gray", alpha=0.1, linewidth=0.8)

    # Plot the Mean Run
    if has_aggregate:
        ax_energy.plot(mean_energy, color="red", linewidth=2, label="Mean Energy")

    # 🥇 Highlight the BEST RUN (Global Winner)
    winner_energy = results[winner_index]["energy"]
    ax_energy.plot(
        winner_energy,
        color="green",
        linewidth=0.5,
        linestyle="-",
        label=f"BEST Run (Seed {results[winner_index].get('seed', 'N/A')} - Final E={final_energies[winner_index]:.2f})",
    )

    # 💀 Highlight the WORST RUN (Global Loser)
    loser_energy = results[loser_index]["energy"]
    ax_energy.plot(
        loser_energy,
        color="purple",
        linewidth=0.5,
        linestyle="--",
        label=f"WORST Run (Seed {results[loser_index].get('seed', 'N/A')} - Final E={final_energies[loser_index]:.2f})",
    )

    ax_energy.set_title("Energy Evolution Across Simulations", fontsize=14)
    ax_energy.set_xlabel("Steps")
    ax_energy.set_ylabel("Energy")
    ax_energy.legend(loc="upper right", fontsize=8)
    ax_energy.grid(True, linestyle="--", alpha=0.7)

    # --- Subplot 2: Attacked Queens (Conflicts) ---
    ax_queens = axes[1]

    # Plot ALL individual runs (faint and transparent)
    for res in results:
        ax_queens.plot(
            res["attacked_queens"], color="tab:blue", alpha=0.1, linewidth=0.8
        )

    # Plot the Mean Run
    if has_aggregate:
        ax_queens.plot(mean_queens, color="navy", linewidth=2, label="Mean Conflicts")

    # 🥇 Highlight the BEST RUN (Conflict Trajectory)
    winner_queens = results[winner_index]["attacked_queens"]
    ax_queens.plot(
        winner_queens,
        color="green",
        linewidth=0.5,
        linestyle="-",
        label=f"BEST Run Trajectory, final numbers: {winner_queens[-1]:.2f}",
    )

    # 💀 Highlight the WORST RUN (Conflict Trajectory)
    loser_queens = results[loser_index]["attacked_queens"]
    ax_queens.plot(
        loser_queens,
        color="purple",
        linewidth=0.5,
        linestyle="--",
        label=f"WORST Run Trajectory, final numbers: {loser_queens[-1]:.2f}",
    )

    ax_queens.set_title("Attacked Queens over Steps", fontsize=14)
    ax_queens.set_xlabel("Steps")
    ax_queens.set_ylabel("Number of Conflicts")
    ax_queens.legend(loc="upper right", fontsize=8)
    ax_queens.grid(True, linestyle="--", alpha=0.7)

    fig.suptitle(
        f"Simulated Annealing Results for {N}-Stacks {state_type if state_type!='stack'else ""} Problem, with {len(results)} simulations and '{mode_init}' Initialization",
        fontsize=16,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel MCMC Annealing for N-Stacks")
    parser.add_argument("--N", type=int, default=8, help="Size of the board (N x N)")
    parser.add_argument(
        "--steps", type=int, default=1000000, help="Number of MCMC steps"
    )
    parser.add_argument(
        "--base_seed", type=int, default=42, help="Random number generator seed"
    )
    parser.add_argument(
        "--state_type",
        type=str,
        default="stack",
        help="Type of state: 'stack' or 'constraint'",
    )
    parser.add_argument(
        "--mode_init",
        type=str,
        default="random_latin_square",
        help="Initialization mode: 'noisy_latin_square' or 'layer_balanced_random' or 'random_latin_square'",
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
        help="Print energy/diagnostics every n steps",
    )
    parser.add_argument(
        "--stats",
        type=bool,
        default=False,
        help="Print more detailed stats for detecting pathological cases",
    )
    parser.add_argument(
        "--n_simulations", type=int, default=10, help="Number of parallel simulations"
    )
    parser.add_argument(
        "--max_workers", type=int, default=None, help="Max number of parallel workers"
    )
    parser.add_argument(
        "--re_heat", type=bool, default=False, help="Use adaptive schedule with re-heating"
    )
    args = parser.parse_args()

    max_workers = (
        args.max_workers if args.max_workers else os.cpu_count()
    )  # ? Use all available CPUs if not specified

    simulations_configs = []

    for i in range(args.n_simulations):
        # Is watch is observed for each "batch" of simulations. For example if n_simulations=10 and max_workers=2, 5 batches will be run with is_watched=True
        should_be_watched = (
            i % max_workers == 0
        )  # Here if we don't want to have tqdm we can set it to False
        config = {
            "N": args.N,
            "number_of_steps": args.steps,
            "rng_seed": args.base_seed + i,
            "T_initial": args.T_initial,
            "mode_init": args.mode_init,
            "state": StackState if args.state_type == "stack" else ConstraintStackState,
            "alpha": args.alpha,
            "max_steps": args.max_steps,
            "verbose_every": args.verbose_every,
            "detailed_stats": args.stats,
            "is_watched": should_be_watched,
            "re_heat": args.re_heat,
        }
        simulations_configs.append(config)

    # ? Run simulations in parallel
    results = []
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_simulation, cfg): cfg
            for cfg in simulations_configs
        }
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)

    duration = time.time() - start_time

    print(f"All simulations completed in {duration:.2f} seconds.")

    # ? Plot results
    plot_results(
        N=args.N, mode_init=args.mode_init, state_type=args.state_type, results=results
    )
