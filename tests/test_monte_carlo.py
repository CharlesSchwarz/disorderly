import warnings

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from disorderly.monte_carlo import (
    KB_EV_PER_K,
    _metropolis_accept,
    move_ion,
    propose_next_configuration,
    resume_monte_carlo,
    run_monte_carlo,
    swap_ion,
)
from disorderly.io import load_checkpoint, save_checkpoint
from disorderly.results import MonteCarloStep, MonteCarloTrajectory
from disorderly.settings import (
    DetailedBalanceWarning,
    MonteCarloPolicy,
    MoveProposal,
    default_policy,
    sublattice_swap_policy,
)

ROCK_SALT_COORDS = [
    [0, 0, 0],
    [0, 0.5, 0.5],
    [0.5, 0, 0.5],
    [0.5, 0.5, 0],
    [0.5, 0.5, 0.5],
    [0.5, 0, 0],
    [0, 0.5, 0],
    [0, 0, 0.5],
]


@pytest.fixture
def na_k_cl() -> Structure:
    """Rock salt with a mixed Na+/K+ cation sublattice (identical charge,
    so swaps leave the Ewald energy invariant -- useful for reproducibility
    and mechanics tests)."""
    lattice = Lattice.cubic(5.64)
    species = ["Na+", "Na+", "K+", "K+", "Cl-", "Cl-", "Cl-", "Cl-"]
    structure = Structure(lattice, species, ROCK_SALT_COORDS)
    structure.add_oxidation_state_by_element({"Na": 1, "K": 1, "Cl": -1})
    return structure


@pytest.fixture
def na_ca_cl() -> Structure:
    """Rock salt with a mixed Na+/Ca2+ cation sublattice (different
    charges, so swaps change the Ewald energy -- useful for testing that
    Metropolis acceptance actually discriminates)."""
    lattice = Lattice.cubic(5.64)
    species = ["Na+", "Na+", "Ca2+", "Ca2+", "Cl-", "Cl-", "Cl-", "Cl-"]
    structure = Structure(lattice, species, ROCK_SALT_COORDS)
    structure.add_oxidation_state_by_element({"Na": 1, "Ca": 2, "Cl": -1})
    return structure


# --- move_ion / swap_ion -----------------------------------------------


def test_move_ion_relocates_species_without_mutating_input(na_k_cl):
    moved = move_ion(na_k_cl, 0, [0.25, 0.25, 0.25])
    assert moved[0].species == na_k_cl[0].species
    assert list(moved[0].frac_coords) == pytest.approx([0.25, 0.25, 0.25])
    # original untouched
    assert list(na_k_cl[0].frac_coords) == pytest.approx([0, 0, 0])


def test_move_ion_keeps_site_label(na_k_cl, tmp_path):
    # Regression: the moved site used to lose its label (pymatgen relabels it
    # "Na+"), so after several hops CIF output warned about duplicate labels.
    labeled = na_k_cl.copy()
    for i, site in enumerate(labeled):
        site.label = f"{site.specie.symbol}{i}"
    moved = move_ion(labeled, 0, [0.25, 0.25, 0.25])
    moved = move_ion(moved, 1, [0.75, 0.75, 0.75])
    assert moved.labels == labeled.labels
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        moved.to(filename=str(tmp_path / "moved.cif"))


def test_move_ion_rejects_occupied_target(na_k_cl):
    with pytest.raises(ValueError):
        move_ion(na_k_cl, 0, na_k_cl[1].frac_coords)


def test_move_ion_rejects_bad_index(na_k_cl):
    with pytest.raises(IndexError):
        move_ion(na_k_cl, 100, [0.25, 0.25, 0.25])


def test_swap_ion_exchanges_species_without_mutating_input(na_k_cl):
    swapped = swap_ion(na_k_cl, 0, 2)
    assert swapped[0].species == na_k_cl[2].species
    assert swapped[2].species == na_k_cl[0].species
    assert na_k_cl[0].species_string == "Na+"
    assert na_k_cl[2].species_string == "K+"


def test_swap_ion_rejects_same_index(na_k_cl):
    with pytest.raises(ValueError):
        swap_ion(na_k_cl, 0, 0)


def test_swap_ion_rejects_bad_index(na_k_cl):
    with pytest.raises(IndexError):
        swap_ion(na_k_cl, 0, 100)


# --- propose_next_configuration ---------------------------------------------------


def test_propose_next_configuration_returns_energy(na_k_cl):
    rng = np.random.default_rng(0)
    candidate, move, props = propose_next_configuration(na_k_cl, default_policy(), rng)
    assert isinstance(candidate, Structure)
    assert move.move_type == "swap"
    assert "energy" in props


def test_propose_next_configuration_raises_when_no_move_allowed(na_k_cl):
    policy = MonteCarloPolicy(
        choose_potential_move=lambda config, rng, ctx: MoveProposal("swap", 0, 2),
        is_move_allowed=lambda *a, **k: False,
    )
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError):
        propose_next_configuration(na_k_cl, policy, rng, max_attempts=5)


def test_propose_next_configuration_raises_on_unknown_move_type(na_k_cl):
    policy = MonteCarloPolicy(
        choose_potential_move=lambda config, rng, ctx: MoveProposal("teleport", 0),
    )
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="move_type"):
        propose_next_configuration(na_k_cl, policy, rng)


def test_propose_next_configuration_raises_when_energy_missing(na_k_cl):
    policy = MonteCarloPolicy(
        choose_potential_move=lambda config, rng, ctx: MoveProposal("swap", 0, 2),
        evaluate_props=lambda config, ctx: {"not_energy": 0.0},
    )
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="energy"):
        propose_next_configuration(na_k_cl, policy, rng)


# --- Metropolis criterion --------------------------------------------------


def test_metropolis_always_accepts_lower_energy():
    rng = np.random.default_rng(0)
    assert _metropolis_accept(current_energy=0.0, candidate_energy=-1.0, temperature=1000, rng=rng)


def test_metropolis_zero_temperature_rejects_higher_energy():
    rng = np.random.default_rng(0)
    assert not _metropolis_accept(current_energy=0.0, candidate_energy=1.0, temperature=0.0, rng=rng)


def test_metropolis_probabilistic_uses_boltzmann_factor():
    delta = 0.01  # eV
    temperature = 300.0
    p_accept = np.exp(-delta / (KB_EV_PER_K * temperature))

    class _FixedRng:
        def random(self):
            return p_accept / 2  # below threshold -> accept

    assert _metropolis_accept(0.0, delta, temperature, _FixedRng())

    class _FixedRngReject:
        def random(self):
            return min(p_accept * 2, 0.999999)  # above threshold -> reject

    assert not _metropolis_accept(0.0, delta, temperature, _FixedRngReject())


def test_metropolis_hastings_ratio_scales_acceptance():
    delta = 0.01  # eV
    temperature = 300.0
    log_ratio = np.log(0.5)
    p_accept = 0.5 * np.exp(-delta / (KB_EV_PER_K * temperature))

    class _FixedRng:
        def __init__(self, value):
            self.value = value

        def random(self):
            return self.value

    assert _metropolis_accept(0.0, delta, temperature, _FixedRng(p_accept * 0.9), log_ratio)
    assert not _metropolis_accept(0.0, delta, temperature, _FixedRng(p_accept * 1.1), log_ratio)
    # A large enough ratio makes an uphill move certain.
    assert _metropolis_accept(0.0, delta, temperature, _FixedRng(0.999), log_proposal_ratio=10.0)
    # A small enough ratio can reject a downhill move.
    assert not _metropolis_accept(0.0, -delta, temperature, _FixedRng(0.5), log_proposal_ratio=-10.0)


def test_metropolis_zero_temperature_ignores_proposal_ratio():
    rng = np.random.default_rng(0)
    assert not _metropolis_accept(0.0, 1.0, 0.0, rng, log_proposal_ratio=100.0)
    assert _metropolis_accept(0.0, -1.0, 0.0, rng, log_proposal_ratio=-100.0)


# --- run_monte_carlo --------------------------------------------------------


def test_run_monte_carlo_records_all_steps(na_k_cl):
    trajectory = run_monte_carlo(
        na_k_cl, default_policy(), temperature=1000, num_steps=20, seed=1
    )
    assert len(trajectory.steps) == 21  # initial + 20
    assert trajectory.steps[0].move is None
    assert all(s.move is not None for s in trajectory.steps[1:])


def test_run_monte_carlo_is_reproducible_with_fixed_seed(na_ca_cl):
    kwargs = dict(temperature=1500, num_steps=50, seed=7)
    traj1 = run_monte_carlo(na_ca_cl, default_policy(), **kwargs)
    traj2 = run_monte_carlo(na_ca_cl, default_policy(), **kwargs)
    assert traj1.energies == traj2.energies
    assert traj1.accepted == traj2.accepted


def test_run_monte_carlo_different_seeds_can_differ(na_ca_cl):
    traj1 = run_monte_carlo(na_ca_cl, default_policy(), temperature=1500, num_steps=50, seed=1)
    traj2 = run_monte_carlo(na_ca_cl, default_policy(), temperature=1500, num_steps=50, seed=2)
    assert traj1.accepted != traj2.accepted or traj1.energies != traj2.energies


def test_run_monte_carlo_does_not_mutate_input(na_k_cl):
    original = [str(site.species) for site in na_k_cl]
    run_monte_carlo(na_k_cl, default_policy(), temperature=1000, num_steps=20, seed=0)
    assert [str(site.species) for site in na_k_cl] == original


def test_run_monte_carlo_respects_num_structures_to_save(na_ca_cl):
    trajectory = run_monte_carlo(
        na_ca_cl, default_policy(), temperature=1500, num_steps=30, seed=3, num_structures_to_save=3
    )
    assert len(trajectory.saved_configurations) <= 3
    energies = [c.energy for c in trajectory.lowest_energy_saved()]
    assert energies == sorted(energies)


def test_run_monte_carlo_record_structure_every(na_k_cl):
    trajectory = run_monte_carlo(
        na_k_cl,
        default_policy(),
        temperature=1000,
        num_steps=10,
        seed=0,
        record_structure_every=5,
    )
    # Step 0 (the initial configuration) is always recorded when structure
    # recording is enabled at all, in addition to every Nth step.
    recorded_steps = {s.step for s in trajectory.steps if s.structure is not None}
    assert recorded_steps == {0, 5, 10}


def test_run_monte_carlo_time_limit_stops_early(na_k_cl):
    trajectory = run_monte_carlo(
        na_k_cl, default_policy(), temperature=1000, num_steps=100_000, seed=0, time_limit=0.0
    )
    assert len(trajectory.steps) < 100_001


def test_run_monte_carlo_records_seed_and_parameters(na_k_cl):
    trajectory = run_monte_carlo(
        na_k_cl, default_policy(), temperature=1000, num_steps=5, seed=123
    )
    assert trajectory.random_seed == 123
    assert trajectory.parameters["temperature"] == 1000
    assert trajectory.parameters["num_steps"] == 5


# --- checkpoints / resume_monte_carlo ------------------------------------


def _step_summary(trajectory):
    return [(s.step, s.energy, s.accepted, s.move) for s in trajectory.steps if s.move is not None]


def test_on_step_called_after_every_step_with_matching_checkpoint(na_ca_cl):
    calls = []
    trajectory = run_monte_carlo(
        na_ca_cl, default_policy(), temperature=1500, num_steps=10, seed=3, record_structure_every=1,
        on_step=lambda step, checkpoint: calls.append((step, checkpoint)),
    )
    assert [step.step for step, _ in calls] == list(range(1, 11))
    for step, checkpoint in calls:
        assert checkpoint.step == step.step
        assert checkpoint.energy == step.energy
        assert checkpoint.random_seed == 3
        # The checkpoint holds the current (post-decision) configuration.
        assert checkpoint.configuration == step.structure
    assert calls[-1][1].saved_configurations == trajectory.saved_configurations


@pytest.fixture
def cation_swap_supercell(na_ca_cl):
    """2x1x1 Na/Ca rock salt with cation-only swaps; at 20000 K the chain
    both accepts and rejects moves and visits several energies."""
    structure = na_ca_cl.copy()
    structure.make_supercell([2, 1, 1])
    cations = [i for i, site in enumerate(structure) if site.specie.symbol != "Cl"]
    return structure, lambda: sublattice_swap_policy(structure, sublattices=[cations])


@pytest.mark.parametrize("interrupt_at", [1, 7, 19])
def test_resume_reproduces_uninterrupted_run(tmp_path, cation_swap_supercell, interrupt_at):
    structure, make_policy = cation_swap_supercell
    kwargs = dict(temperature=20000, num_steps=30, num_structures_to_save=3)
    full = run_monte_carlo(structure, make_policy(), seed=11, **kwargs)
    # Guard against the comparison becoming trivial.
    assert 0 < sum(s.accepted for s in full.steps[1:]) < 30
    assert len({s.energy for s in full.steps}) > 1

    path = tmp_path / "checkpoint.json"

    def checkpoint_and_stop(step, checkpoint):
        save_checkpoint(checkpoint, path)
        if step.step == interrupt_at:
            raise KeyboardInterrupt  # simulate the job being killed

    with pytest.raises(KeyboardInterrupt):
        run_monte_carlo(structure, make_policy(), seed=11, on_step=checkpoint_and_stop, **kwargs)

    checkpoint = load_checkpoint(path)
    assert checkpoint.step == interrupt_at
    resumed = resume_monte_carlo(checkpoint, make_policy(), **kwargs)

    assert resumed.steps[0].step == interrupt_at and resumed.steps[0].move is None
    assert _step_summary(resumed) == _step_summary(full)[interrupt_at:]
    assert [(c.step, c.energy) for c in resumed.saved_configurations] == [
        (c.step, c.energy) for c in full.saved_configurations
    ]
    assert resumed.random_seed == 11
    assert resumed.parameters["resumed_from_step"] == interrupt_at


def test_resume_does_not_reevaluate_checkpointed_configuration(na_ca_cl):
    checkpoints = []
    run_monte_carlo(
        na_ca_cl, default_policy(), temperature=1500, num_steps=5, seed=0,
        on_step=lambda step, checkpoint: checkpoints.append(checkpoint),
    )
    policy = default_policy()
    evaluate = policy.evaluate_props
    calls = []
    policy.evaluate_props = lambda structure, context: calls.append(1) or evaluate(structure, context)
    resume_monte_carlo(checkpoints[-1], policy, temperature=1500, num_steps=8)
    assert len(calls) == 3  # one per new step, none for the starting configuration


def test_resume_at_or_past_num_steps_runs_nothing(na_ca_cl):
    checkpoints = []
    run_monte_carlo(
        na_ca_cl, default_policy(), temperature=1500, num_steps=5, seed=0,
        on_step=lambda step, checkpoint: checkpoints.append(checkpoint),
    )
    resumed = resume_monte_carlo(checkpoints[-1], default_policy(), temperature=1500, num_steps=5)
    assert [s.step for s in resumed.steps] == [5]


def test_resume_does_not_mutate_checkpoint(na_ca_cl):
    checkpoints = []
    run_monte_carlo(
        na_ca_cl, default_policy(), temperature=1500, num_steps=3, seed=0,
        on_step=lambda step, checkpoint: checkpoints.append(checkpoint),
    )
    checkpoint = checkpoints[-1]
    before = checkpoint.as_dict()
    resume_monte_carlo(checkpoint, default_policy(), temperature=1500, num_steps=10)
    assert checkpoint.as_dict() == before


def test_resume_rejects_checkpoint_without_energy(na_ca_cl):
    checkpoints = []
    run_monte_carlo(
        na_ca_cl, default_policy(), temperature=1500, num_steps=1, seed=0,
        on_step=lambda step, checkpoint: checkpoints.append(checkpoint),
    )
    checkpoints[-1].properties.pop("energy")
    with pytest.raises(ValueError, match="energy"):
        resume_monte_carlo(checkpoints[-1], default_policy(), temperature=1500, num_steps=5)


def test_resume_rejects_unknown_bit_generator(na_ca_cl):
    checkpoints = []
    run_monte_carlo(
        na_ca_cl, default_policy(), temperature=1500, num_steps=1, seed=0,
        on_step=lambda step, checkpoint: checkpoints.append(checkpoint),
    )
    checkpoints[-1].rng_state["bit_generator"] = "Generator"
    with pytest.raises(ValueError, match="bit generator"):
        resume_monte_carlo(checkpoints[-1], default_policy(), temperature=1500, num_steps=5)


# --- detailed balance -------------------------------------------------------


def _undeclared(policy, detailed_balance=None):
    return MonteCarloPolicy(
        choose_potential_move=policy.choose_potential_move,
        is_move_allowed=policy.is_move_allowed,
        evaluate_props=policy.evaluate_props,
        context=policy.context,
        detailed_balance=detailed_balance,
    )


def test_builtin_policies_declare_detailed_balance(na_k_cl):
    assert default_policy().detailed_balance is True
    with warnings.catch_warnings():
        warnings.simplefilter("error", DetailedBalanceWarning)
        run_monte_carlo(na_k_cl, default_policy(), temperature=1000, num_steps=3, seed=0)


def test_undeclared_policy_warns(na_k_cl):
    with pytest.warns(DetailedBalanceWarning, match="does not declare"):
        run_monte_carlo(na_k_cl, _undeclared(default_policy()), temperature=1000, num_steps=3, seed=0)


def test_policy_declaring_no_detailed_balance_warns_but_runs(na_k_cl):
    with pytest.warns(DetailedBalanceWarning, match="detailed_balance=False"):
        trajectory = run_monte_carlo(
            na_k_cl, _undeclared(default_policy(), False), temperature=1000, num_steps=3, seed=0
        )
    assert len(trajectory.steps) == 4
    assert trajectory.parameters["detailed_balance"] is False


def test_zero_temperature_does_not_warn(na_k_cl):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DetailedBalanceWarning)
        run_monte_carlo(na_k_cl, _undeclared(default_policy(), False), temperature=0, num_steps=3, seed=0)


def test_resume_also_warns(na_ca_cl):
    checkpoints = []
    run_monte_carlo(na_ca_cl, default_policy(), temperature=1500, num_steps=2, seed=0,
                    on_step=lambda step, checkpoint: checkpoints.append(checkpoint))
    with pytest.warns(DetailedBalanceWarning):
        resume_monte_carlo(checkpoints[-1], _undeclared(default_policy()), temperature=1500, num_steps=4)


def _alternately_disallowed_policy():
    calls = {"n": 0}

    def allowed(configuration, candidate, move, context):
        calls["n"] += 1
        return calls["n"] % 2 == 0

    return MonteCarloPolicy(
        choose_potential_move=lambda configuration, rng, context: MoveProposal("swap", 0, 2),
        is_move_allowed=allowed,
        detailed_balance=True,
    )


def test_disallowed_moves_are_rejected_steps_by_default(na_k_cl):
    trajectory = run_monte_carlo(na_k_cl, _alternately_disallowed_policy(), temperature=1000, num_steps=6, seed=0)
    disallowed = [s for s in trajectory.steps[1:] if not s.allowed]
    assert [s.step for s in disallowed] == [1, 3, 5]
    assert all(not s.accepted and s.properties == {} and s.move["move_type"] == "swap" for s in disallowed)
    assert all(s.allowed and s.accepted for s in trajectory.steps[1:] if s.step % 2 == 0)
    assert trajectory.acceptance_rate() == pytest.approx(0.5)
    assert trajectory.parameters["disallowed_moves"] == "reject"
    assert all(c.step % 2 == 0 for c in trajectory.saved_configurations if c.step > 0)


def test_no_proposal_is_a_rejected_step(na_k_cl):
    policy = MonteCarloPolicy(choose_potential_move=lambda c, rng, ctx: None, detailed_balance=True)
    trajectory = run_monte_carlo(na_k_cl, policy, temperature=1000, num_steps=3, seed=0)
    assert [(s.move, s.allowed, s.accepted) for s in trajectory.steps[1:]] == [(None, False, False)] * 3
    assert trajectory.acceptance_rate() == 0.0


def test_retry_mode_redraws_and_warns_once(na_k_cl):
    with pytest.warns(DetailedBalanceWarning, match="redrawn") as record:
        trajectory = run_monte_carlo(
            na_k_cl, _alternately_disallowed_policy(), temperature=1000, num_steps=4, seed=0, disallowed_moves="retry"
        )
    assert len([w for w in record if "redrawn" in str(w.message)]) == 1
    assert all(s.allowed for s in trajectory.steps)


def test_retry_mode_raises_when_no_move_allowed(na_k_cl):
    policy = MonteCarloPolicy(
        choose_potential_move=lambda c, rng, ctx: MoveProposal("swap", 0, 2),
        is_move_allowed=lambda *a: False,
        detailed_balance=True,
    )
    with pytest.raises(RuntimeError), warnings.catch_warnings():
        warnings.simplefilter("ignore", DetailedBalanceWarning)
        run_monte_carlo(na_k_cl, policy, temperature=1000, num_steps=3, seed=0,
                        disallowed_moves="retry", max_move_attempts=4)


def test_rejects_unknown_disallowed_moves_mode(na_k_cl):
    with pytest.raises(ValueError, match="disallowed_moves"):
        run_monte_carlo(na_k_cl, default_policy(), temperature=1000, num_steps=1, disallowed_moves="skip")


def test_step_allowed_flag_round_trips_and_defaults_to_true():
    step = MonteCarloStep(step=1, energy=0.0, accepted=False, temperature=300.0, allowed=False)
    assert MonteCarloStep.from_dict(step.as_dict()).allowed is False
    legacy = step.as_dict()
    del legacy["allowed"]
    assert MonteCarloStep.from_dict(legacy).allowed is True
    trajectory = MonteCarloTrajectory(steps=[step])
    assert MonteCarloTrajectory.from_dict(trajectory.as_dict()).steps[0].allowed is False
