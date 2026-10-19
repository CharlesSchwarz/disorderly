import math
import warnings

import numpy as np
import pytest

from disorder_mc.analysis import (
    plot_trajectory,
    property_series,
    rolling_acceptance_rate,
    thermal_average,
)
from disorder_mc.results import MonteCarloStep, MonteCarloTrajectory
from disorder_mc.settings import DetailedBalanceWarning


def _trajectory(records, temperature=1000.0, detailed_balance=True) -> MonteCarloTrajectory:
    """Build a trajectory from ``(accepted, proposed_energy, proposed_n)`` records.

    The first record is the initial configuration (no move); ``None`` for the
    proposed energy marks a disallowed step with no evaluated properties.
    """
    trajectory = MonteCarloTrajectory(parameters={"detailed_balance": detailed_balance})
    current = None
    for i, (accepted, energy, n) in enumerate(records):
        if energy is None:
            props, allowed = {}, False
        else:
            props, allowed = {"energy": energy, "copy_energy": energy, "n": n, "label": "x"}, True
        if accepted:
            current = energy
        trajectory.append(
            MonteCarloStep(
                step=i,
                energy=current,
                accepted=accepted,
                temperature=temperature,
                move=None if i == 0 else {"move_type": "swap"},
                properties=props,
                allowed=allowed,
            )
        )
    return trajectory


@pytest.fixture
def simple() -> MonteCarloTrajectory:
    return _trajectory(
        [
            (True, 0.0, 10),
            (False, 5.0, 11),  # rejected: stays at n=10
            (True, -1.0, 12),
            (False, None, None),  # disallowed: stays at n=12
            (True, -2.0, 9),
        ]
    )


def test_property_series_carries_forward_current_configuration(simple):
    np.testing.assert_array_equal(property_series(simple, "n"), [10, 10, 12, 12, 9])
    np.testing.assert_array_equal(property_series(simple, "energy"), [0.0, 0.0, -1.0, -1.0, -2.0])


def test_property_series_missing_property(simple):
    with pytest.raises(ValueError, match="not recorded"):
        property_series(simple, "missing")


def test_property_series_empty_trajectory():
    with pytest.raises(ValueError, match="no steps"):
        property_series(MonteCarloTrajectory(), "energy")


def test_rolling_acceptance_rate_excludes_initial_step_and_counts_disallowed(simple):
    steps, rates = rolling_acceptance_rate(simple, window=2)
    # Moves are steps 1-4 with accepted = F, T, F, T.
    np.testing.assert_array_equal(steps, [2, 3, 4])
    np.testing.assert_allclose(rates, [0.5, 0.5, 0.5])
    steps, rates = rolling_acceptance_rate(simple, window=4)
    np.testing.assert_allclose(rates, [simple.acceptance_rate()])


def test_rolling_acceptance_rate_short_trajectory_and_invalid_window(simple):
    steps, rates = rolling_acceptance_rate(simple, window=10)
    assert steps.size == 0 and rates.size == 0
    with pytest.raises(ValueError):
        rolling_acceptance_rate(simple, window=0)


def test_thermal_average_uses_current_configuration(simple):
    result = thermal_average(simple, "n", equilibration_steps=2, n_blocks=1)
    # Steps 2-4 are at n = 12, 12, 9 (not the proposed 12, -, 9).
    assert result.mean == pytest.approx(11.0)
    assert result.std == pytest.approx(np.std([12, 12, 9]))
    assert result.n_samples == 3
    assert result.temperature == 1000.0
    assert math.isnan(result.standard_error)


def test_thermal_average_batch_means_standard_error():
    # Alternate accepted moves between energies 0 and 1 over 40 moves.
    records = [(True, 0.0, 0)] + [(True, float(i % 2), 0) for i in range(1, 41)]
    trajectory = _trajectory(records)
    result = thermal_average(trajectory, n_blocks=4)
    values = np.array([i % 2 for i in range(41)], dtype=float)
    block_means = values[:40].reshape(4, 10).mean(axis=1)
    assert result.mean == pytest.approx(values.mean())
    assert result.standard_error == pytest.approx(block_means.std(ddof=1) / 2)


def test_thermal_average_invalid_inputs(simple):
    with pytest.raises(ValueError, match="No steps remain"):
        thermal_average(simple, equilibration_steps=100)
    with pytest.raises(ValueError):
        thermal_average(simple, n_blocks=0)
    simple.steps[-1].temperature = 500.0
    with pytest.raises(ValueError, match="several temperatures"):
        thermal_average(simple)
    # Discarding the other temperature makes it valid again.
    assert thermal_average(simple, equilibration_steps=4).temperature == 500.0


def test_thermal_average_warns_without_detailed_balance():
    records = [(True, 0.0, 0), (True, 1.0, 0)]
    with pytest.warns(DetailedBalanceWarning):
        thermal_average(_trajectory(records, detailed_balance=None))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        thermal_average(_trajectory(records, detailed_balance=None, temperature=0.0))
        thermal_average(_trajectory(records, detailed_balance=True))


@pytest.fixture
def plt():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


def _labels(fig):
    return [ax.get_ylabel() for ax in fig.axes]


def test_plot_trajectory_default_panels(simple, plt):
    # Other properties are plotted only when requested.
    fig = plot_trajectory(simple, window=2, equilibration_steps=2)
    assert _labels(fig) == ["Energy (eV)", "Acceptance rate\n(last 2 moves)"]


def test_plot_trajectory_requested_properties(simple, plt):
    fig = plot_trajectory(simple, ["n", "copy_energy"], window=2)
    assert _labels(fig) == ["Energy (eV)", "Acceptance rate\n(last 2 moves)", "n", "copy_energy"]
    np.testing.assert_array_equal(fig.axes[2].lines[0].get_ydata(), [10, 10, 12, 12, 9])


def test_plot_trajectory_without_acceptance_rate(simple, plt):
    fig = plot_trajectory(simple, ["n"], acceptance_rate=False)
    assert _labels(fig) == ["Energy (eV)", "n"]
    fig = plot_trajectory(simple, acceptance_rate=False)
    assert _labels(fig) == ["Energy (eV)"]


def test_plot_trajectory_height_ratios(simple, plt):
    fig = plot_trajectory(simple, ["n"], window=2, height_ratios=[3, 1, 1])
    heights = [ax.get_position().height for ax in fig.axes]
    assert heights[0] == pytest.approx(3 * heights[1], rel=0.05)
    assert heights[1] == pytest.approx(heights[2], rel=0.05)


def test_plot_trajectory_invalid_inputs(simple, plt):
    with pytest.raises(ValueError, match="2 panels"):
        plot_trajectory(simple, height_ratios=[1, 1, 1])
    with pytest.raises(ValueError, match="positive"):
        plot_trajectory(simple, height_ratios=[1, 0])
    n_figures = len(plt.get_fignums())
    with pytest.raises(ValueError, match="not recorded"):
        plot_trajectory(simple, ["missing"])
    assert len(plt.get_fignums()) == n_figures


def test_plot_trajectory_from_real_run(plt):
    from pymatgen.core import Lattice, Structure

    from disorder_mc import default_policy, run_monte_carlo

    structure = Structure(
        Lattice.cubic(5.64),
        ["Na+", "Na+", "Ca2+", "Ca2+", "Cl-", "Cl-", "Cl-", "Cl-"],
        [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0],
         [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
    )
    trajectory = run_monte_carlo(structure, default_policy(), temperature=5000, num_steps=20, seed=0)
    fig = plot_trajectory(trajectory, window=5)
    assert len(fig.axes) == 2
    result = thermal_average(trajectory, equilibration_steps=5, n_blocks=3)
    assert result.n_samples == 16
    assert min(trajectory.energies) <= result.mean <= max(trajectory.energies)
