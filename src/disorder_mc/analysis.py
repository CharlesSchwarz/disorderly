"""Analysis of Monte Carlo trajectories: per-step property series, rolling
acceptance rates, trajectory plots, and thermal averages.

A note on which configuration a value belongs to: ``MonteCarloStep.properties``
holds the properties of the *proposed* configuration at that step, while
``MonteCarloStep.energy`` is the energy of the *current* configuration after
the accept/reject decision. The functions here report properties of the
current configuration (the state of the Markov chain), obtained by carrying
each property forward from the most recent accepted step. That is the series
a Metropolis thermal average must be taken over: a rejected step counts the
current configuration again.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from disorder_mc.results import MonteCarloTrajectory
from disorder_mc.settings import DetailedBalanceWarning

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def property_series(trajectory: MonteCarloTrajectory, prop: str = "energy") -> np.ndarray:
    """Value of ``prop`` for the *current* configuration at every step.

    Args:
        trajectory: A Monte Carlo trajectory.
        prop: A key of the properties returned by the policy's
            ``evaluate_props`` (e.g. ``"energy"`` or a user-defined
            quantity). ``"energy"`` is read from ``MonteCarloStep.energy``.

    Returns:
        A float array aligned with ``trajectory.steps``. On a rejected or
        disallowed step the value is that of the last accepted configuration.

    Raises:
        ValueError: If the trajectory is empty, or ``prop`` is missing from
            its first step or from an accepted step.
    """
    if not trajectory.steps:
        raise ValueError("The trajectory has no steps.")
    if prop == "energy":
        return np.array(trajectory.energies, dtype=float)

    values = np.empty(len(trajectory.steps))
    current = None
    for i, step in enumerate(trajectory.steps):
        if i == 0 or step.accepted:
            if prop not in step.properties:
                available = sorted(step.properties)
                raise ValueError(
                    f"Property {prop!r} is not recorded at step {step.step}; available properties: {available}."
                )
            current = step.properties[prop]
        values[i] = current
    return values


def rolling_acceptance_rate(trajectory: MonteCarloTrajectory, window: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of proposed moves accepted over a trailing window.

    Only proposed moves count: the initial step (and the first step of a
    resumed trajectory, which has no move) is excluded, while disallowed
    proposals count as rejected, as in
    :meth:`MonteCarloTrajectory.acceptance_rate`.

    Args:
        trajectory: A Monte Carlo trajectory.
        window: Number of consecutive moves averaged over.

    Returns:
        ``(steps, rates)``: the step number at the end of each full window and
        the acceptance rate over that window. Both are empty if the trajectory
        has fewer than ``window`` moves.
    """
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}.")
    moves = [s for s in trajectory.steps if s.move is not None or not s.allowed]
    if len(moves) < window:
        return np.array([], dtype=int), np.array([], dtype=float)
    accepted = np.array([s.accepted for s in moves], dtype=float)
    rates = np.convolve(accepted, np.ones(window) / window, mode="valid")
    steps = np.array([s.step for s in moves[window - 1 :]], dtype=int)
    return steps, rates


def plot_trajectory(
    trajectory: MonteCarloTrajectory,
    properties: Sequence[str] = (),
    *,
    acceptance_rate: bool = True,
    window: int = 100,
    equilibration_steps: int | None = None,
    height_ratios: Sequence[float] | None = None,
    figsize: tuple[float, float] | None = None,
) -> "Figure":
    """Plot energy, rolling acceptance rate, and other properties against step.

    One panel per quantity, sharing the step axis, from top to bottom: the
    energy of the current configuration, the acceptance rate over a trailing
    ``window`` of moves (see :func:`rolling_acceptance_rate`) if
    ``acceptance_rate`` is true, then each property in ``properties``, for the
    current configuration (see :func:`property_series`).

    Requires matplotlib (``pip install disorder-mc[plot]``).

    Args:
        trajectory: A Monte Carlo trajectory.
        properties: Names of further properties to plot, one panel each, in
            this order. Only the properties listed here are plotted.
        acceptance_rate: Whether to include the acceptance-rate panel.
        window: Number of moves in the acceptance-rate window.
        equilibration_steps: If given, a dotted vertical line marks this step
            on every panel.
        height_ratios: Relative panel heights, one per panel in the order
            above. ``None`` makes all panels the same height.
        figsize: Figure size; defaults to 7 inches wide and 2.2 inches per
            panel (scaled by ``height_ratios``).

    Returns:
        The matplotlib ``Figure``; its panels are ``figure.axes``.

    Raises:
        ValueError: If ``height_ratios`` does not have one entry per panel or
            has a non-positive entry, or a property is not recorded.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "plot_trajectory requires matplotlib; install it with `pip install disorder-mc[plot]`."
        ) from exc

    names = list(properties)
    n_panels = 1 + int(acceptance_rate) + len(names)
    if height_ratios is None:
        height_ratios = [1.0] * n_panels
    height_ratios = list(height_ratios)
    if len(height_ratios) != n_panels:
        raise ValueError(
            f"height_ratios has {len(height_ratios)} entries but there are {n_panels} panels "
            f"(energy{', acceptance rate' if acceptance_rate else ''}{''.join(', ' + n for n in names)})."
        )
    if any(r <= 0 for r in height_ratios):
        raise ValueError(f"height_ratios must all be positive, got {height_ratios}.")

    # Evaluate every series before creating the figure, so a missing property
    # raises without leaving an empty figure behind.
    steps = np.array([s.step for s in trajectory.steps])
    series = {name: property_series(trajectory, name) for name in ["energy", *names]}

    if figsize is None:
        figsize = (7, 2.2 * n_panels * sum(height_ratios) / n_panels)
    fig, axes = plt.subplots(
        n_panels, 1, sharex=True, figsize=figsize, squeeze=False, gridspec_kw={"height_ratios": height_ratios}
    )
    axes = list(axes[:, 0])
    panels = iter(axes)

    ax = next(panels)
    ax.plot(steps, series["energy"], linewidth=1)
    ax.set_ylabel("Energy (eV)")

    if acceptance_rate:
        ax = next(panels)
        rate_steps, rates = rolling_acceptance_rate(trajectory, window)
        ax.plot(rate_steps, rates, linewidth=1)
        ax.set_ylabel(f"Acceptance rate\n(last {window} moves)")

    for name in names:
        ax = next(panels)
        ax.plot(steps, series[name], linewidth=1, drawstyle="steps-post")
        ax.set_ylabel(name)

    for ax in axes:
        ax.grid(alpha=0.3)
        if equilibration_steps is not None:
            ax.axvline(equilibration_steps, color="0.4", linewidth=1, linestyle=":")
    axes[-1].set_xlabel("Monte Carlo step")
    temperatures = {s.temperature for s in trajectory.steps}
    if len(temperatures) == 1:
        axes[0].set_title(f"T = {temperatures.pop():g} K")
    fig.tight_layout()
    return fig


@dataclass(frozen=True)
class ThermalAverage:
    """A Monte Carlo estimate of a thermal (canonical-ensemble) average.

    Attributes:
        prop: The averaged property.
        temperature: Temperature of the averaged steps (K).
        mean: Average of the property over the averaged steps.
        std: Standard deviation of the property over those steps (the
            thermal fluctuation, not the uncertainty of ``mean``).
        standard_error: Batch-means estimate of the statistical uncertainty
            of ``mean``; ``nan`` if there are fewer than two blocks.
        n_samples: Number of steps averaged.
        n_blocks: Number of blocks used for ``standard_error``.
    """

    prop: str
    temperature: float
    mean: float
    std: float
    standard_error: float
    n_samples: int
    n_blocks: int

    def __str__(self) -> str:
        return (
            f"<{self.prop}> at {self.temperature:g} K = {self.mean:.6g} +/- {self.standard_error:.2g} "
            f"(std {self.std:.3g}, {self.n_samples} steps)"
        )


def thermal_average(
    trajectory: MonteCarloTrajectory,
    prop: str = "energy",
    *,
    equilibration_steps: int = 0,
    n_blocks: int = 20,
) -> ThermalAverage:
    """Thermal average of a property over the equilibrated part of a run.

    A Metropolis chain that satisfies detailed balance visits configurations
    with Boltzmann probability, so the thermal average is the plain mean of
    the property over the chain's *current* configuration at every step after
    equilibration, with rejected steps counting the current configuration
    again (see :func:`property_series`). No Boltzmann reweighting is applied.

    Successive steps are correlated, so the uncertainty of the mean is
    estimated by batch means: the averaged steps are split into ``n_blocks``
    contiguous blocks, and ``standard_error`` is the standard deviation of the
    block means divided by ``sqrt(n_blocks)``. This is reliable only when each
    block is much longer than the autocorrelation time; if it is not, the
    error is underestimated, and a longer run (or fewer blocks) is needed.

    Args:
        trajectory: A Monte Carlo trajectory run at a single temperature.
        prop: Property to average (see :func:`property_series`).
        equilibration_steps: Steps numbered below this are discarded.
        n_blocks: Number of blocks for the error estimate. Leftover steps
            that do not fill a block are left out of the error estimate
            (but not out of ``mean``).

    Returns:
        A :class:`ThermalAverage`.

    Raises:
        ValueError: If no steps remain after equilibration, the remaining
            steps span more than one temperature, or ``n_blocks < 1``.

    Warns:
        DetailedBalanceWarning: If the trajectory's parameters do not record
            ``detailed_balance=True`` for a run above 0 K, in which case the
            chain need not sample the Boltzmann distribution.
    """
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be at least 1, got {n_blocks}.")
    values = property_series(trajectory, prop)
    keep = np.array([s.step >= equilibration_steps for s in trajectory.steps])
    values = values[keep]
    if values.size == 0:
        raise ValueError(f"No steps remain after discarding steps below {equilibration_steps}.")

    temperatures = {s.temperature for s, k in zip(trajectory.steps, keep) if k}
    if len(temperatures) > 1:
        raise ValueError(
            f"The averaged steps span several temperatures {sorted(temperatures)}; "
            "a thermal average needs a single temperature."
        )
    temperature = temperatures.pop()
    if temperature > 0 and trajectory.parameters.get("detailed_balance") is not True:
        warnings.warn(
            "This trajectory does not record detailed_balance=True, so its steps may not be "
            "Boltzmann-distributed and the plain mean may be a biased thermal average.",
            DetailedBalanceWarning,
            stacklevel=2,
        )

    block_size = values.size // n_blocks
    if n_blocks > 1 and block_size > 0:
        block_means = values[: block_size * n_blocks].reshape(n_blocks, block_size).mean(axis=1)
        standard_error = float(block_means.std(ddof=1) / math.sqrt(n_blocks))
    else:
        standard_error = math.nan

    return ThermalAverage(
        prop=prop,
        temperature=float(temperature),
        mean=float(values.mean()),
        std=float(values.std()),
        standard_error=standard_error,
        n_samples=int(values.size),
        n_blocks=n_blocks,
    )
