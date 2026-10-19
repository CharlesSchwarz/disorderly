"""Random configuration generation and energy-ranked sampling of
ordered configurations from a disordered parent structure.

May depend on :mod:`disorderly.structures`, :mod:`disorderly.symmetry`,
and :mod:`disorderly.energy`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from typing import Any, Callable

import numpy as np
from pymatgen.core import PeriodicSite, Structure

from disorderly._warning_filters import warning_filter_scope
from disorderly.energy import disordered_ewald_energy
from disorderly.ordering import ordering_key
from disorderly.results import SavedConfiguration
from disorderly.structures import disordered_site_groups
from disorderly.symmetry import symmetrically_equivalent

EnergyEvaluation = float | dict[str, Any]
EnergyEvaluator = Callable[[Structure], EnergyEvaluation]

__all__ = ["random_ordered_configuration", "random_energy_sampling", "random_ewald_sampling"]

# Shared fallback generator for ``random_ordered_configuration`` calls that
# omit ``rng``: seeded once at import, so successive calls give different
# configurations while a fresh process always reproduces the same sequence.
_DEFAULT_RNG = np.random.default_rng(0)


def random_ordered_configuration(
    parent: Structure, rng: np.random.Generator | None = None, occupancy_tol: float = 1e-2
) -> Structure:
    """Generate one random ordered configuration consistent with a
    disordered parent's site occupancies.

    Every already-ordered site of ``parent`` is copied unchanged. Disordered
    sites are grouped by their exact species/occupancy signature (e.g. all
    sites with occupancy ``{Li+: 0.5, Fe3+: 0.0625}``) -- sites sharing a
    signature are treated as one interchangeable pool, and species are
    assigned to random sites within that pool in numbers proportional to
    the occupancies, so that the produced configuration's overall
    composition matches ``parent``'s (up to rounding to whole sites). If a
    structure has multiple physically distinct disordered sublattices that
    happen to share identical species and occupancies, they are pooled
    together; call this function separately per sublattice (e.g. after
    splitting with :func:`disorderly.structures.replace_species_and_occupancy`) if
    that is not desired.

    If a pool's occupancies sum to (approximately) 1, every site in the pool
    is assigned a species -- the last species (by sorted label) absorbs any
    rounding remainder so site counts sum exactly. If they sum to less than
    1, site counts are rounded independently and the shortfall is left
    vacant (i.e. those sites are simply absent from the returned structure).
    The pools and counts are those reported by
    :func:`disorderly.structures.disordered_site_groups`.

    Args:
        parent: Disordered parent structure.
        rng: Random-number generator (for reproducibility, construct with
            ``numpy.random.default_rng(seed)``). Defaults to a module-level
            generator created once with ``numpy.random.default_rng(0)``
            and shared by all calls that omit ``rng``: repeated calls give
            different configurations, and the sequence is reproducible
            from a fresh interpreter, but depends on how many earlier
            default-``rng`` calls the process has made. Pass an explicit
            generator for reproducibility that is independent of call
            history.
        occupancy_tol: Absolute tolerance for treating a pool's total
            occupancy as exactly 1.

    Returns:
        A new ordered ``Structure``. May have fewer sites than ``parent``
        if any disordered pool is under-occupied (partial vacancy).

    Raises:
        ValueError: If rounding site counts to whole numbers produces a
            count inconsistent with the number of available sites in a
            pool (only possible in pathological rounding edge cases with
            several species per site).
    """
    if rng is None:
        rng = _DEFAULT_RNG
    ordered_sites = [site for site in parent if site.is_ordered]
    assigned_sites: list[PeriodicSite] = []
    for group in disordered_site_groups(parent, occupancy_tol=occupancy_tol, warn=False):
        shuffled = rng.permutation(group.site_indices)
        cursor = 0
        # Vacancies (key None) come last and are simply left out.
        for sp, n in group.counts.items():
            if sp is not None:
                for idx in shuffled[cursor : cursor + n]:
                    assigned_sites.append(PeriodicSite(sp, parent[idx].frac_coords, parent.lattice))
            cursor += n

    return Structure.from_sites(ordered_sites + assigned_sites, to_unit_cell=False)


def random_ewald_sampling(
    parent: Structure,
    n_attempts: int,
    n_return: int = 10,
    seed: int | np.random.Generator | None = None,
    occupancy_tol: float = 1e-2,
    candidate_pool_size: int | None = None,
    symmetry_match_kwargs: dict[str, Any] | None = None,
    time_limit: float | None = None,
    n_jobs: int = 1,
) -> list[SavedConfiguration]:
    """Randomly sample ordered configurations of a disordered parent and
    return the lowest-Ewald-energy symmetrically distinct ones.

    Generate up to ``n_attempts``
    random ordered configurations (:func:`random_ordered_configuration`),
    score them by :func:`disorderly.energy.disordered_ewald_energy`,
    de-duplicate symmetrically equivalent configurations
    (:func:`disorderly.symmetry.symmetrically_equivalent`), and return the
    ``n_return`` lowest-energy distinct configurations.

    To avoid retaining ``n_attempts`` configurations in memory
    simultaneously (which can be large), only a bounded pool of the
    lowest-energy ``candidate_pool_size`` attempts is kept at any time,
    updated incrementally as attempts are generated; de-duplication and
    final ranking are performed over that pool rather than over all
    attempts. If ``candidate_pool_size`` is too small relative to how many
    of the low-energy attempts turn out to be symmetric duplicates of each
    other, fewer than ``n_return`` distinct configurations may be returned
    even though more exist among the untried attempts -- increase
    ``candidate_pool_size`` (or ``n_attempts``) if that happens.

    Args:
        parent: Disordered parent structure to sample orderings of. Must
            carry oxidation-state information (required by
            ``disordered_ewald_energy``).
        n_attempts: Number of random configurations to generate and score.
        n_return: Maximum number of symmetrically distinct, lowest-energy
            configurations to return.
        seed: Random seed (or an existing ``numpy.random.Generator``) for
            reproducible sampling.
        occupancy_tol: Passed to :func:`random_ordered_configuration`.
        candidate_pool_size: Maximum number of (energy-sorted) candidates
            retained in memory at once. Defaults to
            ``max(5 * n_return, 50)``.
        symmetry_match_kwargs: Extra keyword arguments passed to
            :func:`disorderly.symmetry.symmetrically_equivalent` (e.g.
            ``ltol``, ``stol``) when de-duplicating.
        time_limit: Optional wall-clock time limit in seconds. Checked
            between sampling attempts; an in-progress energy evaluation is
            allowed to finish before the sampler stops.
        n_jobs: Number of worker threads used to evaluate candidate energies.
            ``1`` runs serially. Values greater than 1 evaluate batches of
            generated candidates concurrently.

    Returns:
        Up to ``n_return`` :class:`~disorderly.results.SavedConfiguration`
        objects, sorted by ascending energy, with ``step`` set to the
        attempt index at which that configuration was generated.
    """
    return random_energy_sampling(
        parent,
        energy_evaluator=disordered_ewald_energy,
        n_attempts=n_attempts,
        n_return=n_return,
        seed=seed,
        occupancy_tol=occupancy_tol,
        candidate_pool_size=candidate_pool_size,
        symmetry_match_kwargs=symmetry_match_kwargs,
        time_limit=time_limit,
        n_jobs=n_jobs,
    )


def _evaluate_energy(evaluator: EnergyEvaluator, candidate: Structure) -> tuple[float, dict[str, Any]]:
    evaluated = evaluator(candidate)
    if isinstance(evaluated, dict):
        if "energy" not in evaluated:
            raise ValueError(
                "energy_evaluator returned a dict without an 'energy' key; "
                f"got keys {sorted(evaluated)}."
            )
        return float(evaluated["energy"]), dict(evaluated)
    return float(evaluated), {}


def _evaluate_candidate(
    parent: Structure,
    evaluator: EnergyEvaluator,
    candidate: Structure,
    attempt: int,
    occupancy_tol: float,
) -> SavedConfiguration:
    with warning_filter_scope(suppress_future_warnings=True):
        energy, properties = _evaluate_energy(evaluator, candidate)
    return SavedConfiguration(
        step=attempt,
        energy=energy,
        structure=candidate,
        properties=properties,
        ordering_key=ordering_key(parent, candidate, occupancy_tol=occupancy_tol),
    )


def _add_to_pool(pool: list[SavedConfiguration], candidate: SavedConfiguration, pool_size: int) -> None:
    pool.append(candidate)
    pool.sort(key=lambda c: (c.energy, c.step))
    del pool[pool_size:]


def _prepare_energy_evaluator_worker(evaluator: EnergyEvaluator, barrier: threading.Barrier | None = None) -> None:
    if barrier is not None:
        barrier.wait()
    prepare_worker = getattr(evaluator, "prepare_worker", None)
    if callable(prepare_worker):
        prepare_worker()


def random_energy_sampling(
    parent: Structure,
    energy_evaluator: EnergyEvaluator,
    n_attempts: int,
    n_return: int = 10,
    seed: int | np.random.Generator | None = None,
    occupancy_tol: float = 1e-2,
    candidate_pool_size: int | None = None,
    symmetry_match_kwargs: dict[str, Any] | None = None,
    time_limit: float | None = None,
    n_jobs: int = 1,
    verbose: bool = True,
) -> list[SavedConfiguration]:
    """Randomly sample ordered configurations and rank them by a custom energy.

    Args:
        parent: Disordered parent structure to sample orderings of.
        energy_evaluator: Callable invoked as ``energy_evaluator(candidate)``.
            It may return either a scalar energy in eV, or a dict containing
            an ``"energy"`` key plus any additional properties to store.
        n_attempts: Number of random configurations to generate and score.
        n_return: Maximum number of symmetrically distinct, lowest-energy
            configurations to return.
        seed: Random seed (or an existing ``numpy.random.Generator``) for
            reproducible sampling.
        occupancy_tol: Passed to :func:`random_ordered_configuration` and
            :func:`disorderly.ordering.ordering_key`.
        candidate_pool_size: Maximum number of (energy-sorted) candidates
            retained in memory at once. Defaults to
            ``max(5 * n_return, 50)``.
        symmetry_match_kwargs: Extra keyword arguments passed to
            :func:`disorderly.symmetry.symmetrically_equivalent` when
            de-duplicating.
        time_limit: Optional wall-clock time limit in seconds. Checked
            between sampling attempts; an in-progress energy evaluation is
            allowed to finish before the sampler stops. If the limit is
            reached, the best structures retained so far are deduplicated
            and returned.
        n_jobs: Number of worker threads used to evaluate candidate energies.
            ``1`` runs serially. Values greater than 1 evaluate batches of
            generated candidates concurrently. The time limit is checked
            between batches, so in-progress energy evaluations are allowed
            to finish before sampling stops.

    Returns:
        Up to ``n_return`` saved configurations sorted by ascending energy.
    """
    if n_attempts <= 0:
        raise ValueError(f"n_attempts must be positive, got {n_attempts}.")
    if n_return <= 0:
        raise ValueError(f"n_return must be positive, got {n_return}.")
    if n_jobs <= 0:
        raise ValueError(f"n_jobs must be positive, got {n_jobs}.")

    rng = np.random.default_rng(seed)
    pool_size = candidate_pool_size or max(5 * n_return, 50)
    symmetry_match_kwargs = symmetry_match_kwargs or {}

    pool: list[SavedConfiguration] = []
    start_time = time.perf_counter()
    if n_jobs == 1:
        for attempt in range(n_attempts):
            if time_limit is not None and attempt > 0 and (time.perf_counter() - start_time) > time_limit:
                if verbose:
                    print(
                        f"Stopping after {attempt} attempts because time_limit={time_limit} s was exceeded.",
                        flush=True,
                    )
                break
            candidate = random_ordered_configuration(parent, rng, occupancy_tol=occupancy_tol)
            saved = _evaluate_candidate(parent, energy_evaluator, candidate, attempt, occupancy_tol)
            if verbose:
                print(f"Attempt {attempt:>7d}  energy {saved.energy:.6f}", flush=True)
            _add_to_pool(pool, saved, pool_size)
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            warmup_barrier = threading.Barrier(n_jobs)
            warmup_futures = [
                executor.submit(_prepare_energy_evaluator_worker, energy_evaluator, warmup_barrier)
                for _ in range(n_jobs)
            ]
            for future in warmup_futures:
                future.result()

            attempt = 0
            while attempt < n_attempts:
                if time_limit is not None and attempt > 0 and (time.perf_counter() - start_time) > time_limit:
                    if verbose:
                        print(
                            f"Stopping after {attempt} attempts because time_limit={time_limit} s was exceeded.",
                            flush=True,
                        )
                    break

                batch_size = min(n_jobs, n_attempts - attempt)
                batch = [
                    (
                        next_attempt,
                        random_ordered_configuration(parent, rng, occupancy_tol=occupancy_tol),
                    )
                    for next_attempt in range(attempt, attempt + batch_size)
                ]
                futures = [
                    executor.submit(
                        _evaluate_candidate,
                        parent,
                        energy_evaluator,
                        candidate,
                        next_attempt,
                        occupancy_tol,
                    )
                    for next_attempt, candidate in batch
                ]
                for future in as_completed(futures):
                    saved = future.result()
                    if verbose:
                        print(f"Attempt {saved.step:>7d}  energy {saved.energy:.6f}", flush=True)
                    _add_to_pool(pool, saved, pool_size)
                attempt += batch_size

                if time_limit is not None and attempt < n_attempts and (time.perf_counter() - start_time) > time_limit:
                    if verbose:
                        print(
                            f"Stopping after {attempt} attempts because time_limit={time_limit} s was exceeded.",
                            flush=True,
                        )
                    break

    distinct: list[SavedConfiguration] = []
    for candidate in pool:  # already sorted by ascending energy
        if len(distinct) >= n_return:
            break
        is_duplicate = any(
            symmetrically_equivalent(candidate.structure, kept.structure, **symmetry_match_kwargs)
            for kept in distinct
        )
        if not is_duplicate:
            distinct.append(candidate)

    return distinct
