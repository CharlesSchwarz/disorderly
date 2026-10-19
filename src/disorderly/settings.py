"""User-configurable Monte Carlo behavior.

The Monte Carlo engine in :mod:`disorderly.monte_carlo` calls into a
:class:`MonteCarloPolicy` for all problem-specific behavior -- which moves
to propose, whether a proposed move is allowed, and which properties (and
in particular, which scalar energy) to evaluate for a candidate
configuration. This module defines that policy interface, together with a
simple, generically applicable default policy (canonical-ensemble site
swaps) that users can use directly or take as a template for their own.

This module intentionally does not depend on :mod:`disorderly.monte_carlo`,
so that the engine can depend on the policy interface without a circular
import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

import numpy as np
from pymatgen.core import Structure

from disorderly.energy import disordered_ewald_energy

__all__ = [
    "DetailedBalanceWarning",
    "MoveProposal",
    "MonteCarloPolicy",
    "default_policy",
    "default_choose_potential_move",
    "default_is_move_allowed",
    "default_evaluate_props",
    "preserves_composition",
    "sublattice_swap_policy",
    "sublattice_choose_potential_move",
    "sublattice_site_indices",
]


class DetailedBalanceWarning(UserWarning):
    """Emitted when a Monte Carlo run may not satisfy detailed balance.

    Such a run still explores configurations and its energies are exact,
    so it remains usable as a search for low-energy configurations, but it
    does not sample the Boltzmann distribution: averages over its
    trajectory (site occupancies, order parameters, mean energy) are biased.
    Filter it with :mod:`warnings` once the bias is understood and intended.
    """


@dataclass(frozen=True)
class MoveProposal:
    """A candidate Monte Carlo move, as produced by ``choose_potential_move``.

    Attributes:
        move_type: A short label for the kind of move (e.g. ``"swap"`` or
            ``"move"``). Not interpreted by the engine itself; it is passed
            through to :func:`disorderly.monte_carlo.propose_next_configuration`,
            which dispatches on it to decide how to build the candidate
            structure (via ``swap_ion``/``move_ion``), and is recorded in
            the trajectory for analysis.
        site_index: Index (in the current configuration) of the site whose
            species is being relocated.
        target_site_index: For a ``"swap"`` move, the index of the other
            occupied site to swap species with. ``None`` for other move
            types.
        target_frac_coords: For a ``"move"`` move, the fractional
            coordinates the species at ``site_index`` should be relocated
            to (typically an unoccupied allowed site known to the policy's
            ``context``, e.g. a parent lattice). ``None`` for other move
            types.
        metadata: Free-form additional information about the move (e.g.
            species involved, site labels), recorded in the trajectory and
            available to ``is_move_allowed``.
        log_proposal_ratio: ``ln[g(b->a) / g(a->b)]``, where ``g(a->b)`` is
            the probability that ``choose_potential_move`` proposes this
            move from the current configuration ``a`` and ``g(b->a)`` the
            probability that it proposes the reverse move from the
            candidate ``b``. The engine adds it to the Metropolis exponent
            (the Metropolis--Hastings correction), so a policy whose
            proposals are not symmetric can still satisfy detailed balance
            by reporting it. ``0`` (the default) for symmetric proposals,
            such as all the built-in policies'.
    """

    move_type: str
    site_index: int
    target_site_index: int | None = None
    target_frac_coords: tuple[float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    log_proposal_ratio: float = 0.0


class MoveChooser(Protocol):
    def __call__(
        self, configuration: Structure, rng: np.random.Generator, context: dict[str, Any]
    ) -> MoveProposal | None: ...


class MoveValidator(Protocol):
    def __call__(
        self,
        configuration: Structure,
        proposed_configuration: Structure,
        move: MoveProposal,
        context: dict[str, Any],
    ) -> bool: ...


class PropertyEvaluator(Protocol):
    def __call__(self, configuration: Structure, context: dict[str, Any]) -> dict[str, float]: ...


@dataclass
class MonteCarloPolicy:
    """Bundles the user-configurable behavior of a Monte Carlo simulation.

    Args:
        choose_potential_move: Chooses the type of move and/or the
            sites/species involved in a proposed move. Should return
            ``None`` if no move could be proposed (the engine will retry,
            up to ``max_move_attempts``).
        is_move_allowed: Determines whether a proposed move/configuration is
            allowed, e.g. to enforce problem-specific constraints (allowed
            sublattices, mobile species, minimum distances). Defaults to
            :func:`default_is_move_allowed` (only checks that composition is
            preserved).
        evaluate_props: Evaluates properties for a newly generated
            configuration. **The Monte Carlo engine obtains the scalar
            energy used for the Metropolis acceptance criterion from
            ``evaluate_props(configuration, context)["energy"]``** -- any
            policy supplied to the engine must include an ``"energy"`` key
            in the dict it returns. Additional keys are recorded in the
            trajectory but not used for acceptance. Defaults to
            :func:`default_evaluate_props` (disordered Ewald energy).
        context: Arbitrary user data made available to all three callables
            above (e.g. a parent/allowed-sites lattice, mobile species
            list, neighbor cutoff radius). Opaque to the engine.
        detailed_balance: Whether ``choose_potential_move`` (together with
            each proposal's ``log_proposal_ratio``) satisfies detailed
            balance: every move's reverse can be proposed, and the forward
            and reverse proposal probabilities are equal or corrected for by
            ``log_proposal_ratio``. The engine cannot check this itself; it
            emits a :class:`DetailedBalanceWarning` at the start of a
            finite-temperature run if this is ``False`` (a policy
            deliberately breaking detailed balance) or ``None`` (not
            declared). :func:`disorderly.diagnostics.check_proposal_symmetry`
            tests a policy's proposals empirically. The built-in policies
            declare ``True``.
    """

    choose_potential_move: MoveChooser
    is_move_allowed: MoveValidator = None  # type: ignore[assignment]
    evaluate_props: PropertyEvaluator = None  # type: ignore[assignment]
    context: dict[str, Any] = field(default_factory=dict)
    detailed_balance: bool | None = None

    def __post_init__(self) -> None:
        if self.is_move_allowed is None:
            self.is_move_allowed = default_is_move_allowed
        if self.evaluate_props is None:
            self.evaluate_props = default_evaluate_props


def preserves_composition(structure1: Structure, structure2: Structure, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Whether two structures have the same overall composition.

    A convenience building block for custom ``is_move_allowed`` functions
    that must enforce fixed overall composition.
    """
    return structure1.composition.almost_equals(structure2.composition, rtol=rtol, atol=atol)


def default_choose_potential_move(
    configuration: Structure, rng: np.random.Generator, context: dict[str, Any]
) -> MoveProposal | None:
    """Propose swapping the species of two randomly chosen sites that
    currently hold different species.

    This is a generically applicable canonical-ensemble move: it requires
    no knowledge of a parent lattice or of which sites/species are
    "allowed" to move, only that ``configuration`` is fully occupied. Note
    that *any* two sites with different species may be swapped, including
    an anion with a cation; use :func:`sublattice_swap_policy` to restrict
    swaps to the disordered sublattice(s) of a parent structure. Problems
    with a genuine vacant sublattice (partial occupancy, ion hopping) need
    a custom ``choose_potential_move`` that also proposes ``"move"``-type
    moves to vacant sites; see the package README/notebooks for an example.

    A swap leaves every species count unchanged, so a swap and its reverse
    are proposed with the same probability and plain Metropolis acceptance
    satisfies detailed balance.
    """
    n_sites = len(configuration)
    if n_sites < 2:
        return None
    site_index = int(rng.integers(n_sites))
    site_species = configuration[site_index].species
    candidates = [i for i in range(n_sites) if i != site_index and configuration[i].species != site_species]
    if not candidates:
        return None
    target_index = int(rng.choice(candidates))
    return MoveProposal(
        move_type="swap",
        site_index=site_index,
        target_site_index=target_index,
        metadata={
            "species_from": str(site_species),
            "species_to": str(configuration[target_index].species),
        },
    )


def default_is_move_allowed(
    configuration: Structure,
    proposed_configuration: Structure,
    move: MoveProposal,
    context: dict[str, Any],
) -> bool:
    """Allow a move iff it preserves the overall composition.

    Sufficient as a safety net for swap-only policies (which preserve
    composition by construction); required for policies that also propose
    ``"move"``-type moves to previously vacant sites, where a bug could
    otherwise create or destroy species.
    """
    return preserves_composition(configuration, proposed_configuration)


def default_evaluate_props(configuration: Structure, context: dict[str, Any]) -> dict[str, float]:
    """Evaluate the disordered Ewald energy of ``configuration``.

    Returns a dict with a single ``"energy"`` key, in eV (see
    :func:`disorderly.energy.disordered_ewald_energy`). ``configuration``
    must carry explicit oxidation states.
    """
    return {"energy": disordered_ewald_energy(configuration)}


def default_policy(context: dict[str, Any] | None = None) -> MonteCarloPolicy:
    """A ready-to-use policy: random composition-preserving site swaps,
    scored by disordered Ewald energy."""
    return MonteCarloPolicy(
        choose_potential_move=default_choose_potential_move,
        is_move_allowed=default_is_move_allowed,
        evaluate_props=default_evaluate_props,
        context=dict(context or {}),
        detailed_balance=True,
    )


#: ``context`` key under which :func:`sublattice_swap_policy` stores the
#: fractional coordinates (one ``(n, 3)`` array per sublattice) of the parent
#: sites that may exchange species.
SWAP_SUBLATTICES_KEY = "swap_sublattices"
#: ``context`` key for the maximum distance (Angstrom) between a site of a
#: configuration and the sublattice position it is matched to.
SWAP_SITE_TOL_KEY = "swap_site_tol"


def sublattice_site_indices(
    configuration: Structure, sublattices: Sequence[np.ndarray], tol: float = 0.5
) -> list[list[int]]:
    """Indices of ``configuration``'s sites lying on each swap sublattice.

    Sites are matched to sublattice positions by (periodic) distance, not by
    index, because ordered configurations need not list their sites in the
    parent's order (e.g. :func:`disorderly.sampling.random_ordered_configuration`
    lists already-ordered sites first). Sites farther than ``tol`` from every
    sublattice position belong to no sublattice; sublattice positions with
    no matching site (vacancies) are simply absent from the result.

    Args:
        configuration: Ordered configuration on the parent's lattice.
        sublattices: Fractional coordinates of each sublattice's positions,
            as stored by :func:`sublattice_swap_policy`.
        tol: Maximum matching distance in Angstrom.

    Returns:
        One list of ``configuration`` site indices per sublattice.
    """
    groups: list[list[int]] = [[] for _ in sublattices]
    if not len(sublattices) or not len(configuration):
        return groups
    positions = np.concatenate([np.asarray(coords, dtype=float).reshape(-1, 3) for coords in sublattices])
    labels = np.concatenate([np.full(len(coords), g) for g, coords in enumerate(sublattices)])
    dists = configuration.lattice.get_all_distances(configuration.frac_coords, positions)
    nearest = dists.argmin(axis=1)
    for i, j in enumerate(nearest):
        if dists[i, j] <= tol:
            groups[int(labels[j])].append(i)
    return groups


def sublattice_choose_potential_move(
    configuration: Structure, rng: np.random.Generator, context: dict[str, Any]
) -> MoveProposal | None:
    """Propose swapping two sites of the same sublattice that hold different
    species.

    The sublattices are read from ``context[SWAP_SUBLATTICES_KEY]`` (see
    :func:`sublattice_swap_policy`). A site is chosen uniformly from all
    sublattice sites, then a partner uniformly from the sites of the same
    sublattice holding a different species. Because a swap leaves each
    sublattice's species counts unchanged, the forward and reverse
    proposal probabilities of every swap are equal, so plain Metropolis
    acceptance satisfies detailed balance.

    Returns ``None`` if the chosen site has no valid partner (the engine
    then retries).
    """
    groups = sublattice_site_indices(
        configuration, context[SWAP_SUBLATTICES_KEY], tol=context.get(SWAP_SITE_TOL_KEY, 0.5)
    )
    sites = [(g, i) for g, group in enumerate(groups) for i in group]
    if len(sites) < 2:
        return None
    group_index, site_index = sites[int(rng.integers(len(sites)))]
    site_species = configuration[site_index].species
    partners = [j for j in groups[group_index] if configuration[j].species != site_species]
    if not partners:
        return None
    target_index = int(rng.choice(partners))
    return MoveProposal(
        move_type="swap",
        site_index=site_index,
        target_site_index=target_index,
        metadata={
            "sublattice": group_index,
            "species_from": str(site_species),
            "species_to": str(configuration[target_index].species),
        },
    )


def _species_signature(site) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((str(sp), float(occ)) for sp, occ in site.species.items()))


def sublattice_swap_policy(
    parent: Structure,
    sublattices: Sequence[Sequence[int]] | None = None,
    evaluate_props: PropertyEvaluator | None = None,
    context: dict[str, Any] | None = None,
    site_tol: float = 0.5,
) -> MonteCarloPolicy:
    """A policy that only swaps species within sublattices of ``parent``.

    Unlike :func:`default_policy`, which may swap any two different species
    (including anions with cations), this restricts swaps to chosen groups
    of parent sites, e.g. the Li/Ti-shared 16d sites of Li4Ti5O12.

    Args:
        parent: The disordered parent structure. Configurations passed to
            the engine must be on the same lattice; their sites are matched
            to parent sites by position.
        sublattices: Groups of ``parent`` site indices; swaps happen only
            between two sites of the same group. Groups may include ordered
            parent sites (e.g. all cation sites of a spinel, to allow
            inversion). Defaults to one group per distinct species/occupancy
            signature among ``parent``'s disordered sites -- the same pools
            used by :func:`disorderly.sampling.random_ordered_configuration`.
        evaluate_props: Property evaluator; defaults to
            :func:`default_evaluate_props` (disordered Ewald energy).
        context: Extra user context. The sublattice positions are added
            under ``SWAP_SUBLATTICES_KEY`` and ``site_tol`` under
            ``SWAP_SITE_TOL_KEY``.
        site_tol: Maximum distance (Angstrom) between a configuration site
            and the parent site it is matched to.

    Returns:
        A :class:`MonteCarloPolicy` using
        :func:`sublattice_choose_potential_move` and
        :func:`default_is_move_allowed`.

    Raises:
        ValueError: If ``sublattices`` is not given and ``parent`` has no
            disordered sites, or if a group has fewer than two sites, an
            out-of-range index, or shares a site with another group.
    """
    n_sites = len(parent)
    if sublattices is None:
        pools: dict[tuple[tuple[str, float], ...], list[int]] = {}
        for i, site in enumerate(parent):
            if not site.is_ordered:
                pools.setdefault(_species_signature(site), []).append(i)
        if not pools:
            raise ValueError("`parent` has no disordered sites; pass `sublattices` explicitly.")
        sublattices = list(pools.values())

    seen: set[int] = set()
    for group in sublattices:
        if len(group) < 2:
            raise ValueError(f"Each sublattice needs at least two sites to swap between; got {list(group)}.")
        for i in group:
            if not 0 <= i < n_sites:
                raise ValueError(f"Sublattice site index {i} out of range for a parent with {n_sites} sites.")
            if i in seen:
                raise ValueError(f"Parent site {i} appears in more than one sublattice.")
            seen.add(i)

    frac_coords = np.asarray(parent.frac_coords)
    policy_context = dict(context or {})
    policy_context[SWAP_SUBLATTICES_KEY] = [frac_coords[list(group)] for group in sublattices]
    policy_context[SWAP_SITE_TOL_KEY] = site_tol
    return MonteCarloPolicy(
        choose_potential_move=sublattice_choose_potential_move,
        is_move_allowed=default_is_move_allowed,
        evaluate_props=evaluate_props or default_evaluate_props,
        context=policy_context,
        detailed_balance=True,
    )
