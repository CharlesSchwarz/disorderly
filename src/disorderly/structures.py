"""Utilities for analyzing and manipulating ordered and disordered structures.

This module intentionally does not depend on symmetry, energy, or Monte
Carlo functionality.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Union
import warnings

from pymatgen.core import Structure
from pymatgen.core.periodic_table import DummySpecies, Element, Species, get_el_sp

#: Anything pymatgen can resolve to a Species/Element/DummySpecies via
#: :func:`pymatgen.core.periodic_table.get_el_sp`.
SpeciesLike = Union[str, Element, Species, DummySpecies]

__all__ = [
    "replace_species_and_occupancy",
    "set_site_occupancies",
    "is_disordered",
    "disordered_site_indices",
    "DisorderedSiteGroup",
    "disordered_site_groups",
    "supercell_multiplier",
    "count_configurations",
]


def replace_species_and_occupancy(
    structure: Structure,
    target_species: SpeciesLike,
    new_species: SpeciesLike,
    target_occupancy: float | None = None,
    new_occupancy: float | None = None,
    occupancy_tol: float = 1e-2,
) -> Structure:
    """Replace a species (and optionally its occupancy) throughout a structure.

    For every site that carries ``target_species`` (optionally restricted to
    sites where that species has occupancy close to ``target_occupancy``),
    the ``target_species`` entry on that site is replaced by ``new_species``
    at ``new_occupancy``. Any *other* species already present on the same
    site (e.g. a mixed-occupancy site such as ``Li+:0.708, Fe3+:0.0625``) are
    left untouched, so partially occupied and mixed sites are handled
    correctly rather than being silently overwritten.

    Args:
        structure: Structure to modify. Not mutated; a modified copy is
            returned.
        target_species: Species to look for on each site (e.g. ``"Li+"``,
            ``Species("Li", 1)``, or ``Element("Li")``). Matching is exact on
            the resolved species (including oxidation state, if any) -- it
            does not match other oxidation states of the same element.
        new_species: Species to substitute in place of ``target_species``.
        target_occupancy: If given, only replace ``target_species`` on sites
            where its occupancy is within ``occupancy_tol`` of this value.
            If ``None`` (default), replace ``target_species`` wherever it is
            found, regardless of its occupancy on that site.
        new_occupancy: Occupancy to assign to ``new_species``. If ``None``
            (default), the occupancy of the replaced ``target_species`` entry
            is preserved (i.e. only the species identity changes).
        occupancy_tol: Absolute tolerance used both to match
            ``target_occupancy`` and to validate that the resulting site
            occupancy does not exceed 1.

    Returns:
        A modified copy of ``structure``.

    Raises:
        ValueError: If ``target_species`` is not found on any site of
            ``structure``, or if replacing it would push a site's total
            occupancy above 1 (within ``occupancy_tol``).
    """
    target = get_el_sp(target_species)
    new = get_el_sp(new_species)

    modified = structure.copy()
    n_replaced = 0

    for site in modified:
        # Composition.get() returns 0 (not None) for a missing key, since
        # Composition treats missing species as having amount zero -- so
        # presence must be checked with `in`, not by testing the looked-up
        # amount.
        if target not in site.species:
            continue
        current_occ = site.species[target]
        if target_occupancy is not None and not math.isclose(
            current_occ, target_occupancy, abs_tol=occupancy_tol
        ):
            continue

        occ_to_assign = current_occ if new_occupancy is None else new_occupancy

        new_species_map = {
            sp: occ for sp, occ in site.species.items() if sp != target
        }
        new_species_map[new] = new_species_map.get(new, 0.0) + occ_to_assign

        total_occ = sum(new_species_map.values())
        if total_occ > 1.0 + occupancy_tol:
            raise ValueError(
                f"Replacing {target} with {new} at occupancy {occ_to_assign} "
                f"would give site {site} a total occupancy of {total_occ:.4f}, "
                "which exceeds 1."
            )

        site.species = new_species_map
        n_replaced += 1

    if n_replaced == 0:
        occ_msg = "" if target_occupancy is None else f" at occupancy {target_occupancy}"
        raise ValueError(
            f"Species {target} was not found on any site of the structure{occ_msg}."
        )

    return modified


def set_site_occupancies(
    structure: Structure,
    site_indices: Sequence[int],
    occupancies: Mapping[SpeciesLike, float],
    occupancy_tol: float = 1e-2,
) -> Structure:
    """Set the species and occupancies of selected sites.

    Unlike :func:`replace_species_and_occupancy`, which swaps one species for
    another, this overwrites everything on each selected site, so it can add
    a co-occupying species (e.g. a dopant) or change several occupancies at
    once. Combine with :func:`disordered_site_groups` to select a whole
    disordered sublattice.

    Args:
        structure: Structure to modify. Not mutated; a modified copy is
            returned.
        site_indices: Indices of the sites to modify.
        occupancies: New occupancy of each species on every selected site,
            e.g. ``{"Li+": 7 / 24, "Ga3+": 2 / 24}``. Species with zero
            occupancy are dropped; the remainder up to 1 is vacant.
        occupancy_tol: Absolute tolerance on the total occupancy exceeding 1.

    Returns:
        A modified copy of ``structure``.

    Raises:
        ValueError: If an occupancy is negative, no species has a positive
            occupancy, or the occupancies sum to more than 1.
    """
    species: dict = {}
    for sp, occ in occupancies.items():
        if occ < 0:
            raise ValueError(f"Occupancy of {sp} must be non-negative, got {occ}.")
        if occ > 0:
            resolved = get_el_sp(sp)
            species[resolved] = species.get(resolved, 0.0) + float(occ)
    if not species:
        raise ValueError(
            "At least one species must have a positive occupancy; remove the "
            "site instead to make it fully vacant."
        )
    total_occ = sum(species.values())
    if total_occ > 1.0 + occupancy_tol:
        raise ValueError(f"Occupancies {occupancies} sum to {total_occ:.4f}, which exceeds 1.")

    modified = structure.copy()
    for i in site_indices:
        modified[i].species = species
    return modified


def is_disordered(structure: Structure) -> bool:
    """Return whether any site of ``structure`` is not fully, singly occupied."""
    return not structure.is_ordered


def disordered_site_indices(structure: Structure) -> list[int]:
    """Indices of sites in ``structure`` that are not fully, singly occupied."""
    return [i for i, site in enumerate(structure) if not site.is_ordered]


@dataclass(frozen=True)
class DisorderedSiteGroup:
    """A pool of interchangeable disordered sites that all carry the same
    species and occupancies, as returned by :func:`disordered_site_groups`.

    Vacancies are keyed by ``None`` in ``ideal_counts`` and ``counts`` (the
    same convention as :mod:`disorderly.ordering`), and only appear when
    the occupancies sum to less than 1.

    Attributes:
        site_indices: Indices of the group's sites in the structure.
        occupancies: Per-site occupancy of each species, sorted by species
            label.
        ideal_counts: ``n_sites * occupancy`` for each species (and the
            vacancy). Not necessarily whole numbers.
        counts: Whole numbers of each species (and vacancies) an ordered
            configuration of the structure places on these sites -- exactly
            what :func:`disorderly.sampling.random_ordered_configuration`
            uses.
        commensurate: Whether every ideal count is a whole number, i.e.
            whether ``counts`` reproduces the occupancies exactly.
    """

    site_indices: tuple[int, ...]
    occupancies: dict
    ideal_counts: dict
    counts: dict
    commensurate: bool

    @property
    def n_sites(self) -> int:
        return len(self.site_indices)


def _site_signature(site) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((str(sp), float(occ)) for sp, occ in site.species.items()))


def _is_whole(value: float, count_tol: float) -> bool:
    return abs(value - round(value)) <= count_tol


def _count_label(sp) -> str:
    return "vacancy" if sp is None else str(sp)


def _format_counts(counts: Mapping) -> str:
    return ", ".join(f"{_count_label(sp)}: {n:g}" for sp, n in counts.items())


def disordered_site_groups(
    structure: Structure,
    occupancy_tol: float = 1e-2,
    count_tol: float = 0.05,
    warn: bool = True,
) -> list[DisorderedSiteGroup]:
    """Group the disordered sites of ``structure`` and count the species an
    ordered configuration places on each group.

    Disordered sites are grouped by their exact species/occupancy signature
    (e.g. all sites with occupancy ``{Li+: 0.5417}``); sites sharing a
    signature are one interchangeable pool. Physically distinct sublattices
    that happen to share a signature are pooled together.

    Each group's occupancies are converted to whole-site counts. If they sum
    to (approximately) 1, every site gets a species and the last species (by
    sorted label) absorbs any rounding remainder. Otherwise each species'
    count is rounded independently and the remaining sites are vacant.

    Args:
        structure: Disordered structure.
        occupancy_tol: Absolute tolerance for treating a group's total
            occupancy as exactly 1.
        count_tol: Absolute tolerance (in sites) for treating
            ``n_sites * occupancy`` as a whole number.
        warn: If true, emit a ``UserWarning`` when any group's occupancies
            are not whole numbers of sites in this cell, recommending how
            many times the cell must be multiplied (see
            :func:`supercell_multiplier`).

    Returns:
        One :class:`DisorderedSiteGroup` per signature, in order of first
        appearance in ``structure``.

    Raises:
        ValueError: If rounding produces more atoms than a group has sites
            (only possible in pathological rounding edge cases with several
            species per site).
    """
    pools: dict[tuple[tuple[str, float], ...], list[int]] = {}
    for i, site in enumerate(structure):
        if not site.is_ordered:
            pools.setdefault(_site_signature(site), []).append(i)

    groups = []
    for indices in pools.values():
        species_occ = sorted(structure[indices[0]].species.items(), key=lambda kv: str(kv[0]))
        n_sites = len(indices)
        total_occ = sum(occ for _, occ in species_occ)
        ideal_counts = {sp: occ * n_sites for sp, occ in species_occ}

        # pymatgen guarantees every site's species occupancies already sum to
        # at most 1 (+ a small tolerance), so `total_occ > 1` cannot occur
        # for a validly constructed `structure`.
        counts: dict = {}
        if math.isclose(total_occ, 1.0, abs_tol=occupancy_tol):
            assigned = 0
            for sp, occ in species_occ[:-1]:
                counts[sp] = round(occ * n_sites)
                assigned += counts[sp]
            counts[species_occ[-1][0]] = n_sites - assigned
        else:
            for sp, occ in species_occ:
                counts[sp] = round(occ * n_sites)
            counts[None] = n_sites - sum(counts.values())
            ideal_counts[None] = (1.0 - total_occ) * n_sites

        if any(n < 0 for n in counts.values()):
            raise ValueError(
                f"Computed site counts {_format_counts(counts)} are inconsistent with the "
                f"{n_sites} available sites for this disordered site group."
            )

        groups.append(
            DisorderedSiteGroup(
                site_indices=tuple(indices),
                occupancies=dict(species_occ),
                ideal_counts=ideal_counts,
                counts=counts,
                commensurate=all(_is_whole(x, count_tol) for x in ideal_counts.values()),
            )
        )

    if warn and not all(group.commensurate for group in groups):
        _warn_incommensurate(groups, count_tol)
    return groups


def _multiplier_for_groups(
    groups: Sequence[DisorderedSiteGroup], count_tol: float, max_multiplier: int
) -> int | None:
    for m in range(1, max_multiplier + 1):
        if all(_is_whole(m * x, count_tol) for group in groups for x in group.ideal_counts.values()):
            return m
    return None


def _warn_incommensurate(
    groups: Sequence[DisorderedSiteGroup], count_tol: float, max_multiplier: int = 1000
) -> None:
    details = "; ".join(
        f"{group.n_sites} sites of {_format_counts(group.occupancies)} hold "
        f"{_format_counts({sp: round(x, 3) for sp, x in group.ideal_counts.items()})}, "
        f"rounded to {_format_counts(group.counts)}"
        for group in groups
        if not group.commensurate
    )
    m = _multiplier_for_groups(groups, count_tol, max_multiplier)
    if m is None:
        advice = f"No supercell up to {max_multiplier}x this cell makes every count a whole number."
    else:
        advice = (
            f"Multiply the cell by {m} (any supercell with {m}x as many sites, "
            f"e.g. structure.make_supercell([{m}, 1, 1])) to make every count a whole number."
        )
    warnings.warn(
        "Site occupancies are not whole numbers of atoms in this cell, so ordered "
        f"configurations will not reproduce them exactly: {details}. {advice}",
        UserWarning,
        stacklevel=3,
    )


def supercell_multiplier(
    structure: Structure,
    occupancy_tol: float = 1e-2,
    count_tol: float = 0.05,
    max_multiplier: int = 1000,
) -> int | None:
    """Smallest factor the cell must be multiplied by for every disordered
    occupancy to correspond to a whole number of atoms.

    Returns 1 if the occupancies already fit this cell. Any supercell with
    that many times as many sites works (e.g. ``make_supercell([m, 1, 1])``),
    though a more isotropic one is usually preferable for sampling.

    Args:
        structure: Disordered structure.
        occupancy_tol: Passed to :func:`disordered_site_groups`.
        count_tol: Absolute tolerance (in sites) for treating a count as a
            whole number.
        max_multiplier: Largest factor to try.

    Returns:
        The multiplier, or ``None`` if no factor up to ``max_multiplier``
        works.
    """
    groups = disordered_site_groups(structure, occupancy_tol=occupancy_tol, count_tol=count_tol, warn=False)
    return _multiplier_for_groups(groups, count_tol, max_multiplier)


def count_configurations(structure: Structure, occupancy_tol: float = 1e-2) -> int:
    """Total number of ordered configurations of ``structure`` in this cell.

    For each disordered site group the number of ways to place its whole-site
    counts (see :func:`disordered_site_groups`), vacancies included, is the
    multinomial coefficient ``n! / (n_1! n_2! ...)``; the total is the
    product over groups. Configurations related by symmetry are counted
    separately -- see :func:`disorderly.symmetry.count_distinct_configurations`.

    Args:
        structure: Disordered structure.
        occupancy_tol: Passed to :func:`disordered_site_groups`.

    Returns:
        The exact number of configurations (an arbitrary-precision int).
    """
    total = 1
    for group in disordered_site_groups(structure, occupancy_tol=occupancy_tol):
        ways = math.factorial(group.n_sites)
        for n in group.counts.values():
            ways //= math.factorial(n)
        total *= ways
    return total
