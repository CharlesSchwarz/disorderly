"""Symmetry analysis and comparison of ordered and disordered structures.

May depend on :mod:`disorder_mc.structures` but not on energy or Monte Carlo
functionality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import PeriodicSite, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from disorder_mc.structures import DisorderedSiteGroup, disordered_site_groups

__all__ = [
    "get_spacegroup",
    "get_wyckoff_sites",
    "get_wyckoff_labels",
    "label_wyckoff_sites",
    "describe_wyckoff_sites",
    "symmetrically_equivalent",
    "configuration_degeneracy",
    "count_distinct_configurations",
    "site_permutations",
]


@dataclass(frozen=True)
class SpacegroupInfo:
    """Space group of a structure, as returned by :func:`get_spacegroup`."""

    symbol: str
    number: int
    crystal_system: str


def get_spacegroup(structure: Structure, symprec: float = 1e-3, angle_tolerance: float = 5.0) -> SpacegroupInfo:
    """Determine the crystallographic space group of ``structure``.

    A thin wrapper around :class:`pymatgen.symmetry.analyzer.SpacegroupAnalyzer`.

    Args:
        structure: Structure to analyze. May be ordered or disordered;
            spglib treats a disordered site as occupied by all its species,
            so disordered structures are typically only meaningful here to
            the extent that their *parent* lattice symmetry is of interest.
        symprec: Symmetry-finding tolerance (Angstrom), passed to spglib.
        angle_tolerance: Symmetry-finding angle tolerance (degrees), passed
            to spglib.

    Returns:
        Space group symbol, international number, and crystal system.
    """
    sga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
    return SpacegroupInfo(
        symbol=sga.get_space_group_symbol(),
        number=sga.get_space_group_number(),
        crystal_system=sga.get_crystal_system(),
    )


def get_wyckoff_sites(
    structure: Structure, symprec: float = 1e-3, angle_tolerance: float = 5.0
) -> list[str]:
    """Determine the Wyckoff letter of each site in ``structure``.

    Args:
        structure: Structure to analyze.
        symprec: Symmetry-finding tolerance (Angstrom), passed to spglib.
        angle_tolerance: Symmetry-finding angle tolerance (degrees), passed
            to spglib.

    Returns:
        List of Wyckoff letters (e.g. ``"a"``, ``"b"``), one per site of
        ``structure``, in the same order as ``structure``'s sites.
    """
    sga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
    dataset = sga.get_symmetry_dataset()
    wyckoffs = getattr(dataset, "wyckoffs", None)
    if wyckoffs is None:
        wyckoffs = dataset["wyckoffs"]
    return list(wyckoffs)


def get_wyckoff_labels(
    structure: Structure, symprec: float = 1e-3, angle_tolerance: float = 5.0
) -> list[str]:
    """Full Wyckoff label (multiplicity and letter, e.g. ``"24d"``) of each
    site in ``structure``.

    Built on :meth:`SpacegroupAnalyzer.get_symmetrized_structure`, whose
    ``wyckoff_symbols`` count the sites of each orbit in the *input* cell.
    That count is only the crystallographic multiplicity when the input is
    the standard conventional cell, so it is rescaled here by
    (conventional cell volume / input cell volume). The same crystal
    therefore gets the same labels whether it is given as a conventional
    cell, a primitive cell, or a supercell (e.g. rock salt is always
    ``4a``/``4b``, never ``1a``/``1b`` or ``8a``/``8b``).

    Conventions follow spglib's standardized setting:

    * Rhombohedral (R) space groups use hexagonal axes, so multiplicities
      are those of the hexagonal (triple) cell, e.g. ``3a`` rather than
      ``1a``.
    * Letters refer to the standard setting and origin choice, which may
      differ from the labels in a source CIF that uses another setting.
      Where the space group allows equivalent origins, spglib may pick a
      different one for different cells of the same crystal, swapping
      letters of equal multiplicity (e.g. spinel Fd-3m ``8a``/``16d`` in
      the conventional cell but ``8b``/``16c`` in a 3x1x1 supercell).
      Multiplicities are unaffected.
    * Disordered sites are supported: each distinct species/occupancy is
      its own atom type, so sublattices with different occupancies are
      never merged. If ``symprec`` is too tight for noisy coordinates the
      detected symmetry can drop to P1, where every site is ``1a``.

    Args:
        structure: Structure to analyze (ordered or disordered).
        symprec: Symmetry-finding tolerance (Angstrom), passed to spglib.
        angle_tolerance: Symmetry-finding angle tolerance (degrees), passed
            to spglib.

    Returns:
        One label per site, in the same order as ``structure``'s sites.

    Raises:
        ValueError: If a rescaled multiplicity is not a whole number, which
            indicates inconsistent symmetry detection.
    """
    labels = [""] * len(structure)
    for indices, label in _wyckoff_orbits(structure, symprec, angle_tolerance):
        for i in indices:
            labels[i] = label
    return labels


def _wyckoff_orbits(structure: Structure, symprec: float, angle_tolerance: float) -> list[tuple[list[int], str]]:
    """Site indices of each Wyckoff orbit of ``structure`` with its full
    label (e.g. ``"24d"``), in the order pymatgen's symmetrized structure
    lists them. See :func:`get_wyckoff_labels` for the label conventions."""
    sga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
    symmetrized = sga.get_symmetrized_structure()
    dataset = sga.get_symmetry_dataset()
    std_lattice = getattr(dataset, "std_lattice", None)
    if std_lattice is None:
        std_lattice = dataset["std_lattice"]
    cell_ratio = abs(np.linalg.det(std_lattice)) / structure.volume

    orbits = []
    for indices, symbol in zip(symmetrized.equivalent_indices, symmetrized.wyckoff_symbols):
        multiplicity = len(indices) * cell_ratio
        if abs(multiplicity - round(multiplicity)) > 1e-2:
            raise ValueError(
                f"Wyckoff orbit {symbol} rescales to a non-integer multiplicity "
                f"{multiplicity:.3f}; symmetry detection is inconsistent (try another symprec)."
            )
        orbits.append((list(indices), f"{round(multiplicity)}{symbol.lstrip('0123456789')}"))
    return orbits


def _wyckoff_site_label(site: PeriodicSite, wyckoff: str) -> str:
    """``"<elements>_<wyckoff>"`` label of a site, e.g. ``"Li-Ti_16d"``."""
    elements = dict.fromkeys(sp.symbol for sp in sorted(site.species, key=str))
    return f"{'-'.join(elements)}_{wyckoff}"


def label_wyckoff_sites(
    structure: Structure, symprec: float = 1e-3, angle_tolerance: float = 5.0
) -> Structure:
    """Label every site of ``structure`` with its Wyckoff position.

    Adds a ``"wyckoff"`` site property (see :func:`get_wyckoff_labels`) and
    sets each site's label to ``"<elements>_<wyckoff>"``, e.g. ``"Li_24d"``
    or ``"Li-Ti_16d"`` for a mixed site. pymatgen writes site labels to CIF
    (``_atom_site_label``), so the labels survive
    :meth:`pymatgen.core.Structure.to`.

    Args:
        structure: Structure to label. Not mutated; a labelled copy is
            returned.
        symprec, angle_tolerance: Passed to :func:`get_wyckoff_labels`.

    Returns:
        A labelled copy of ``structure``.
    """
    wyckoffs = get_wyckoff_labels(structure, symprec=symprec, angle_tolerance=angle_tolerance)
    labelled = structure.copy()
    labelled.add_site_property("wyckoff", wyckoffs)
    for site, wyckoff in zip(labelled, wyckoffs):
        site.label = _wyckoff_site_label(site, wyckoff)
    return labelled


def describe_wyckoff_sites(
    structure: Structure,
    symprec: float = 1e-3,
    angle_tolerance: float = 5.0,
    occupancy_tol: float = 1e-2,
) -> str:
    """Summarize ``structure`` one Wyckoff orbit at a time, as a text table.

    Each row is one orbit of symmetry-equivalent sites, labelled as in
    :func:`label_wyckoff_sites` (e.g. ``Li_24d``), with the species and
    occupancies on its sites, the number of its sites in the cell, whether
    it is disordered, and, for a disordered orbit, the whole numbers of
    each species and vacancies that every ordered configuration places on
    it (the counts of :func:`disorder_mc.structures.disordered_site_groups`,
    as used by :func:`disorder_mc.sampling.random_ordered_configuration`).

    Distinct orbits that share a Wyckoff label get separate rows. Ordered
    configurations pool all disordered sites with the same species and
    occupancies, even across orbits; when such a pool spans several orbits,
    its counts are for the whole pool and the row names the orbits it is
    pooled with.

    Args:
        structure: Structure to describe (ordered or disordered).
        symprec, angle_tolerance: Passed to :func:`get_wyckoff_labels`.
        occupancy_tol: Passed to
            :func:`disorder_mc.structures.disordered_site_groups`.

    Returns:
        The table as a multi-line string, e.g. for ``print``.
    """
    orbits = _wyckoff_orbits(structure, symprec, angle_tolerance)
    labels = [_wyckoff_site_label(structure[indices[0]], wyckoff) for indices, wyckoff in orbits]
    orbit_of_site = {i: k for k, (indices, _) in enumerate(orbits) for i in indices}
    group_of_orbit = {}
    groups = disordered_site_groups(structure, occupancy_tol=occupancy_tol, warn=False)
    for group in groups:
        for i in group.site_indices:
            group_of_orbit[orbit_of_site[i]] = group

    lines = [
        f"Wyckoff sites for {structure.composition.formula}",
        f"{'label':<15}{'species':<30}{'sites in cell':>14}   {'disordered?':<13}ordered configuration",
    ]
    for k, ((indices, _), label) in enumerate(zip(orbits, labels)):
        group = group_of_orbit.get(k)
        if group is None:
            config = "-"
        else:
            config = ", ".join(f"{n} {'vacancy' if sp is None else sp}" for sp, n in group.counts.items())
            pooled = sorted({orbit_of_site[i] for i in group.site_indices} - {k})
            if pooled:
                config += f" (pooled with {', '.join(labels[j] for j in pooled)})"
        disordered = "no" if group is None else "yes"
        lines.append(f"{label:<15}{structure[indices[0]].species_string:<30}{len(indices):>14}   {disordered:<13}{config}")
    return "\n".join(lines)


def symmetrically_equivalent(
    structure1: Structure,
    structure2: Structure,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
    primitive_cell: bool = True,
    scale: bool = True,
    attempt_supercell: bool = False,
) -> bool:
    """Determine whether two ordered structures should be considered
    symmetrically equivalent.

    Two structures are considered equivalent here if
    :meth:`pymatgen.analysis.structure_matcher.StructureMatcher.fit` finds a
    species-preserving mapping between them (up to reduction to a common
    primitive cell when ``primitive_cell=True``, an affine lattice fit within
    ``ltol``/``angle_tol``, and a site-distance match within ``stol``, in
    units of the average free length per atom). This correctly accounts for
    periodicity (both structures are treated as infinite periodic lattices,
    not just their input unit cells) and for the crystal symmetry implicit
    in that periodicity, but it does *not* independently apply any external
    symmetry operations -- it is a numerical/topological equivalence test on
    the two periodic point sets, not a check against a specific parent
    structure's space group. Species are compared exactly (including
    oxidation state, if present); strip oxidation states first if chemical
    equivalence regardless of oxidation state is desired.

    Use :func:`configuration_degeneracy` instead when equivalence must be
    defined specifically with respect to a disordered parent's symmetry
    operations.

    Args:
        structure1, structure2: Ordered structures to compare.
        ltol, stol, angle_tol, primitive_cell, scale, attempt_supercell: See
            :class:`pymatgen.analysis.structure_matcher.StructureMatcher`.

    Returns:
        Whether the structures are symmetrically/periodically equivalent.
    """
    matcher = StructureMatcher(
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        primitive_cell=primitive_cell,
        scale=scale,
        attempt_supercell=attempt_supercell,
    )
    return matcher.fit(structure1, structure2)


def _nearest_site_indices(
    query_frac_coords: np.ndarray, reference_frac_coords: np.ndarray, lattice
) -> tuple[np.ndarray, float]:
    """For each row of ``query_frac_coords``, index of the nearest row of
    ``reference_frac_coords`` under minimum-image distance.

    Returns the index array together with the worst (maximum) distance
    (Angstrom) between a query point and its assigned nearest reference
    point, so callers can validate the match.
    """
    dists = lattice.get_all_distances(query_frac_coords, reference_frac_coords)
    idx = dists.argmin(axis=1)
    worst = dists[np.arange(len(query_frac_coords)), idx].max() if len(query_frac_coords) else 0.0
    return idx, worst


def site_permutations(parent: Structure, symprec: float = 1e-3, match_tol: float | None = None) -> list[tuple[int, ...]]:
    """Site-index permutations induced by the symmetry operations of ``parent``.

    Each symmetry operation of ``parent`` (as found by
    :class:`~pymatgen.symmetry.analyzer.SpacegroupAnalyzer`) maps every site
    of ``parent`` onto another (or the same) site of ``parent``. This
    function returns those site-index permutations, deduplicated, as a
    sorted list of tuples ``p`` with ``p[j] == k`` meaning "the operation
    maps site ``j`` onto site ``k``".

    These permutations form the symmetry group used by
    :func:`configuration_degeneracy` to enumerate the orbit of an ordering.

    Args:
        parent: Structure whose symmetry operations define the permutations.
            This is normally the disordered parent structure (or a version
            of it with all disordered sites carrying a common dummy
            species), so that all symmetry-allowed site permutations are
            captured, not just those preserving one particular decoration.
        symprec: Symmetry-finding tolerance (Angstrom), passed to spglib.
        match_tol: Maximum allowed distance (Angstrom) between a
            symmetry-mapped site and the parent site it is assigned to.
            Defaults to ``10 * symprec``.

    Returns:
        Sorted list of site-index permutations, each a tuple of length
        ``len(parent)``.
    """
    if match_tol is None:
        match_tol = 10 * symprec

    lattice = parent.lattice
    frac_coords = np.asarray(parent.frac_coords)
    n_sites = len(frac_coords)

    sga = SpacegroupAnalyzer(parent, symprec=symprec)
    operations = sga.get_symmetry_operations()

    perms: set[tuple[int, ...]] = set()
    for op in operations:
        mapped = op.operate_multi(frac_coords) % 1.0
        idx, worst = _nearest_site_indices(mapped, frac_coords, lattice)
        if worst > match_tol:
            raise ValueError(
                f"A symmetry operation of the parent moved a site {worst:.4g} A "
                f"from any parent site (tolerance {match_tol:.4g} A). Try "
                "raising symprec, or check that `parent` is the correct "
                "reference lattice."
            )
        if len(set(idx.tolist())) != n_sites:
            raise ValueError(
                "A symmetry operation of the parent did not induce a "
                "bijection on parent sites; the parent structure may not be "
                "consistent with its own detected symmetry."
            )
        perms.add(tuple(int(i) for i in idx))
    return sorted(perms)


def _site_label(site: PeriodicSite) -> str:
    """A hashable label capturing a site's species and occupancies, used to
    compare ordered/disordered decorations across the sites of a parent."""
    if site.is_ordered:
        return site.species_string
    return "|".join(
        f"{sp}:{occ:.6f}" for sp, occ in sorted(site.species.items(), key=lambda kv: str(kv[0]))
    )


def _occupations_on_parent(
    parent: Structure, ordered: Structure, tol: float = 0.5, vacancy_label: str = "__vacant__"
) -> tuple[str, ...]:
    """Label of each parent site as realized in ``ordered``, indexed by
    parent site. Sites of ``ordered`` are matched to parent sites by
    position (not by input order), so ``ordered`` may have fewer sites than
    ``parent`` (unmatched parent sites are labelled ``vacancy_label``)."""
    ref = np.asarray(parent.frac_coords)
    cand = np.asarray(ordered.frac_coords)
    if len(cand) > len(ref):
        raise ValueError(
            f"`ordered` has {len(cand)} sites, more than the parent's {len(ref)}."
        )

    labels = [vacancy_label] * len(ref)
    if len(cand):
        idx, worst = _nearest_site_indices(cand % 1.0, ref, parent.lattice)
        if worst > tol:
            raise ValueError(
                f"A site of `ordered` sits {worst:.3g} A from its nearest "
                f"parent site (tolerance {tol} A) -- `ordered` may not be a "
                "realization of this `parent` (wrong cell, origin, or "
                "structure)."
            )
        if len(set(idx.tolist())) != len(cand):
            raise ValueError(
                "Two sites of `ordered` matched the same parent site; "
                "`ordered` and `parent` are not on a common lattice."
            )
        for k, j in enumerate(idx):
            labels[j] = _site_label(ordered[k])
    return tuple(labels)


def configuration_degeneracy(
    parent: Structure,
    ordered: Structure,
    symprec: float = 1e-3,
    tol: float = 0.5,
    perms: list[tuple[int, ...]] | None = None,
) -> int:
    """Degeneracy of an ordered configuration with respect to its disordered
    parent's symmetry.

    The degeneracy is defined as the size of the orbit of ``ordered`` under
    the site permutations induced by the symmetry operations of ``parent``
    (see :func:`site_permutations`): the number of distinct site-decorations
    reachable by applying parent symmetry operations to ``ordered``. This is
    the multiplicity ``g`` such that a partition function over symmetrically
    distinct orderings can be written ``Z = sum_i g_i * exp(-beta * E_i)``.

    This is distinct from:

    * the space group / point symmetry of ``ordered`` itself (its
      *stabilizer* subgroup of the parent's symmetry group -- in fact
      ``degeneracy == |parent symmetry group| // |stabilizer of ordered|``);
    * whether two *different* orderings are periodically equivalent to each
      other (use :func:`symmetrically_equivalent` for that, comparing
      structures directly rather than via a shared parent).

    Args:
        parent: The disordered parent structure whose symmetry operations
            define the orbit. Sites of ``ordered`` are matched to sites of
            ``parent`` by position.
        ordered: The ordered configuration to compute the degeneracy of. May
            have fewer sites than ``parent`` (e.g. a partially vacant
            sublattice); unmatched parent sites are treated as vacant.
        symprec: Symmetry-finding tolerance (Angstrom) for `parent`, passed
            to spglib. Ignored if ``perms`` is given.
        tol: Maximum distance (Angstrom) a site of `ordered` may sit from
            its assigned parent site.
        perms: Precomputed site permutations from
            ``site_permutations(parent, symprec=symprec)``. Pass this when
            calling ``configuration_degeneracy`` in a loop over many
            orderings of the same parent, to avoid re-running symmetry
            analysis on `parent` every time.

    Returns:
        The orbit size (a positive integer).
    """
    if perms is None:
        perms = site_permutations(parent, symprec=symprec)
    labels = _occupations_on_parent(parent, ordered, tol=tol)
    orbit = {tuple(labels[p[j]] for j in range(len(labels))) for p in perms}
    return len(orbit)


def _fixed_decorations(perm: tuple[int, ...], group: DisorderedSiteGroup) -> int:
    """Number of decorations of ``group`` with its exact counts that are
    left unchanged by the site permutation ``perm``.

    A decoration is unchanged only if it is constant on every cycle of
    ``perm``, so this counts the ways to give each cycle one species such
    that every species' cycle lengths add up to its count.
    """
    members = set(group.site_indices)
    seen: set[int] = set()
    cycle_lengths = []
    for start in group.site_indices:
        if start in seen:
            continue
        length = 0
        site = start
        while site not in seen:
            if site not in members:
                raise ValueError(
                    "A parent symmetry operation maps a disordered site onto a site with "
                    "different species or occupancies; the permutations do not belong to this parent."
                )
            seen.add(site)
            site = perm[site]
            length += 1
        cycle_lengths.append(length)

    # ways[remaining counts] = number of ways to decorate the cycles so far.
    ways = {tuple(group.counts.values()): 1}
    for length in cycle_lengths:
        next_ways: dict[tuple[int, ...], int] = {}
        for remaining, n in ways.items():
            for k, left in enumerate(remaining):
                if left >= length:
                    key = remaining[:k] + (left - length,) + remaining[k + 1 :]
                    next_ways[key] = next_ways.get(key, 0) + n
        ways = next_ways
    return ways.get((0,) * len(group.counts), 0)


def count_distinct_configurations(
    parent: Structure,
    symprec: float = 1e-3,
    perms: list[tuple[int, ...]] | None = None,
    occupancy_tol: float = 1e-2,
) -> int:
    """Number of symmetrically distinct ordered configurations of a
    disordered parent in its cell.

    Counts the orbits of the parent's site permutations (see
    :func:`site_permutations`) acting on all ordered configurations with
    the whole-site counts of :func:`disorder_mc.structures.disordered_site_groups`,
    using Burnside's lemma::

        n_distinct = (1 / |G|) * sum over g in G of |Fix(g)|

    where ``Fix(g)`` is the set of configurations left unchanged by ``g``,
    counted exactly from the cycle structure of ``g`` without enumerating
    configurations. The orbits are the classes whose sizes
    :func:`configuration_degeneracy` returns, so the degeneracies of all
    distinct configurations sum to
    :func:`disorder_mc.structures.count_configurations`. When most
    configurations have no symmetry, the result is close to
    ``count_configurations(parent) / |G|``.

    Equivalence is with respect to the parent's symmetry in this cell only;
    configurations of different supercells are not compared.

    Args:
        parent: Disordered parent structure.
        symprec: Symmetry-finding tolerance (Angstrom), passed to spglib.
            Ignored if ``perms`` is given.
        perms: Precomputed ``site_permutations(parent, symprec=symprec)``.
        occupancy_tol: Passed to
            :func:`disorder_mc.structures.disordered_site_groups`.

    Returns:
        The exact number of symmetrically distinct configurations.
    """
    if perms is None:
        perms = site_permutations(parent, symprec=symprec)
    groups = disordered_site_groups(parent, occupancy_tol=occupancy_tol)

    total_fixed = 0
    for perm in perms:
        fixed = 1
        for group in groups:
            fixed *= _fixed_decorations(perm, group)
            if fixed == 0:
                break
        total_fixed += fixed

    n_distinct, remainder = divmod(total_fixed, len(perms))
    if remainder:
        raise ValueError(
            "Burnside count is not an integer; `perms` is not a complete symmetry group of `parent`."
        )
    return n_distinct
