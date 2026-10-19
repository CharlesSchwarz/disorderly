"""Exhaustive enumeration of symmetrically distinct ordered configurations
of a disordered parent structure, using icet's ``enumerate_structures``.

icet (and its ASE dependency) is optional and imported lazily, so the base
``disorderly`` package remains importable without it.

May depend on :mod:`disorderly.structures`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from pymatgen.core import Lattice, Structure

from disorderly.structures import DisorderedSiteGroup, _is_whole, disordered_site_groups

__all__ = ["enumerate_ordered_configurations"]


def _missing_icet_dependency_error(exc: ModuleNotFoundError) -> ModuleNotFoundError:
    return ModuleNotFoundError(
        f"Missing optional dependency {exc.name!r}; install disorderly with the "
        "icet extra or install icet in this environment."
    )


def _normalize_sizes(sizes: int | Iterable[int]) -> list[int]:
    sizes = list(sizes) if isinstance(sizes, Iterable) else [sizes]
    if not sizes:
        raise ValueError("`sizes` must contain at least one supercell size.")
    for n in sizes:
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"Supercell sizes must be positive integers; got {n!r}.")
    return sorted(set(sizes))


def _counts_at_size(group: DisorderedSiteGroup, n: int, count_tol: float) -> dict | None:
    """Whole-site counts of ``group`` in an ``n``-fold supercell, or ``None``
    if the occupancies are not whole numbers of sites at that size."""
    if not all(_is_whole(n * x, count_tol) for x in group.ideal_counts.values()):
        return None
    return {sp: round(n * x) for sp, x in group.ideal_counts.items()}


def enumerate_ordered_configurations(
    parent: Structure,
    sizes: int | Iterable[int] = 1,
    symprec: float = 1e-3,
    position_tolerance: float | None = None,
    occupancy_tol: float = 1e-2,
    count_tol: float = 0.05,
    niggli_reduce: bool = False,
) -> Iterator[Structure]:
    """Yield every symmetrically distinct ordered configuration of a
    disordered parent, enumerated with icet's ``enumerate_structures``.

    icet implements the Hart--Forcade algorithm (Phys. Rev. B 77, 224115
    (2008); 80, 014120 (2009)): for each supercell size it enumerates the
    symmetrically inequivalent supercell shapes of the parent lattice and
    the inequivalent decorations of each, removing duplicates under the
    parent's space-group operations and supercell translations.

    Allowed decorations and composition follow the rest of the package:

    * every disordered site group of
      :func:`disorderly.structures.disordered_site_groups` (sites sharing
      one species/occupancy signature) holds exactly ``size * n_sites *
      occupancy`` of each species, so composition is fixed per group --
      the same constraint :func:`disorderly.sampling.random_ordered_configuration`
      and composition-preserving Monte Carlo moves obey;
    * a group whose occupancies sum to less than 1 also holds the matching
      number of vacancies, which are left out of the yielded structures
      (a vacant parent site is simply absent);
    * ordered parent sites keep their species.

    Symmetry is that of the disordered parent, with sites sharing a
    signature treated as identical (as in
    :func:`disorderly.symmetry.site_permutations`). For ``sizes=1`` the
    configurations are therefore one representative of each orbit counted
    by :func:`disorderly.symmetry.count_distinct_configurations`, and
    their :func:`~disorderly.symmetry.configuration_degeneracy` values sum
    to :func:`disorderly.structures.count_configurations`.

    Supercell sizes: size ``n`` yields configurations in supercells with
    ``n`` times as many sites as ``parent``, across all inequivalent
    supercell shapes, and only those whose decoration does not repeat with
    a smaller supercell of the parent lattice (those appear at the smaller
    size instead). Every distinct ordering therefore appears exactly once
    across ``sizes=range(1, n_max + 1)``. To enumerate *all* orderings of
    one particular supercell, build it with ``parent.make_supercell(...)``
    and enumerate it with ``sizes=1``. Sizes at which the occupancies are
    not whole numbers of sites (see
    :func:`disorderly.structures.supercell_multiplier`) are skipped.

    Enumeration is exhaustive and the number of configurations grows
    combinatorially; check ``count_distinct_configurations(parent)`` first
    for large cells. Configurations are generated lazily.

    Args:
        parent: Disordered parent structure. Not modified.
        sizes: Supercell size, or sizes, in multiples of the parent cell.
        symprec: Symmetry-finding tolerance (Angstrom) passed to spglib.
        position_tolerance: Tolerance used by icet to map symmetry-
            transformed sites onto parent sites, in fractional
            coordinates. Defaults to ``symprec``.
        occupancy_tol: Passed to
            :func:`disorderly.structures.disordered_site_groups`.
        count_tol: Absolute tolerance (in sites) for treating
            ``size * n_sites * occupancy`` as a whole number.
        niggli_reduce: If true, Niggli-reduce each supercell. The default
            ``False`` keeps the cell ``H @ parent.lattice.matrix`` for
            supercell matrix ``H``, so ``sizes=1`` configurations share
            the parent's lattice and can be passed to
            :func:`disorderly.ordering.ordering_key` and
            :func:`~disorderly.symmetry.configuration_degeneracy`.

    Yields:
        Ordered ``Structure`` objects carrying the parent's species
        (including oxidation states). Site properties are not copied.

    Raises:
        ValueError: If ``sizes`` is empty or not all positive integers, or
            the occupancies are not whole numbers of sites at any
            requested size.
        ModuleNotFoundError: If icet is not installed.
    """
    # Validate eagerly, so errors are raised at call time rather than on the
    # first ``next()`` of the generator.
    size_list = _normalize_sizes(sizes)
    groups = disordered_site_groups(parent, occupancy_tol=occupancy_tol, count_tol=count_tol, warn=False)
    counts_by_size = {}
    for n in size_list:
        counts = [_counts_at_size(group, n, count_tol) for group in groups]
        if all(c is not None for c in counts):
            counts_by_size[n] = counts
    if not counts_by_size:
        raise ValueError(
            f"The parent's occupancies are not whole numbers of sites at any requested size {size_list}; "
            "see disorderly.structures.supercell_multiplier for the smallest size that works."
        )

    try:
        from ase import Atoms
        from ase.data import chemical_symbols as ase_symbols
        from icet.tools import enumerate_structures
    except ModuleNotFoundError as exc:
        raise _missing_icet_dependency_error(exc) from exc

    return _enumerate(
        parent, groups, counts_by_size, Atoms, ase_symbols, enumerate_structures,
        symprec=symprec, position_tolerance=position_tolerance, niggli_reduce=niggli_reduce,
    )


def _enumerate(
    parent, groups, counts_by_size, Atoms, ase_symbols, enumerate_structures,
    *, symprec, position_tolerance, niggli_reduce,
) -> Iterator[Structure]:
    # icet works with ASE chemical symbols, which cannot carry oxidation
    # states or distinguish the same species on different site groups. Each
    # (group, species) choice therefore gets its own placeholder symbol;
    # this also makes icet's global concentration restriction on that
    # placeholder an exact per-group count.
    placeholders = iter(ase_symbols[1:])
    species_of: dict[str, object] = {}  # placeholder -> pymatgen species, or None for a vacancy

    def new_placeholder(species) -> str:
        try:
            symbol = next(placeholders)
        except StopIteration:
            raise ValueError("Too many distinct species/site-group combinations to label for icet.") from None
        species_of[symbol] = species
        return symbol

    allowed: list[list[str] | None] = [None] * len(parent)
    group_placeholders: list[dict] = []
    for group in groups:
        symbols = {sp: new_placeholder(sp) for sp in group.counts}
        group_placeholders.append(symbols)
        for i in group.site_indices:
            allowed[i] = list(symbols.values())
    fixed: dict = {}
    for i, site in enumerate(parent):
        if site.is_ordered:
            if site.specie not in fixed:
                fixed[site.specie] = new_placeholder(site.specie)
            allowed[i] = [fixed[site.specie]]

    # Site types for spglib: one per group and one per fixed species, so the
    # detected symmetry is that of the disordered parent.
    site_type = [0] * len(parent)
    for k, group in enumerate(groups):
        for i in group.site_indices:
            site_type[i] = k + 1
    fixed_types = {sp: len(groups) + 1 + k for k, sp in enumerate(fixed)}
    for i, site in enumerate(parent):
        if site.is_ordered:
            site_type[i] = fixed_types[site.specie]

    pbc = tuple(bool(p) for p in parent.lattice.pbc)
    prim = Atoms(
        numbers=site_type,
        scaled_positions=parent.frac_coords,
        cell=parent.lattice.matrix,
        pbc=pbc,
    )

    n_parent_sites = len(parent)
    for n, counts in counts_by_size.items():
        n_sites = n * n_parent_sites
        concentrations = {
            symbols[sp]: (count / n_sites, count / n_sites)
            for symbols, group_counts in zip(group_placeholders, counts)
            for sp, count in group_counts.items()
        }
        for atoms in enumerate_structures(
            prim,
            sizes=[n],
            chemical_symbols=allowed,
            concentration_restrictions=concentrations or None,
            niggli_reduce=niggli_reduce,
            symprec=symprec,
            position_tolerance=position_tolerance,
        ):
            kept = [(species_of[s], pos) for s, pos in zip(atoms.get_chemical_symbols(), atoms.positions)]
            kept = [(sp, pos) for sp, pos in kept if sp is not None]
            yield Structure(
                Lattice(atoms.cell[:], pbc=pbc),
                [sp for sp, _ in kept],
                [pos for _, pos in kept],
                coords_are_cartesian=True,
            )
