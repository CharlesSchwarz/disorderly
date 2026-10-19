"""Reversible, compact keys for orderings of a disordered parent.

The code produced here is intentionally parent-relative: it records which
allowed species occupies each disordered parent site, while the parent
structure supplies the lattice, coordinates, ordered sites, and species
vocabulary needed to reconstruct the ordered configuration.
"""

from __future__ import annotations

import base64
from typing import Final

import numpy as np
from pymatgen.core import PeriodicSite, Structure

__all__ = ["ordering_key", "structure_from_ordering_key"]

_PREFIX: Final = "ok1_"
_LEGACY_PREFIX: Final = "oh1_"
_VACANCY: Final = None


def _ordered_options(site, occupancy_tol: float) -> list:
    options = [sp for sp, _ in sorted(site.species.items(), key=lambda kv: str(kv[0]))]
    total_occ = sum(float(occ) for occ in site.species.values())
    if total_occ < 1.0 - occupancy_tol:
        options.append(_VACANCY)
    return options


def _disordered_indices_and_options(parent: Structure, occupancy_tol: float) -> tuple[list[int], list[list]]:
    indices: list[int] = []
    options: list[list] = []
    for i, site in enumerate(parent):
        if not site.is_ordered:
            site_options = _ordered_options(site, occupancy_tol)
            if not site_options:
                raise ValueError(f"Disordered parent site {i} has no allowed species.")
            indices.append(i)
            options.append(site_options)
    return indices, options


def _encode_digits(digits: list[int], radices: list[int]) -> str:
    value = 0
    multiplier = 1
    for digit, radix in zip(digits, radices):
        if digit < 0 or digit >= radix:
            raise ValueError(f"Digit {digit} is outside radix {radix}.")
        value += digit * multiplier
        multiplier *= radix

    n_bytes = max(1, (value.bit_length() + 7) // 8)
    raw = value.to_bytes(n_bytes, byteorder="big")
    return _PREFIX + base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_digits(code: str, radices: list[int]) -> list[int]:
    if code.startswith(_PREFIX):
        prefix = _PREFIX
    elif code.startswith(_LEGACY_PREFIX):
        prefix = _LEGACY_PREFIX
    else:
        raise ValueError(f"Ordering key must start with {_PREFIX!r}.")

    payload = code[len(prefix) :]
    if not payload:
        raise ValueError("Ordering key has an empty payload.")

    padded = payload + "=" * (-len(payload) % 4)
    try:
        value = int.from_bytes(base64.urlsafe_b64decode(padded.encode("ascii")), byteorder="big")
    except Exception as exc:
        raise ValueError(f"Ordering key payload is not valid URL-safe base64: {payload!r}.") from exc

    digits: list[int] = []
    for radix in radices:
        digits.append(value % radix)
        value //= radix

    if value:
        raise ValueError("Ordering key contains more information than this parent structure can decode.")
    return digits


def _match_ordered_to_parent(parent: Structure, ordered: Structure, site_match_tol: float) -> dict[int, int]:
    if len(ordered) > len(parent):
        raise ValueError(f"Ordered structure has {len(ordered)} sites, more than parent has {len(parent)}.")
    if not ordered.is_ordered:
        raise ValueError("Ordered structure must not contain partially occupied sites.")
    if not np.allclose(parent.lattice.matrix, ordered.lattice.matrix, atol=site_match_tol):
        raise ValueError("Parent and ordered structures must have the same lattice to encode an ordering key.")

    if len(ordered) == 0:
        return {}

    distances = parent.lattice.get_all_distances(np.asarray(ordered.frac_coords) % 1.0, np.asarray(parent.frac_coords))
    parent_indices = distances.argmin(axis=1)
    worst = distances[np.arange(len(ordered)), parent_indices].max()
    if worst > site_match_tol:
        raise ValueError(
            f"A site in ordered structure is {worst:.4g} A from the nearest parent site "
            f"(tolerance {site_match_tol:.4g} A)."
        )
    if len(set(parent_indices.tolist())) != len(parent_indices):
        raise ValueError("Two ordered sites matched the same parent site.")

    return {int(parent_i): int(ordered_i) for ordered_i, parent_i in enumerate(parent_indices)}


def ordering_key(
    parent: Structure,
    ordered: Structure,
    *,
    occupancy_tol: float = 1e-2,
    site_match_tol: float = 1e-3,
) -> str:
    """Return a short, reversible key for ``ordered`` relative to ``parent``.

    The returned string is not a cryptographic digest. It is a compact
    parent-relative encoding of the species/vacancy choice on each
    disordered parent site. Given the same parent structure, recover the
    ordered configuration with :func:`structure_from_ordering_key`.

    Ordered sites in ``parent`` are not encoded because their identity is
    fixed by the parent. They are still validated against ``ordered``.

    Args:
        parent: Disordered parent structure defining the site set and allowed
            species on each disordered site.
        ordered: Ordered configuration on the parent's sites. It may have
            fewer sites than ``parent`` when partially occupied parent sites
            are vacant in the configuration.
        occupancy_tol: Tolerance for deciding whether a disordered parent
            site allows vacancy (total occupancy below 1).
        site_match_tol: Maximum Cartesian distance in angstrom for matching
            ordered sites back to parent sites.

    Returns:
        A compact string beginning with ``"ok1_"``.
    """
    disordered_indices, options_by_site = _disordered_indices_and_options(parent, occupancy_tol)
    options_by_parent = dict(zip(disordered_indices, options_by_site))
    ordered_by_parent = _match_ordered_to_parent(parent, ordered, site_match_tol)

    digits: list[int] = []
    for i, parent_site in enumerate(parent):
        ordered_i = ordered_by_parent.get(i)
        if parent_site.is_ordered:
            if ordered_i is None:
                raise ValueError(f"Ordered parent site {i} is missing from the ordered structure.")
            if ordered[ordered_i].species != parent_site.species:
                raise ValueError(
                    f"Ordered parent site {i} has species {ordered[ordered_i].species}, "
                    f"expected {parent_site.species}."
                )
            continue

        options = options_by_parent[i]
        if ordered_i is None:
            if _VACANCY not in options:
                raise ValueError(f"Fully occupied disordered parent site {i} is missing from ordered structure.")
            digits.append(options.index(_VACANCY))
            continue

        species = ordered[ordered_i].specie
        if species not in options:
            raise ValueError(f"Species {species} at parent site {i} is not allowed by the parent structure.")
        digits.append(options.index(species))

    return _encode_digits(digits, [len(options) for options in options_by_site])


def structure_from_ordering_key(
    parent: Structure,
    code: str,
    *,
    occupancy_tol: float = 1e-2,
) -> Structure:
    """Recover an ordered configuration from ``parent`` and an ordering key.

    Args:
        parent: The same parent structure used to create ``code``.
        code: A string returned by :func:`ordering_key`.
        occupancy_tol: Tolerance for deciding whether a disordered parent
            site allows vacancy. Use the same value used during encoding.

    Returns:
        A new ordered ``pymatgen.Structure``. Vacant disordered parent sites
        are omitted, matching the package's ordered-configuration convention.
    """
    disordered_indices, options_by_site = _disordered_indices_and_options(parent, occupancy_tol)
    radices = [len(options) for options in options_by_site]
    digits = _decode_digits(code, radices)
    choice_by_parent = dict(zip(disordered_indices, digits))
    options_by_parent = dict(zip(disordered_indices, options_by_site))

    sites: list[PeriodicSite] = []
    for i, parent_site in enumerate(parent):
        if parent_site.is_ordered:
            sites.append(parent_site)
            continue

        options = options_by_parent[i]
        choice = options[choice_by_parent[i]]
        if choice is None:
            continue
        sites.append(
            PeriodicSite(
                choice,
                parent_site.frac_coords,
                parent.lattice,
                to_unit_cell=False,
                coords_are_cartesian=False,
                properties=parent_site.properties,
            )
        )

    return Structure.from_sites(sites, to_unit_cell=False)
