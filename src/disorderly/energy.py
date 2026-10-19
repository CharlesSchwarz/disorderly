"""Electrostatic (Ewald) energy calculations for ordered and disordered
structures.

May depend on :mod:`disorderly.structures` but not on Monte Carlo
functionality.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pymatgen.core import DummySpecies, Structure

try:
    from pymatgen.core.ewald import EwaldSummation, compute_average_oxidation_state
except ModuleNotFoundError:  # pragma: no cover - depends on installed pymatgen version
    from pymatgen.analysis.ewald import EwaldSummation, compute_average_oxidation_state

__all__ = ["disordered_ewald_energy", "effective_site_charges", "PrecomputedEwald"]


def effective_site_charges(structure: Structure) -> list[float]:
    """The effective charge of each site of ``structure``.

    For a fully occupied, ordered site this is simply that species'
    oxidation state. For a partially occupied (disordered) site this is the
    occupancy-weighted average oxidation state of the species present::

        q_site = sum(occupancy_i * charge_i for each species i on the site)

    This is the effective charge used by :func:`disordered_ewald_energy`
    (delegated to pymatgen's ``compute_average_oxidation_state``, which
    implements exactly this formula).

    Args:
        structure: Structure whose sites carry oxidation-state-decorated
            species (e.g. via ``Structure.add_oxidation_state_by_element``)
            or an explicit ``"charge"`` site property.

    Returns:
        List of effective charges, one per site, in the same order as
        ``structure``'s sites.

    Raises:
        ValueError: If a site has neither oxidation-state-decorated species
            nor an explicit charge (see :func:`disordered_ewald_energy` for
            how to fix this).
    """
    return [_effective_charge(site) for site in structure]


def _effective_charge(site) -> float:
    try:
        return compute_average_oxidation_state(site)
    except ValueError as exc:
        raise ValueError(
            f"Site {site} has no oxidation-state information ({exc}). "
            "This package does not guess oxidation states; "
            "decorate the structure explicitly first, e.g. via "
            "`Structure.add_oxidation_state_by_element(...)`, "
            "`Structure.add_oxidation_state_by_guess()` (an explicit, "
            "opt-in guess), or by setting a `charge` site property."
        ) from exc


def disordered_ewald_energy(
    structure: Structure,
    real_space_cut: float | None = None,
    recip_space_cut: float | None = None,
    eta: float | None = None,
    acc_factor: float = 12.0,
) -> float:
    """Approximate electrostatic (Ewald) energy of a structure, ordered or
    disordered.

    For a partially occupied site, the site is assigned an effective charge
    equal to the occupancy-weighted average charge of the species on that
    site (see :func:`effective_site_charges`), and the Ewald sum is
    evaluated using these effective site charges. This is the standard
    "average/virtual-site" approximation for disordered electrostatics: it
    captures the mean-field electrostatic effect of the disordered
    occupancy but does not resolve the internal charge fluctuations within
    a site (e.g. correlations between which fraction of a mixed-occupancy
    site is charged versus not).

    ``structure`` must already carry explicit oxidation states (e.g. via
    ``Structure.add_oxidation_state_by_element``) or explicit site charges;
    this function never guesses oxidation states.

    Args:
        structure: Ordered or disordered structure with oxidation-state
            (or site-charge) information on every site.
        real_space_cut, recip_space_cut, eta, acc_factor: Passed through to
            pymatgen's ``EwaldSummation``; see that class for details.
            Defaults reproduce pymatgen's automatic convergence parameters.

    Returns:
        Total electrostatic energy in eV.

    Raises:
        ValueError: If any site lacks oxidation-state or charge information.
    """
    # Validate up front (with a clearer, package-specific message) rather
    # than letting a missing-oxidation-state error surface deep inside
    # EwaldSummation's internals.
    effective_site_charges(structure)

    ewald = EwaldSummation(
        structure,
        real_space_cut=real_space_cut,
        recip_space_cut=recip_space_cut,
        eta=eta,
        acc_factor=acc_factor,
    )
    return float(ewald.total_energy)


class PrecomputedEwald:
    """Fast Ewald energies for configurations on a fixed set of site positions.

    The Ewald energy of point charges ``q`` on fixed positions is a quadratic
    form, ``E = q @ M @ q + c * sum(q)**2``, where ``M`` and ``c`` depend only
    on the lattice and positions. ``M`` (pymatgen's total energy matrix for
    unit charges) and ``c`` (the charged-cell term) are computed once here,
    so each evaluation is a matrix-vector product instead of a full Ewald
    summation. The result equals :func:`disordered_ewald_energy` with the same
    convergence parameters, up to rounding.

    This suits swap-based Monte Carlo, where sites exchange species but never
    move. Configuration sites are matched to ``reference`` positions by
    distance, so sites may be listed in any order; reference positions with
    no matching site (vacancies) carry zero charge.

    Args:
        reference: Structure whose lattice and site positions every evaluated
            configuration must share (typically the disordered parent).
            Its species are ignored.
        site_tol: Maximum distance (Angstrom) between a configuration site
            and the reference position it is matched to.
        **ewald_kwargs: Passed to pymatgen's ``EwaldSummation``
            (``real_space_cut``, ``recip_space_cut``, ``eta``,
            ``acc_factor``).
    """

    def __init__(self, reference: Structure, site_tol: float = 0.5, **ewald_kwargs: Any) -> None:
        self.lattice = reference.lattice
        self.frac_coords = np.array(reference.frac_coords)
        self.site_tol = site_tol
        n_sites = len(reference)
        unit = Structure(self.lattice, [DummySpecies("X", oxidation_state=1)] * n_sites, self.frac_coords)
        ewald = EwaldSummation(unit, **ewald_kwargs)
        self.matrix = np.array(ewald.total_energy_matrix)
        self.charged_cell_coefficient = (ewald.total_energy - self.matrix.sum()) / n_sites**2
        self._cached_key: bytes | None = None
        self._cached_indices: np.ndarray | None = None

    def reference_indices(self, structure: Structure) -> np.ndarray:
        """Index of the reference position matched by each site of ``structure``.

        Raises:
            ValueError: If ``structure`` is on a different lattice, a site is
                farther than ``site_tol`` from every reference position, or
                two sites match the same position.
        """
        if not np.allclose(structure.lattice.matrix, self.lattice.matrix, atol=1e-6):
            raise ValueError("`structure` is not on the reference lattice; PrecomputedEwald requires fixed positions.")
        frac_coords = np.array(structure.frac_coords)
        key = frac_coords.tobytes()
        if key == self._cached_key:
            return self._cached_indices
        dists = self.lattice.get_all_distances(frac_coords, self.frac_coords)
        indices = dists.argmin(axis=1)
        worst = dists[np.arange(len(indices)), indices].max() if len(indices) else 0.0
        if worst > self.site_tol:
            raise ValueError(
                f"A site sits {worst:.3g} A from the nearest reference position (tolerance {self.site_tol} A)."
            )
        if len(set(indices.tolist())) != len(indices):
            raise ValueError("Two sites matched the same reference position.")
        self._cached_key, self._cached_indices = key, indices
        return indices

    def energy(self, structure: Structure) -> float:
        """Ewald energy of ``structure`` in eV (see :func:`disordered_ewald_energy`)."""
        charges = np.zeros(len(self.frac_coords))
        charges[self.reference_indices(structure)] = effective_site_charges(structure)
        total = charges.sum()
        return float(charges @ self.matrix @ charges + self.charged_cell_coefficient * total**2)

    def evaluate_props(self, configuration: Structure, context: dict[str, Any] | None = None) -> dict[str, float]:
        """Monte-Carlo-compatible properties (``{"energy": ...}``)."""
        return {"energy": self.energy(configuration)}

    def __call__(self, structure: Structure) -> float:
        """Evaluate ``structure`` as a plain sampling energy callable."""
        return self.energy(structure)
