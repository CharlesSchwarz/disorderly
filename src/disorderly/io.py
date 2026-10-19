"""Saving and loading structures, trajectories, saved configurations, and
run metadata.

May depend on :mod:`disorderly.results` and :mod:`disorderly.settings`.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from monty.json import MontyEncoder
from pymatgen.core import Structure

from disorderly.ordering import structure_from_ordering_key
from disorderly.results import MonteCarloCheckpoint, MonteCarloTrajectory, SavedConfiguration
from disorderly.settings import MonteCarloPolicy

__all__ = [
    "save_structures",
    "save_trajectory",
    "load_trajectory",
    "save_checkpoint",
    "load_checkpoint",
    "save_configurations",
    "load_configurations",
    "save_json",
    "load_json",
    "load_energy_evaluator",
    "load_policy",
]


def save_structures(
    structures: Sequence[Structure],
    directory: str | Path,
    prefix: str = "structure",
    fmt: str = "cif",
) -> list[Path]:
    """Save several structures as sequentially numbered files.

    Args:
        structures: Structures to save, in order.
        directory: Directory to write into (created if necessary).
        prefix: Filename prefix; files are named ``{prefix}_{i}.{fmt}``.
        fmt: File extension/format, passed to ``Structure.to``.

    Returns:
        The list of paths written, in the same order as ``structures``.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, structure in enumerate(structures):
        path = directory / f"{prefix}_{i}.{fmt}"
        structure.to(filename=str(path))
        paths.append(path)
    return paths


def save_json(obj: Any, path: str | Path) -> Path:
    """Save ``obj`` as JSON, using :class:`monty.json.MontyEncoder` to
    gracefully handle numpy scalars/arrays and MSONable objects that were
    not already converted to plain dicts (e.g. inside a user-supplied
    ``properties`` or ``metadata`` dict)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, cls=MontyEncoder, indent=2))
    return path


def load_json(path: str | Path) -> Any:
    """Load plain JSON from ``path``.

    Deliberately does *not* use :class:`monty.json.MontyDecoder`: this
    package's own ``as_dict``/``from_dict`` methods (e.g.
    :meth:`~disorderly.results.MonteCarloTrajectory.from_dict`) already
    know how to reconstruct their nested ``Structure`` objects explicitly,
    and auto-decoding here would pre-convert those nested dicts into
    ``Structure`` objects before ``from_dict`` gets to see them.
    """
    return json.loads(Path(path).read_text())


def save_trajectory(trajectory: MonteCarloTrajectory, path: str | Path) -> Path:
    """Save a Monte Carlo trajectory to a JSON file."""
    return save_json(trajectory.as_dict(), path)


def load_trajectory(path: str | Path) -> MonteCarloTrajectory:
    """Load a Monte Carlo trajectory previously saved with
    :func:`save_trajectory`."""
    return MonteCarloTrajectory.from_dict(load_json(path))


def save_checkpoint(checkpoint: MonteCarloCheckpoint, path: str | Path) -> Path:
    """Save a Monte Carlo checkpoint to a JSON file, atomically.

    The file is written to a temporary name and then renamed over ``path``,
    so a job killed mid-write leaves the previous checkpoint intact.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    save_json(checkpoint.as_dict(), tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path) -> MonteCarloCheckpoint:
    """Load a Monte Carlo checkpoint previously saved with
    :func:`save_checkpoint`."""
    return MonteCarloCheckpoint.from_dict(load_json(path))


def save_configurations(
    configurations: Sequence[SavedConfiguration],
    path: str | Path,
    *,
    include_structures: bool = False,
) -> Path:
    """Save a list of configurations (e.g. from
    :func:`disorderly.sampling.random_ewald_sampling`) to a JSON file.

    By default the JSON stores the reversible ``ordering_key`` but does not
    embed full structure site data. Pass ``include_structures=True`` when a
    self-contained JSON file is more important than compact output.
    """
    return save_json([c.as_dict(include_structure=include_structures) for c in configurations], path)


def load_configurations(
    path: str | Path,
    *,
    parent: Structure | str | Path | None = None,
    occupancy_tol: float = 1e-2,
) -> list[SavedConfiguration]:
    """Load configurations previously saved with
    :func:`save_configurations`.

    Compact configuration JSON files do not embed full structures. To load
    those records as :class:`~disorderly.results.SavedConfiguration` objects,
    pass the same parent structure used to generate the ``ordering_key`` values.
    Older/self-contained JSON files containing ``"structure"`` still load
    without a parent.
    """
    parent_structure = Structure.from_file(parent) if isinstance(parent, (str, Path)) else parent

    configurations = []
    for item in load_json(path):
        if item.get("structure") is not None:
            configurations.append(SavedConfiguration.from_dict(item))
            continue

        key = item.get("ordering_key", item.get("ordering_hash"))
        if key is None:
            raise ValueError("Configuration record has neither 'structure' nor 'ordering_key'.")
        if parent_structure is None:
            raise ValueError(
                "Configuration JSON does not contain full structures; pass parent=... "
                "to reconstruct structures from ordering_key values."
            )
        configurations.append(
            SavedConfiguration(
                step=item["step"],
                energy=item["energy"],
                structure=structure_from_ordering_key(parent_structure, key, occupancy_tol=occupancy_tol),
                properties=dict(item.get("properties") or {}),
                ordering_key=key,
            )
        )
    return configurations


def load_energy_evaluator(module_path: str, factory: str = "get_energy_evaluator", **kwargs: Any):
    """Load a custom random-sampling energy evaluator from a Python module.

    The factory should return a callable invoked as ``evaluator(structure)``.
    That callable may return either a scalar energy or a dict containing an
    ``"energy"`` key plus additional serializable properties.
    """
    module = importlib.import_module(module_path)
    try:
        factory_fn = getattr(module, factory)
    except AttributeError as exc:
        raise AttributeError(f"Module {module_path!r} has no attribute {factory!r}.") from exc
    if not callable(factory_fn):
        raise TypeError(f"{module_path}.{factory} is not callable.")
    evaluator = factory_fn(**kwargs)
    if not callable(evaluator):
        raise TypeError(f"{module_path}.{factory}(...) must return a callable, got {type(evaluator)!r}.")
    return evaluator


def load_policy(module_path: str, factory: str = "get_policy", **kwargs: Any) -> MonteCarloPolicy:
    """Load a :class:`~disorderly.settings.MonteCarloPolicy` from a Python
    module (a simulation's "settings file").

    A ``MonteCarloPolicy`` is fundamentally a bundle of Python callables
    (move selection, move validation, property evaluation), not data --
    there is no serialization format for arbitrary user logic. This
    package's interpretation of "load settings" is therefore: import a
    Python module by dotted path (which must be importable, e.g. on ``sys.path`` or
    installed), and call a factory function within it (by default
    ``get_policy``) to construct the policy. This keeps user ordering rules
    as ordinary, testable Python code rather than a restrictive
    configuration-file DSL.

    Args:
        module_path: Dotted path of an importable module defining the
            factory function (e.g. ``"my_project.mc_settings"``).
        factory: Name of the zero-or-more-argument callable in that module
            returning a ``MonteCarloPolicy``. Defaults to ``"get_policy"``.
        **kwargs: Forwarded to the factory function.

    Returns:
        The policy constructed by the factory function.

    Raises:
        ImportError: If ``module_path`` cannot be imported.
        AttributeError: If ``module_path`` has no attribute ``factory``.
        TypeError: If that attribute is not callable, or does not return a
            ``MonteCarloPolicy``.
    """
    module = importlib.import_module(module_path)
    try:
        factory_fn = getattr(module, factory)
    except AttributeError as exc:
        raise AttributeError(f"Module {module_path!r} has no attribute {factory!r}.") from exc
    if not callable(factory_fn):
        raise TypeError(f"{module_path}.{factory} is not callable.")
    policy = factory_fn(**kwargs)
    if not isinstance(policy, MonteCarloPolicy):
        raise TypeError(
            f"{module_path}.{factory}(...) must return a MonteCarloPolicy, got {type(policy)!r}."
        )
    return policy
