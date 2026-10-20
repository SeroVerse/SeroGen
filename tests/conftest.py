"""Shared pytest fixtures for SeroGen genome assemblies and predictors."""

from pathlib import Path

import pytest

from serogen.core.genome import GenomeAssembly
from serogen.genes import OrfPredictor

DATA_DIR: Path = Path(__file__).resolve().parent / "data"


# ----------------------------------------------------------------------------------------------------------------------
# Raw Genome Assembly Path Fixtures
# ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def pao1_path() -> Path:
    """Path to the zstd-compressed *Pseudomonas aeruginosa* PAO1 genome assembly."""
    return DATA_DIR / "PAO1.fna.zst"


@pytest.fixture(scope="session")
def sgh10_path() -> Path:
    """Path to the zstd-compressed *Klebsiella pneumoniae* SGH10 chromosome assembly."""
    return DATA_DIR / "SGH10_chromosome.fna.zst"


@pytest.fixture(scope="session")
def stmc125_path() -> Path:
    """Path to the zstd-compressed *Salmonella enterica* STMC125 genome assembly."""
    return DATA_DIR / "STMC125.fna.zst"


@pytest.fixture(
    scope="session",
    params=("pao1_path", "sgh10_path", "stmc125_path"),
    ids=("PAO1", "SGH10_chromosome", "STMC125"),
)
def assembly_path(request: pytest.FixtureRequest) -> Path:
    """Parametrized fixture yielding each test genome assembly file path."""
    path: Path = request.getfixturevalue(request.param)
    return path


# ----------------------------------------------------------------------------------------------------------------------
# Parsed GenomeAssembly Fixtures (Derived from Path Fixtures)
# ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def pao1_assembly(pao1_path: Path) -> GenomeAssembly:
    """Parsed `GenomeAssembly` instance for *P. aeruginosa* PAO1."""
    return GenomeAssembly.from_file(pao1_path)


@pytest.fixture(scope="session")
def sgh10_assembly(sgh10_path: Path) -> GenomeAssembly:
    """Parsed `GenomeAssembly` instance for *K. pneumoniae* SGH10 chromosome."""
    return GenomeAssembly.from_file(sgh10_path)


@pytest.fixture(scope="session")
def stmc125_assembly(stmc125_path: Path) -> GenomeAssembly:
    """Parsed `GenomeAssembly` instance for *S. enterica* STMC125."""
    return GenomeAssembly.from_file(stmc125_path)


@pytest.fixture(
    scope="session",
    params=("pao1_assembly", "sgh10_assembly", "stmc125_assembly"),
    ids=("PAO1", "SGH10_chromosome", "STMC125"),
)
def genome_assembly(request: pytest.FixtureRequest) -> GenomeAssembly:
    """Parametrized fixture yielding each parsed `GenomeAssembly` instance."""
    assembly: GenomeAssembly = request.getfixturevalue(request.param)
    return assembly


# ----------------------------------------------------------------------------------------------------------------------
# ORF Predictor Fixtures
# ----------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def orf_predictor() -> OrfPredictor:
    """Session-scoped whole-genome `OrfPredictor` instance."""
    return OrfPredictor(whole_genome=True)
