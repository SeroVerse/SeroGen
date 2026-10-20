"""First-class Structure-of-Arrays reference database container (`SerotypeDatabase`) for locus serotyping."""

from collections.abc import Iterable, Mapping
from collections.abc import Sequence as PySequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self, cast, overload

import numpy as np
from scipy.sparse import csr_array

from serogen.core.collections import BatchedContainer, ScalarView
from serogen.core.genome import GenomeAssembly
from serogen.core.graph.align import Alignment
from serogen.core.seq.clustering import Clusterer
from serogen.core.seq.seq import Sequence, Sequences
from serogen.loci.archetypes import ArchetypeCategory, LocusArchetype
from serogen.loci.matcher import BENIGN_VARIANT_MASK, LocusMatcher, Matches, MatchType
from serogen.loci.models import GeneNode, Loci, Locus
from serogen.loci.predictor import LocusPredictor


def _to_str_array(values: PySequence[str | bytes] | np.ndarray | None, n: int, default: str = "") -> np.ndarray:
    """Normalize an optional sequence or NumPy array of strings/bytes into a 1D unicode `np.ndarray` of length `n`."""
    if n == 0:
        return np.empty(0, dtype="U1")
    if values is None:
        return np.full(n, default, dtype=f"U{max(1, len(default))}")
    if isinstance(values, np.ndarray):
        if len(values) == 0:
            return np.full(n, default, dtype=f"U{max(1, len(default))}")
        if values.dtype.kind == "S":
            arr = np.char.decode(values, "utf-8", errors="replace")
        elif values.dtype.kind == "U":
            arr = values
        else:
            arr = values.astype(str)
    else:
        raw = [x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x) for x in values]
        if not raw:
            return np.full(n, default, dtype=f"U{max(1, len(default))}")
        arr = np.array(raw, dtype=str)
    if len(arr) != n:
        raise ValueError(f"Expected metadata array of length {n}, got {len(arr)}.")
    return arr


@dataclass(slots=True, frozen=True)
class SerotypeRecord(ScalarView):
    """Scalar view of a single reference serotype entry in a [`SerotypeDatabase`][serogen.loci.database.SerotypeDatabase]."""

    name: str
    serogroup: str
    accession: str
    strain: str
    locus: Locus

    @property
    def serotype(self) -> str:
        """Alias for `self.name` (the canonical serotype designation)."""
        return self.name

    @property
    def archetype(self) -> LocusArchetype:
        """Biosynthetic `LocusArchetype` of this reference locus."""
        return self.locus.archetype

    @property
    def num_genes(self) -> int:
        """Number of genes in this reference locus."""
        return self.locus.num_genes


@dataclass(slots=True, frozen=True)
class SerotypeDatabase(BatchedContainer[SerotypeRecord, "SerotypeDatabase"]):
    """First-class Structure-of-Arrays (SoA) reference database for biosynthetic locus serotyping.

    Consolidates reference [`Loci`][serogen.loci.models.Loci] (with packed protein homology `CLUSTER_ID`s),
    per-locus taxonomic/serogroup metadata (`names`, `serogroups`, `accessions`, `strains`), a pre-indexed
    [`LocusMatcher`][serogen.loci.matcher.LocusMatcher] (`RandstrobeIndex` over cluster centroids and target
    proteins), and a compressed consensus pan-locus synteny graph (`pan_adj`, `pan_nodes`, `pan_counts`).
    Supports zero-pickle, zero-compute `.npz` persistence via [`save`][serogen.loci.database.SerotypeDatabase.save]
    and [`load`][serogen.loci.database.SerotypeDatabase.load].
    """

    loci: Loci
    names: np.ndarray[tuple[int], np.dtype[np.str_]] = field(default_factory=lambda: np.empty(0, dtype="U1"))
    serogroups: np.ndarray[tuple[int], np.dtype[np.str_]] = field(default_factory=lambda: np.empty(0, dtype="U1"))
    accessions: np.ndarray[tuple[int], np.dtype[np.str_]] = field(default_factory=lambda: np.empty(0, dtype="U1"))
    strains: np.ndarray[tuple[int], np.dtype[np.str_]] = field(default_factory=lambda: np.empty(0, dtype="U1"))
    matcher: LocusMatcher = field(default_factory=lambda: LocusMatcher(Loci.empty()))
    pan_adj: csr_array = field(default_factory=lambda: csr_array((0, 0), dtype=np.uint32))
    pan_nodes: np.ndarray[tuple[int], np.dtype[np.uint32]] = field(default_factory=lambda: np.empty(0, dtype=np.uint32))
    pan_counts: np.ndarray[tuple[int], np.dtype[np.uint32]] = field(
        default_factory=lambda: np.empty(0, dtype=np.uint32)
    )

    def __post_init__(self) -> None:
        n = len(self.loci)
        resolved_names = _to_str_array(self.names, n) if len(self.names) > 0 else _to_str_array(self.loci.names, n)
        object.__setattr__(self, "names", resolved_names)
        object.__setattr__(self, "serogroups", _to_str_array(self.serogroups, n))
        object.__setattr__(self, "accessions", _to_str_array(self.accessions, n))
        object.__setattr__(self, "strains", _to_str_array(self.strains, n))
        if n > 0 and not np.array_equal(self.loci.names, resolved_names):
            object.__setattr__(self, "loci", self.loci.with_names(resolved_names))

    def __len__(self) -> int:
        return len(self.loci)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return bool(np.any(self.names == item) or np.any(self.accessions == item))
        if isinstance(item, SerotypeRecord):
            return bool(np.any(self.names == item.name))
        return False

    @property
    def serotypes(self) -> np.ndarray[tuple[int], np.dtype[np.str_]]:
        """1D array of serotype names (`self.names`)."""
        return self.names

    @property
    def total_genes(self) -> int:
        """Total number of genes across all reference loci in the database."""
        return self.loci.total_nodes

    @property
    def num_clusters(self) -> int:
        """Number of unique protein homology clusters (`CLUSTER_ID > 0`) in the database."""
        cids = self.loci.cluster_ids
        return int(len(np.unique(cids[cids > 0])))

    @property
    def novel_jaccard_threshold(self) -> float:
        """Jaccard threshold separating intra-serogroup variants from novel serogroups (`MatchType.NOVEL_LOCUS`)."""
        return self.matcher.novel_jaccard_threshold

    def infer_novel_threshold(self) -> float:
        """Infer the optimal `novel_jaccard_threshold` from this database's loci and serogroup metadata."""
        return self.matcher.infer_novel_threshold(serogroups=self.serogroups)

    @property
    def pan_graph(
        self,
    ) -> tuple[
        csr_array,
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
    ]:
        """Consensus pan-locus graph tuple `(pan_adj, pan_nodes, pan_counts)`."""
        return self.pan_adj, self.pan_nodes, self.pan_counts

    @classmethod
    def empty(cls) -> Self:
        """Create an empty 0-locus `SerotypeDatabase`."""
        empty_loci = Loci.empty()
        return cls(
            loci=empty_loci,
            names=np.empty(0, dtype="U1"),
            serogroups=np.empty(0, dtype="U1"),
            accessions=np.empty(0, dtype="U1"),
            strains=np.empty(0, dtype="U1"),
            matcher=LocusMatcher(empty_loci),
            pan_adj=csr_array((0, 0), dtype=np.uint32),
            pan_nodes=np.empty(0, dtype=np.uint32),
            pan_counts=np.empty(0, dtype=np.uint32),
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        """Concatenate multiple `SerotypeDatabase` instances and re-index their combined loci."""
        non_empty = [b for b in batches if len(b) > 0]
        if not non_empty:
            return cls.empty()
        if len(non_empty) == 1:
            return non_empty[0]
        first = non_empty[0]
        return cls.build(
            Loci.concat([b.loci for b in non_empty]),
            names=np.concatenate([b.names for b in non_empty]),
            serogroups=np.concatenate([b.serogroups for b in non_empty]),
            accessions=np.concatenate([b.accessions for b in non_empty]),
            strains=np.concatenate([b.strains for b in non_empty]),
            canonicalize_strands=False,
            min_identity=first.matcher.min_identity,
            min_coverage=first.matcher.min_coverage,
            min_bitscore=first.matcher.min_bitscore,
            cluster_mode=cast(Any, first.matcher.cluster_mode),
            inflation=first.matcher.inflation,
            novel_jaccard_threshold=first.matcher.novel_jaccard_threshold,
            min_score=first.matcher.min_score,
            k=first.matcher.k,
            s=first.matcher.s,
            aligner_k=first.matcher.aligner_k,
        )

    @classmethod
    def build(
        cls,
        sources: Loci | PySequence[Locus] | Sequences | PySequence[Sequence] | GenomeAssembly,
        *,
        names: PySequence[str | bytes] | np.ndarray | None = None,
        serogroups: PySequence[str | bytes] | np.ndarray | None = None,
        accessions: PySequence[str | bytes] | np.ndarray | None = None,
        strains: PySequence[str | bytes] | np.ndarray | None = None,
        predictor: LocusPredictor | None = None,
        categories: ArchetypeCategory | str | Iterable[ArchetypeCategory | str] | None = None,
        archetypes: LocusArchetype | str | int | Iterable[LocusArchetype | str | int] | None = None,
        canonicalize_strands: bool = True,
        heal_mges: bool = False,
        recluster: bool = False,
        cluster_mode: Literal["cdhit", "mcl", "greedy_cover", "upgma", "connected_components"] = "cdhit",
        min_identity: float = 80.0,
        cluster_identity: float | None = None,
        min_coverage: float = 0.80,
        min_bitscore: float = 0.0,
        inflation: float = 2.0,
        novel_jaccard_threshold: float | Literal["auto"] = 0.50,
        min_score: int = 2,
        k: int = 10,
        s: int = 5,
        aligner_k: int = 20,
    ) -> Self:
        """Build, cluster, and index a `SerotypeDatabase` from reference sequences or segmented `Loci`."""
        if cluster_identity is not None:
            min_identity = float(cluster_identity * 100.0 if cluster_identity <= 1.0 else cluster_identity)
        elif min_identity <= 1.0:
            min_identity = float(min_identity * 100.0)

        eff_min_identity = 25.0 if (cluster_mode == "mcl" and min_identity == 80.0) else float(min_identity)
        eff_min_bitscore = 40.0 if (cluster_mode == "mcl" and min_bitscore <= 0.0) else float(min_bitscore)

        inferred_acc: list[str] | None = None
        if isinstance(sources, Loci):
            loci_batch = sources
        elif isinstance(sources, (Sequences, GenomeAssembly)):
            seq_container = (
                sources
                if isinstance(sources, Sequences)
                else Sequences(
                    ids=sources.ids,
                    seqs=sources.seqs,
                    offsets=sources.offsets,
                    lengths=sources.lengths,
                )
            )
            if len(seq_container) == 0:
                return cls.empty()
            pred_engine = predictor if predictor is not None else LocusPredictor(whole_genome=False)
            segmented: list[Locus] = []
            inferred_acc = []
            for i in range(len(seq_container)):
                seq_i = seq_container[i]
                seq_id = seq_i.id.decode("utf-8", errors="replace").split()[0]
                inferred_acc.append(seq_id)
                pred_i = pred_engine.predict(seq_i, categories=categories, archetypes=archetypes)
                best_loc = pred_i.best()
                if best_loc is None and (categories is not None or archetypes is not None):
                    best_loc = pred_engine.predict(seq_i).best()
                if best_loc is None:
                    genes_i = pred_engine.orf_predictor.predict(seq_i)
                    roles_i = (
                        pred_engine.role_classifier.predict(genes_i.translations)
                        if len(genes_i) > 0
                        else np.empty(0, dtype=np.int32)
                    )
                    best_loc = Locus.from_genes(genes_i, roles_i, archetype=LocusArchetype.CPS_WZY)
                if canonicalize_strands:
                    best_loc = best_loc.canonicalize_strand()
                segmented.append(best_loc)
            loci_batch = Loci.from_list(segmented)
        else:
            src_list = list(sources)
            if not src_list:
                return cls.empty()
            if isinstance(src_list[0], Sequence):
                seq_batch = Sequences.from_records([s for s in src_list if isinstance(s, Sequence)])
                return cls.build(
                    seq_batch,
                    names=names,
                    serogroups=serogroups,
                    accessions=accessions,
                    strains=strains,
                    predictor=predictor,
                    categories=categories,
                    archetypes=archetypes,
                    canonicalize_strands=canonicalize_strands,
                    heal_mges=heal_mges,
                    recluster=recluster,
                    cluster_mode=cluster_mode,
                    min_identity=eff_min_identity,
                    min_coverage=min_coverage,
                    min_bitscore=eff_min_bitscore,
                    inflation=inflation,
                    novel_jaccard_threshold=novel_jaccard_threshold,
                    min_score=min_score,
                    k=k,
                    s=s,
                    aligner_k=aligner_k,
                )
            loci_batch = Loci.from_list([loc for loc in src_list if isinstance(loc, Locus)])

        n = len(loci_batch)
        if n == 0:
            return cls.empty()

        if heal_mges:
            loci_batch = loci_batch.heal_mobile_elements(remove_nodes=True)
        if canonicalize_strands:
            loci_batch = loci_batch.canonicalize_strands()

        resolved_names = (
            _to_str_array(names, n)
            if names is not None
            else (loci_batch.names if bool(np.any(loci_batch.names != "")) else _to_str_array(inferred_acc, n))
        )
        resolved_acc = _to_str_array(accessions, n) if accessions is not None else _to_str_array(inferred_acc, n)
        resolved_sg = _to_str_array(serogroups, n)
        resolved_strains = _to_str_array(strains, n)

        loci_batch = loci_batch.with_names(resolved_names)

        has_cids = loci_batch.total_nodes > 0 and bool(np.any(loci_batch.cluster_ids > 0))
        if recluster or not has_cids:
            clusterer = Clusterer(
                min_identity=eff_min_identity,
                min_coverage=min_coverage,
                min_bitscore=eff_min_bitscore,
                inflation=inflation,
                min_score=min_score,
                k=k,
                s=s,
                aligner_k=aligner_k,
                mode=cluster_mode,
            )
            loci_batch = loci_batch.cluster_genes(clusterer=clusterer)

        matcher = LocusMatcher(
            loci_batch,
            min_identity=eff_min_identity,
            min_coverage=min_coverage,
            min_bitscore=eff_min_bitscore,
            cluster_mode=cluster_mode,
            inflation=inflation,
            novel_jaccard_threshold=novel_jaccard_threshold,
            serogroups=resolved_sg,
            min_score=min_score,
            k=k,
            s=s,
            aligner_k=aligner_k,
        )
        pan_adj, pan_nodes, pan_counts = loci_batch.as_pan_graph(
            node_mask=GeneNode.mask("ROLE_ID", "CLUSTER_ID"),
            canonicalize_strands=False,
            heal_mges=False,
            cluster_genes=False,
        )
        return cls(
            loci=matcher.targets,
            names=resolved_names,
            serogroups=resolved_sg,
            accessions=resolved_acc,
            strains=resolved_strains,
            matcher=matcher,
            pan_adj=pan_adj,
            pan_nodes=pan_nodes,
            pan_counts=pan_counts,
        )

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        """Serialize all database metadata, `Loci`, `LocusMatcher` indices, and pan-graph into a flat dictionary."""
        out: dict[str, np.ndarray] = {
            "db_names": self.names,
            "db_serogroups": self.serogroups,
            "db_accessions": self.accessions,
            "db_strains": self.strains,
            "pan_data": np.asarray(self.pan_adj.data, dtype=np.uint32),
            "pan_indices": np.asarray(self.pan_adj.indices, dtype=np.int32),
            "pan_indptr": np.asarray(self.pan_adj.indptr, dtype=np.int32),
            "pan_shape": np.array(self.pan_adj.shape, dtype=np.int64),
            "pan_nodes": self.pan_nodes,
            "pan_counts": self.pan_counts,
        }
        out.update(self.loci.to_npz_dict(prefix="locus_"))
        out.update(self.matcher.to_npz_dict(prefix="matcher_", include_targets=False))
        return out

    @classmethod
    def from_npz_dict(cls, data: Mapping[str, Any]) -> Self:
        """Reconstruct a `SerotypeDatabase` from a `.npz` mapping with zero re-clustering or re-hashing."""
        loci = Loci.from_npz_dict(data, prefix="locus_")
        if len(loci) == 0:
            return cls.empty()
        matcher = LocusMatcher.from_npz_dict(data, targets=loci, prefix="matcher_")
        pan_shape = tuple(int(x) for x in data["pan_shape"])
        pan_adj = csr_array(
            (data["pan_data"], data["pan_indices"], data["pan_indptr"]),
            shape=pan_shape,
        )
        return cls(
            loci=loci,
            names=data["db_names"],
            serogroups=data["db_serogroups"],
            accessions=data["db_accessions"],
            strains=data["db_strains"],
            matcher=matcher,
            pan_adj=pan_adj,
            pan_nodes=data["pan_nodes"],
            pan_counts=data["pan_counts"],
        )

    def save(self, path: Path | str) -> Path:
        """Save this `SerotypeDatabase` to a compressed NumPy `.npz` archive (`allow_pickle=False`)."""
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".npz")
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **cast(Any, self.to_npz_dict()))
        return p

    @classmethod
    def load(cls, path: Path | str) -> Self:
        """Load a pre-indexed `SerotypeDatabase` from a compressed `.npz` archive (`allow_pickle=False`)."""
        p = Path(path)
        if not p.suffix and not p.exists():
            p = p.with_suffix(".npz")
        with np.load(p, allow_pickle=False) as data:
            return cls.from_npz_dict(data)

    def serogroup_of(self, name_or_index: str | int | np.integer[Any]) -> str:
        """Return the serogroup string for a serotype `name` or 0-based target index."""
        if isinstance(name_or_index, (int, np.integer)):
            return str(self.serogroups[int(name_or_index)])
        matches = np.nonzero(self.names == str(name_or_index))[0]
        if len(matches) == 0:
            raise KeyError(f"Serotype {name_or_index!r} not found in SerotypeDatabase.")
        return str(self.serogroups[int(matches[0])])

    def filter(
        self,
        *,
        names: str | Iterable[str] | None = None,
        serogroups: str | Iterable[str] | None = None,
        accessions: str | Iterable[str] | None = None,
        categories: ArchetypeCategory | str | Iterable[ArchetypeCategory | str] | None = None,
        archetypes: LocusArchetype | str | int | Iterable[LocusArchetype | str | int] | None = None,
        min_completeness: float = 0.0,
        min_score: float = 0.0,
    ) -> Self:
        """Filter the reference database by serotype `names`, `serogroups`, `accessions`, or locus properties."""
        n = len(self)
        if n == 0:
            return self
        mask = np.ones(n, dtype=np.bool_)
        if names is not None:
            items = [names] if isinstance(names, str) else list(names)
            mask &= np.isin(self.names, np.array([str(x) for x in items], dtype=str))
        if serogroups is not None:
            items = [serogroups] if isinstance(serogroups, str) else list(serogroups)
            mask &= np.isin(self.serogroups, np.array([str(x) for x in items], dtype=str))
        if accessions is not None:
            items = [accessions] if isinstance(accessions, str) else list(accessions)
            mask &= np.isin(self.accessions, np.array([str(x) for x in items], dtype=str))
        if min_completeness > 0.0:
            mask &= self.loci.completeness >= min_completeness
        if min_score > 0.0:
            mask &= self.loci.scores >= min_score
        if categories is not None or archetypes is not None:
            sub_loci = self.loci.filter(categories=categories, archetypes=archetypes)
            mask &= np.isin(np.arange(n), np.nonzero(np.isin(self.names, sub_loci.names))[0])
        return self[mask]

    def without(self, names_or_indices: str | int | Iterable[str | int]) -> Self:
        """Return a new `SerotypeDatabase` excluding the specified serotype `name`(s) or 0-based index(es)."""
        n = len(self)
        if n == 0:
            return self
        items = [names_or_indices] if isinstance(names_or_indices, (str, int, np.integer)) else list(names_or_indices)
        keep = np.ones(n, dtype=np.bool_)
        for item in items:
            if isinstance(item, (int, np.integer)):
                idx = int(item)
                if idx < 0:
                    idx += n
                if 0 <= idx < n:
                    keep[idx] = False
            else:
                keep &= self.names != str(item)
        return self[keep]

    def match(
        self,
        query: Locus | Loci | Iterable[Locus],
        *,
        top_k: int | None = None,
        min_score: float = 0.0,
        sort: bool = True,
    ) -> Matches:
        """Match one or more query `Locus` / `Loci` against this reference database."""
        return self.matcher.match(query, top_k=top_k, min_score=min_score, sort=sort)

    def is_novel(
        self,
        query: Locus | Loci,
        *,
        allow_variants: MatchType = BENIGN_VARIANT_MASK,
        min_coverage: float = 1.0,
    ) -> bool | np.ndarray[tuple[int], np.dtype[np.bool_]]:
        """Return True if `query` exhibits structural or gene-content novelty relative to this database."""
        return self.matcher.is_novel(query, allow_variants=allow_variants, min_coverage=min_coverage)

    def align(
        self,
        query: Locus,
        target: int | str,
        *,
        canonicalize_strand: bool = True,
        heal_mges: bool = False,
    ) -> Alignment:
        """Perform node-by-node bipartite `GraphAligner` alignment between `query` and a target serotype (`int` or `str`)."""
        return self.matcher.align(
            query,
            target,
            canonicalize_strand=canonicalize_strand,
            heal_mges=heal_mges,
        )

    def type_genome(
        self,
        target: bytes | str | Sequence | Sequences | GenomeAssembly,
        *,
        predictor: LocusPredictor | None = None,
        categories: ArchetypeCategory | str | Iterable[ArchetypeCategory | str] | None = ArchetypeCategory.CAPSULE,
        archetypes: LocusArchetype | str | int | Iterable[LocusArchetype | str | int] | None = None,
        top_k: int = 5,
    ) -> tuple[Locus | None, Matches]:
        """Segment a complete genome or assembly and match its primary antigen locus against this database.

        Args:
            target: Input genomic sequence, `Sequences` batch, or `GenomeAssembly`.
            predictor: Optional pre-configured `LocusPredictor` (defaults to `LocusPredictor(whole_genome=True)`).
            categories: Optional `ArchetypeCategory` filter (defaults to `ArchetypeCategory.CAPSULE`).
            archetypes: Optional `LocusArchetype` filter.
            top_k: Number of top candidate database matches to return (`5`).

        Returns:
            tuple[Locus | None, Matches]: The primary segmented `Locus` (or `None` if absent) and its top `Matches`.
        """
        engine = predictor if predictor is not None else LocusPredictor(whole_genome=True)
        pred_loci = engine.predict(target, categories=categories, archetypes=archetypes)
        best_loc = pred_loci.best()
        if best_loc is None:
            return None, Matches.empty()
        return best_loc, self.match(best_loc, top_k=top_k)

    @overload
    def __getitem__(self, item: int | np.integer[Any] | str) -> SerotypeRecord: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> Self: ...

    @overload
    def __getitem__(
        self, item: int | np.integer[Any] | str | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> SerotypeRecord | Self: ...

    def __getitem__(
        self, item: int | np.integer[Any] | str | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> SerotypeRecord | Self:
        if isinstance(item, str):
            matches = np.nonzero(self.names == item)[0]
            if len(matches) == 0:
                raise KeyError(f"Serotype {item!r} not found in SerotypeDatabase.")
            item = int(matches[0])

        if isinstance(item, (int, np.integer)):
            idx = int(item)
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(f"SerotypeDatabase index {item} out of range [0, {len(self) - 1}]")
            return SerotypeRecord(
                name=str(self.names[idx]),
                serogroup=str(self.serogroups[idx]),
                accession=str(self.accessions[idx]),
                strain=str(self.strains[idx]),
                locus=self.loci[idx],
            )

        indices = self._normalize_indices(item)
        if len(indices) == 0:
            return self.empty()

        sub_loci = self.loci[indices]
        sub_matcher = LocusMatcher(
            sub_loci,
            min_identity=self.matcher.min_identity,
            min_coverage=self.matcher.min_coverage,
            min_bitscore=self.matcher.min_bitscore,
            cluster_mode=cast(Any, self.matcher.cluster_mode),
            inflation=self.matcher.inflation,
            novel_jaccard_threshold=self.matcher.novel_jaccard_threshold,
            min_score=self.matcher.min_score,
            k=self.matcher.k,
            s=self.matcher.s,
            aligner_k=self.matcher.aligner_k,
        )
        pan_adj, pan_nodes, pan_counts = sub_loci.as_pan_graph(
            node_mask=GeneNode.mask("ROLE_ID", "CLUSTER_ID"),
            canonicalize_strands=False,
            heal_mges=False,
            cluster_genes=False,
        )
        return type(self)(
            loci=sub_loci,
            names=self.names[indices],
            serogroups=self.serogroups[indices],
            accessions=self.accessions[indices],
            strains=self.strains[indices],
            matcher=sub_matcher,
            pan_adj=pan_adj,
            pan_nodes=pan_nodes,
            pan_counts=pan_counts,
        )

    def __repr__(self) -> str:
        return (
            f"SerotypeDatabase(serotypes={len(self)}, genes={self.total_genes}, "
            f"clusters={self.num_clusters}, pan_nodes={len(self.pan_nodes)})"
        )


LocusDatabase = SerotypeDatabase
