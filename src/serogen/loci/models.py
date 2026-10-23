"""Data models for biosynthetic gene clusters (`Locus`, `Loci`) and syntenic graph primitives."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Self, cast, overload

import numpy as np
from scipy.sparse import csr_array

from serogen.core.graph.models import TAG_EDGE, BitField, DirectedEdge, GraphBatch, GraphModel, IntNode
from serogen.core.interval import Interval, Intervals, Strand
from serogen.core.seq.clustering import Clusterer, Clusters
from serogen.genes.models import Genes
from serogen.genes.roles import GeneRole
from serogen.loci.archetypes import ArchetypeCategory, LocusArchetype, get_archetype_tensors, roles_to_mask


# ----------------------------------------------------------------------------------------------------------------------
# Syntenic Graph Node & Edge Bit-Packed Primitives
# ----------------------------------------------------------------------------------------------------------------------
class IntergenicOrientation(IntEnum):
    """Relative transcriptional orientation between two adjacent genes in a locus."""

    CODIRECTIONAL = 0  # -> -> or <- <-
    CONVERGENT = 1  # -> <-
    DIVERGENT = 2  # <- ->


class IntergenicType(IntNode):
    """32-bit integer encoding for an intergenic synteny transition between adjacent genes.

    Bit 0..1 (2 bits):  Relative transcriptional orientation (`IntergenicOrientation`, 0..2).
    Bit 2..5 (4 bits):  Discretized intergenic distance bin (0 = overlap, 1 = <50bp, 2 = 50..200bp,
                        3 = >200bp, 15 = healed MGE scar).
    Bit 31 (1 bit):     `TAG_EDGE` (`1` = intermediate edge node in bipartite graphs).
    """

    ORIENTATION = BitField(shift=0, width=2)
    GAP_BIN = BitField(shift=2, width=4)
    TAG_EDGE = BitField(shift=31, width=1)

    def __new__(
        cls,
        val_or_orientation: int | IntergenicOrientation = IntergenicOrientation.CODIRECTIONAL,
        gap_bin: int | None = None,
    ) -> Self:
        if gap_bin is None and int(val_or_orientation) & TAG_EDGE:
            return super().__new__(cls, int(val_or_orientation))
        g_bin = 1 if gap_bin is None else int(gap_bin)
        packed = cls.ORIENTATION.pack(int(val_or_orientation)) | cls.GAP_BIN.pack(g_bin) | cls.TAG_EDGE.pack(1)
        return super().__new__(cls, packed)

    @classmethod
    def from_adjacent(cls, strand_a: int, strand_b: int, gap_bp: int = 0) -> Self:
        """Construct an `IntergenicType` from adjacent gene strands and base-pair distance."""
        if strand_a == strand_b:
            orient = IntergenicOrientation.CODIRECTIONAL
        elif strand_a >= 0 and strand_b < 0:
            orient = IntergenicOrientation.CONVERGENT
        else:
            orient = IntergenicOrientation.DIVERGENT

        if gap_bp < 0:
            g_bin = 0
        elif gap_bp < 50:
            g_bin = 1
        elif gap_bp <= 200:
            g_bin = 2
        else:
            g_bin = 3
        return cls(orient, gap_bin=g_bin)

    @property
    def orientation(self) -> IntergenicOrientation:
        """Relative transcriptional orientation between source and target genes."""
        return IntergenicOrientation(int(self.ORIENTATION))

    @property
    def gap_bin(self) -> int:
        """Discretized intergenic distance bin (`0..15`)."""
        return int(self.GAP_BIN)

    @property
    def is_contig_break(self) -> bool:
        """True if this edge bridges a contig boundary in a fragmented assembly."""
        return int(self.GAP_BIN) == 14

    @property
    def is_healed_scar(self) -> bool:
        """True if this edge bridges a healed mobile genetic element (MGE) insertion."""
        return int(self.GAP_BIN) == 15

    def __repr__(self) -> str:
        arrows = {
            IntergenicOrientation.CODIRECTIONAL: "->->",
            IntergenicOrientation.CONVERGENT: "-><-",
            IntergenicOrientation.DIVERGENT: "<-->",
        }[self.orientation]
        if self.is_healed_scar:
            suffix = ",scar"
        elif self.is_contig_break:
            suffix = ",break"
        else:
            suffix = f",bin={self.gap_bin}"
        return f"IntergenicType({arrows}{suffix})"


@dataclass(slots=True, frozen=True)
class IntergenicEdge(DirectedEdge):
    """Directed synteny edge connecting adjacent gene nodes `(source -> target)` in a `Locus`."""

    source: int
    target: int
    edge_type: IntergenicType


class GeneNode(IntNode):
    """32-bit integer encoding for a gene node inside a biosynthetic locus graph.

    Bit 0..4 (5 bits):   Canonical `GeneRole` index (`0..29`).
    Bit 5 (1 bit):       Frameshift / pseudogene indicator (`0` = intact, `1` = frameshifted).
    Bit 6..30 (25 bits): Sequence homology cluster ID (`0` = unclustered, `1..33,554,431` = cluster ID).
    Bit 31 (1 bit):      `TAG_EDGE` (`0` = primary gene node).
    """

    ROLE_ID = BitField(shift=0, width=5)
    IS_FRAMESHIFT = BitField(shift=5, width=1)
    CLUSTER_ID = BitField(shift=6, width=25)
    TAG_EDGE = BitField(shift=31, width=1)

    def __new__(
        cls,
        val_or_role: int | str | bytes | GeneRole,
        cluster_id: int | None = None,
        is_frameshift: bool = False,
    ) -> Self:
        if (
            isinstance(val_or_role, (int, np.integer))
            and not isinstance(val_or_role, GeneRole)
            and cluster_id is None
            and not is_frameshift
            and int(val_or_role) >= len(GeneRole)
        ):
            return super().__new__(cls, int(val_or_role))

        role_enum = GeneRole.resolve(val_or_role) if not isinstance(val_or_role, GeneRole) else val_or_role
        cid = 0 if cluster_id is None else int(cluster_id)
        packed = (
            cls.ROLE_ID.pack(int(role_enum)) | cls.IS_FRAMESHIFT.pack(int(is_frameshift)) | cls.CLUSTER_ID.pack(cid)
        )
        return super().__new__(cls, packed)

    @property
    def role_idx(self) -> int:
        """Integer `GeneRole` index (`0..29`)."""
        return int(self.ROLE_ID)

    @property
    def role(self) -> GeneRole:
        """Canonical `GeneRole` enum member."""
        return GeneRole(int(self.ROLE_ID))

    @property
    def is_frameshift(self) -> bool:
        """True if this gene contains a frameshift indel or premature stop."""
        return bool(self.IS_FRAMESHIFT)

    @property
    def cluster_id(self) -> int:
        """Sequence homology cluster ID (`0` if unclustered)."""
        return int(self.CLUSTER_ID)

    @property
    def is_clustered(self) -> bool:
        """True if assigned to a homology cluster (`cluster_id > 0`)."""
        return int(self.CLUSTER_ID) > 0

    @property
    def is_mobile_element(self) -> bool:
        """True if this node is a mobile genetic element (`TRANSPOSASE` or `INTEGRASE`)."""
        return self.role.is_mobile_element

    @property
    def is_biosynthetic(self) -> bool:
        """True if this node is an active biosynthetic, modifier, or transport gene."""
        return self.role.is_biosynthetic

    def with_cluster(self, cluster_id: int) -> GeneNode:
        """Return a copy of this `GeneNode` with an updated homology `cluster_id`."""
        cleared = int(self) & ~type(self).CLUSTER_ID.mask
        return GeneNode(cleared | type(self).CLUSTER_ID.pack(int(cluster_id)))

    def __repr__(self) -> str:
        r_abbr = self.role.abbreviation
        fs = ",FS" if self.is_frameshift else ""
        cid = f",c={self.cluster_id}" if self.cluster_id > 0 else ""
        return f"GeneNode({r_abbr}{cid}{fs})"


# ----------------------------------------------------------------------------------------------------------------------
# Scalar Locus Container
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True, eq=False)
class Locus(GraphModel):
    """Scalar view and directed syntenic graph of a single biosynthetic gene cluster.

    Embeds its constituent [`Genes`][serogen.genes.models.Genes] container (`self.genes`),
    deriving per-gene intervals, strands, lengths, coding sequences, and frameshifts directly
    without duplicating metadata across parallel tuples.
    """

    nodes: tuple[GeneNode, ...]
    edges: tuple[IntergenicEdge, ...]
    genes: Genes = field(default_factory=Genes.empty)
    interval: Interval = Interval(0, 0, Strand.FORWARD)
    contig_index: int = 0
    archetype: LocusArchetype = LocusArchetype.UNKNOWN
    completeness: float = 0.0
    score: float = 0.0
    is_tandem_split: bool = False
    name: str = ""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Locus):
            return NotImplemented
        return bool(
            self.nodes == other.nodes
            and self.edges == other.edges
            and self.interval == other.interval
            and self.contig_index == other.contig_index
            and self.archetype == other.archetype
            and np.isclose(self.completeness, other.completeness)
            and np.isclose(self.score, other.score)
            and self.is_tandem_split == other.is_tandem_split
            and self.name == other.name
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.nodes,
                self.edges,
                self.interval,
                self.contig_index,
                self.archetype,
                round(float(self.completeness), 6),
                round(float(self.score), 6),
                self.is_tandem_split,
                self.name,
            )
        )

    def with_name(self, name: str | bytes) -> Locus:
        """Return a copy of this `Locus` with an updated identifier `name`."""
        name_str = name.decode("utf-8", errors="replace") if isinstance(name, bytes) else str(name)
        return Locus(
            nodes=self.nodes,
            edges=self.edges,
            genes=self.genes,
            interval=self.interval,
            contig_index=self.contig_index,
            archetype=self.archetype,
            completeness=self.completeness,
            score=self.score,
            is_tandem_split=self.is_tandem_split,
            name=name_str,
        )

    @property
    def num_genes(self) -> int:
        """Number of genes in this locus."""
        return len(self.nodes)

    @property
    def start(self) -> int:
        """0-based inclusive start coordinate on the parent contig."""
        return self.interval.start

    @property
    def end(self) -> int:
        """0-based exclusive end coordinate on the parent contig."""
        return self.interval.end

    @property
    def strand(self) -> int:
        """Dominant transcriptional strand (`+1` or `-1`) of the locus."""
        return self.interval.strand

    @property
    def strands(self) -> np.ndarray[tuple[int], np.dtype[np.int8]]:
        """1D `int8` array of per-gene transcriptional strands from `self.genes`."""
        if len(self.genes) > 0:
            return self.genes.intervals.strands
        return np.ones(len(self.nodes), dtype=np.int8)

    @property
    def lengths(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """1D `int32` array of per-gene base-pair lengths from `self.genes`."""
        if len(self.genes) > 0:
            return self.genes.intervals.lengths
        return np.full(len(self.nodes), 1000, dtype=np.int32)

    @property
    def roles(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """1D array of integer `GeneRole` indices (`0..29`) for each gene in the locus."""
        if not self.nodes:
            return np.empty(0, dtype=np.int32)
        arr = np.fromiter((int(n) for n in self.nodes), dtype=np.uint32, count=len(self.nodes))
        return GeneNode.ROLE_ID.extract_array(arr).astype(np.int32)

    @property
    def cluster_ids(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """1D array of homology `cluster_id` values for each gene in the locus."""
        if not self.nodes:
            return np.empty(0, dtype=np.int32)
        arr = np.fromiter((int(n) for n in self.nodes), dtype=np.uint32, count=len(self.nodes))
        return GeneNode.CLUSTER_ID.extract_array(arr).astype(np.int32)

    @property
    def frameshifts(self) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        """1D boolean array indicating which genes carry frameshift mutations."""
        if not self.nodes:
            return np.empty(0, dtype=np.bool_)
        arr = np.fromiter((int(n) for n in self.nodes), dtype=np.uint32, count=len(self.nodes))
        return GeneNode.IS_FRAMESHIFT.extract_array(arr).astype(np.bool_)

    @property
    def role_mask(self) -> int:
        """32-bit integer presence bitmask of all `GeneRole`s present in this locus."""
        return roles_to_mask(self.roles)

    @property
    def has_mobile_elements(self) -> bool:
        """True if this locus contains any `TRANSPOSASE` or `INTEGRASE` insertions."""
        if not self.nodes:
            return False
        roles = self.roles
        return bool(np.any((roles == int(GeneRole.TRANSPOSASE)) | (roles == int(GeneRole.INTEGRASE))))

    @property
    def has_mge(self) -> bool:
        """Alias for `has_mobile_elements`."""
        return self.has_mobile_elements

    @property
    def is_fragmented(self) -> bool:
        """True if this locus spans multiple contigs across an assembly break."""
        if any(e.edge_type.is_contig_break for e in self.edges):
            return True
        c_idx = self.genes.contig_indices
        return len(c_idx) > 1 and not bool(np.all(c_idx == c_idx[0]))

    @property
    def num_frameshifts(self) -> int:
        """Number of genes in this locus carrying frameshift mutations or premature stops."""
        if not self.nodes:
            return 0
        return int(np.sum(self.frameshifts))

    @property
    def frameshift_fraction(self) -> float:
        """Fraction of genes in this locus (`[0.0, 1.0]`) carrying frameshift mutations."""
        if not self.nodes:
            return 0.0
        return float(self.num_frameshifts) / float(len(self.nodes))

    @property
    def is_intact(self) -> bool:
        """True if this locus is on a single contig, has no mobile elements, and has no frameshifted genes."""
        return not self.is_fragmented and not self.has_mobile_elements and self.num_frameshifts == 0

    @property
    def category(self) -> ArchetypeCategory:
        """High-level `ArchetypeCategory` (`Capsule`, `O-Antigen`, `Exopolysaccharide`, `Other`) of this locus."""
        return self.archetype.category

    @property
    def num_contigs(self) -> int:
        """Number of distinct contigs spanned by this locus."""
        c_idx = self.genes.contig_indices
        if len(c_idx) > 0:
            return int(len(np.unique(c_idx)))
        return 1

    def canonicalize_strand(self) -> Locus:
        """Orient the locus into canonical 5' -> 3' operon order (`strand = +1`).

        If `self.strand >= 0`, returns `self` unchanged. If `self.strand < 0`, reverses
        `nodes` and `genes` and re-indexes `edges` so that directed synteny transitions
        run 5' -> 3' along the dominant transcriptional strand.
        """
        if self.strand >= 0:
            return self
        n = len(self.nodes)
        new_interval = Interval(self.interval.start, self.interval.end, Strand.FORWARD)
        if n == 0:
            return Locus(
                nodes=(),
                edges=(),
                genes=Genes.empty(),
                interval=new_interval,
                contig_index=self.contig_index,
                archetype=self.archetype,
                completeness=self.completeness,
                score=self.score,
                is_tandem_split=self.is_tandem_split,
                name=self.name,
            )

        new_nodes = self.nodes[::-1]
        new_edges = tuple(
            IntergenicEdge(
                source=n - 1 - e.target,
                target=n - 1 - e.source,
                edge_type=e.edge_type,
            )
            for e in reversed(self.edges)
        )
        new_genes = self.genes[::-1] if len(self.genes) == n else self.genes
        return Locus(
            nodes=new_nodes,
            edges=new_edges,
            genes=new_genes,
            interval=new_interval,
            contig_index=self.contig_index,
            archetype=self.archetype,
            completeness=self.completeness,
            score=self.score,
            is_tandem_split=self.is_tandem_split,
            name=self.name,
        )

    def heal_mobile_elements(self, remove_nodes: bool = True) -> Locus:
        """Excise mobile genetic element (`TRANSPOSASE`, `INTEGRASE`) insertions and heal synteny edges."""
        if not self.has_mobile_elements or not remove_nodes:
            return self

        roles = self.roles
        keep_mask = (roles != int(GeneRole.TRANSPOSASE)) & (roles != int(GeneRole.INTEGRASE))
        kept_idx = np.nonzero(keep_mask)[0]
        if len(kept_idx) == 0:
            return Locus(
                nodes=(),
                edges=(),
                genes=Genes.empty(),
                interval=self.interval,
                contig_index=self.contig_index,
                archetype=self.archetype,
                completeness=self.completeness,
                score=self.score,
                is_tandem_split=self.is_tandem_split,
                name=self.name,
            )

        new_nodes = tuple(self.nodes[int(i)] for i in kept_idx)
        new_genes = self.genes[keep_mask] if len(self.genes) == len(self.nodes) else self.genes

        strands = new_genes.intervals.strands if len(new_genes) > 0 else np.ones(len(new_nodes), dtype=np.int8)
        starts = new_genes.intervals.starts if len(new_genes) > 0 else np.arange(len(new_nodes), dtype=np.int32) * 1000
        ends = new_genes.intervals.ends if len(new_genes) > 0 else starts + 900
        c_indices = new_genes.contig_indices if len(new_genes) > 0 else np.zeros(len(new_nodes), dtype=np.uint32)
        edge_map = {(e.source, e.target): e.edge_type for e in self.edges}

        new_edges: list[IntergenicEdge] = []
        for k in range(len(kept_idx) - 1):
            orig_a = int(kept_idx[k])
            orig_b = int(kept_idx[k + 1])
            step_sign = 1 if int(starts[k + 1]) >= int(starts[k]) else -1
            s_a = int(strands[k]) * step_sign
            s_b = int(strands[k + 1]) * step_sign
            if orig_b - orig_a > 1:
                if s_a == s_b:
                    orient = IntergenicOrientation.CODIRECTIONAL
                elif s_a >= 0 and s_b < 0:
                    orient = IntergenicOrientation.CONVERGENT
                else:
                    orient = IntergenicOrientation.DIVERGENT
                spans_break = int(c_indices[k]) != int(c_indices[k + 1]) or any(
                    edge_map.get((m, m + 1), IntergenicType()).is_contig_break for m in range(orig_a, orig_b)
                )
                etype = IntergenicType(orient, gap_bin=14 if spans_break else 15)
            elif (orig_a, orig_b) in edge_map:
                etype = edge_map[(orig_a, orig_b)]
            else:
                gap_bp = int(starts[k + 1]) - int(ends[k]) if step_sign > 0 else int(starts[k]) - int(ends[k + 1])
                etype = IntergenicType.from_adjacent(s_a, s_b, gap_bp=gap_bp)
            new_edges.append(IntergenicEdge(source=k, target=k + 1, edge_type=etype))

        return Locus(
            nodes=new_nodes,
            edges=tuple(new_edges),
            genes=new_genes,
            interval=self.interval,
            contig_index=self.contig_index,
            archetype=self.archetype,
            completeness=self.completeness,
            score=self.score,
            is_tandem_split=self.is_tandem_split,
            name=self.name,
        )

    @classmethod
    def from_genes(
        cls,
        genes: Genes | Sequence[str | bytes | int | GeneRole] | np.ndarray,
        roles: Sequence[str | bytes | int | GeneRole] | np.ndarray | None = None,
        *,
        strands: Sequence[int] | np.ndarray | None = None,
        lengths: Sequence[int] | np.ndarray | None = None,
        clusters: Sequence[int] | np.ndarray | None = None,
        frameshifts: Sequence[bool] | np.ndarray | None = None,
        archetype: LocusArchetype = LocusArchetype.UNKNOWN,
        completeness: float = 0.0,
        score: float = 0.0,
        is_tandem_split: bool = False,
        contig_index: int = 0,
        name: str = "",
    ) -> Self:
        """Construct a `Locus` from either a `Genes` container or a sequence of gene symbols/roles."""
        if isinstance(genes, Genes):
            genes_obj = genes
            n = len(genes_obj)
            if n == 0:
                return cls(nodes=(), edges=(), genes=Genes.empty(), archetype=archetype, name=name)
            if roles is None:
                role_ids = np.full(n, int(GeneRole.UNKNOWN), dtype=np.int32)
            elif isinstance(roles, np.ndarray) and roles.dtype.kind in ("i", "u"):
                role_ids = roles.astype(np.int32, copy=False)
            else:
                role_ids = np.array(
                    [int(GeneRole.label(r) if isinstance(r, (str, bytes)) else GeneRole.resolve(r)) for r in roles],
                    dtype=np.int32,
                )
            fs_arr = np.asarray(frameshifts, dtype=np.bool_) if frameshifts is not None else genes_obj.has_frameshift
            c_idx = int(genes_obj.contig_indices[0]) if len(genes_obj.contig_indices) > 0 else contig_index
        else:
            raw_list = list(genes)
            n = len(raw_list)
            if n == 0:
                return cls(nodes=(), edges=(), genes=Genes.empty(), archetype=archetype, name=name)

            if roles is not None:
                role_ids = np.array(
                    [int(GeneRole.label(r) if isinstance(r, (str, bytes)) else GeneRole.resolve(r)) for r in roles],
                    dtype=np.int32,
                )
            else:
                role_ids = np.empty(n, dtype=np.int32)
                for i, item in enumerate(raw_list):
                    if isinstance(item, (GeneRole, int, np.integer)):
                        role_ids[i] = int(item)
                    else:
                        s_item = item.decode("utf-8", errors="replace") if isinstance(item, bytes) else str(item)
                        try:
                            role_ids[i] = int(GeneRole.from_name(s_item))
                        except KeyError:
                            role_ids[i] = int(GeneRole.label(s_item))

            strands_arr = np.asarray(strands, dtype=np.int8) if strands is not None else np.ones(n, dtype=np.int8)
            lengths_arr = (
                np.asarray(lengths, dtype=np.int32) if lengths is not None else np.full(n, 1000, dtype=np.int32)
            )
            starts_arr = np.zeros(n, dtype=np.int32)
            if n > 1:
                np.cumsum(lengths_arr[:-1] + 50, out=starts_arr[1:])
            ends_arr = starts_arr + lengths_arr
            genes_obj = Genes(
                intervals=Intervals(starts=starts_arr, ends=ends_arr, strands=strands_arr),
                contig_indices=np.full(n, contig_index, dtype=np.uint32),
            )
            fs_arr = np.asarray(frameshifts, dtype=np.bool_) if frameshifts is not None else np.zeros(n, dtype=np.bool_)
            c_idx = contig_index

        cluster_arr = np.asarray(clusters, dtype=np.int32) if clusters is not None else np.zeros(n, dtype=np.int32)

        nodes = tuple(
            GeneNode(int(role_ids[i]), cluster_id=int(cluster_arr[i]), is_frameshift=bool(fs_arr[i])) for i in range(n)
        )

        g_starts = genes_obj.intervals.starts
        g_ends = genes_obj.intervals.ends
        g_strands = genes_obj.intervals.strands
        g_contigs = genes_obj.contig_indices

        edge_list: list[IntergenicEdge] = []
        for i in range(n - 1):
            c_a = int(g_contigs[i])
            c_b = int(g_contigs[i + 1])
            if c_b != c_a:
                s_a = int(g_strands[i])
                s_b = int(g_strands[i + 1])
                if s_a == s_b:
                    orient = IntergenicOrientation.CODIRECTIONAL
                elif s_a >= 0 and s_b < 0:
                    orient = IntergenicOrientation.CONVERGENT
                else:
                    orient = IntergenicOrientation.DIVERGENT
                etype = IntergenicType(orient, gap_bin=14)
            else:
                etype = IntergenicType.from_adjacent(
                    int(g_strands[i]),
                    int(g_strands[i + 1]),
                    gap_bp=int(g_starts[i + 1]) - int(g_ends[i]),
                )
            edge_list.append(IntergenicEdge(source=i, target=i + 1, edge_type=etype))

        edges = tuple(edge_list)
        dom_strand = Strand.FORWARD if int(np.sum(g_strands)) >= 0 else Strand.REVERSE
        locus_interval = Interval(int(g_starts.min()), int(g_ends.max()), dom_strand)

        return cls(
            nodes=nodes,
            edges=edges,
            genes=genes_obj,
            interval=locus_interval,
            contig_index=c_idx,
            archetype=archetype,
            completeness=float(completeness),
            score=float(score),
            is_tandem_split=bool(is_tandem_split),
            name=name,
        )

    def match(self, targets: Locus | Loci, **kwargs: Any) -> Any:
        """Compare this `Locus` against a target `Locus` (returns `Match`) or target `Loci` database (returns `Matches`)."""
        from serogen.loci.matcher import LocusMatcher

        matcher = LocusMatcher(targets, **kwargs)
        matches = matcher.match(self)
        if isinstance(targets, Locus):
            return matches[0]
        return matches

    def __repr__(self) -> str:
        name_part = f"name={self.name!r}, " if self.name else ""
        return (
            f"Locus({name_part}archetype={self.archetype.name}, genes={self.num_genes}, "
            f"interval={self.start}..{self.end}({self.strand:+d}), "
            f"completeness={self.completeness:.2f}, score={self.score:.2f})"
        )


# ----------------------------------------------------------------------------------------------------------------------
# Batched Structure-of-Arrays Loci Container
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class Loci(GraphBatch[Locus, "Loci"]):
    """Structure-of-Arrays (SoA) batch container for biosynthetic gene clusters (`Locus`).

    Combines `GraphBatch` sparse topology arrays (`node_types`, `node_offsets`, `edge_sources`,
    `edge_targets`, `edge_types`, `edge_offsets`) with a flat [`Genes`][serogen.genes.models.Genes]
    container (`self.genes`, indexed per locus by `node_offsets`) and per-locus archetype metadata.
    """

    node_types: np.ndarray[tuple[int], np.dtype[np.uint32]]
    node_offsets: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_sources: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_targets: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_types: np.ndarray[tuple[int], np.dtype[np.uint32]]
    edge_offsets: np.ndarray[tuple[int], np.dtype[np.uint32]]
    genes: Genes = field(default_factory=Genes.empty)
    intervals: Intervals = field(default_factory=Intervals.empty)
    contig_indices: np.ndarray[tuple[int], np.dtype[np.uint32]] = field(
        default_factory=lambda: np.empty(0, dtype=np.uint32)
    )
    archetypes: np.ndarray[tuple[int], np.dtype[np.int8]] = field(default_factory=lambda: np.empty(0, dtype=np.int8))
    completeness: np.ndarray[tuple[int], np.dtype[np.float32]] = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    scores: np.ndarray[tuple[int], np.dtype[np.float32]] = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    is_tandem_split: np.ndarray[tuple[int], np.dtype[np.bool_]] = field(
        default_factory=lambda: np.empty(0, dtype=np.bool_)
    )
    names: np.ndarray[tuple[int], np.dtype[np.str_]] = field(default_factory=lambda: np.empty(0, dtype="U1"))

    def __post_init__(self) -> None:
        n = max(0, len(self.node_offsets) - 1)
        if len(self.names) == 0 and n > 0:
            object.__setattr__(self, "names", np.full(n, "", dtype="U1"))
        elif self.names.dtype.kind == "S":
            object.__setattr__(self, "names", np.char.decode(cast(Any, self.names), "utf-8", errors="replace"))
        elif self.names.dtype.kind != "U":
            object.__setattr__(self, "names", np.asarray(self.names, dtype=str))

    def __len__(self) -> int:
        return len(self.node_offsets) - 1

    def with_names(self, names: Sequence[str | bytes] | np.ndarray) -> Self:
        """Return a new `Loci` batch with updated per-locus identifier `names`."""
        raw = [x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x) for x in names]
        if len(raw) != len(self):
            raise ValueError(f"Expected {len(self)} locus names, got {len(raw)}.")
        names_arr = np.array(raw, dtype=str) if raw else np.empty(0, dtype="U1")
        return type(self)(
            node_types=self.node_types,
            node_offsets=self.node_offsets,
            edge_sources=self.edge_sources,
            edge_targets=self.edge_targets,
            edge_types=self.edge_types,
            edge_offsets=self.edge_offsets,
            genes=self.genes,
            intervals=self.intervals,
            contig_indices=self.contig_indices,
            archetypes=self.archetypes,
            completeness=self.completeness,
            scores=self.scores,
            is_tandem_split=self.is_tandem_split,
            names=names_arr,
        )

    @property
    def roles(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """Flat 1D array of `GeneRole` indices (`0..29`) across all gene nodes in the batch."""
        return GeneNode.ROLE_ID.extract_array(self.node_types).astype(np.int32)

    @property
    def cluster_ids(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """Flat 1D array of homology `cluster_id` values across all gene nodes in the batch."""
        return GeneNode.CLUSTER_ID.extract_array(self.node_types).astype(np.int32)

    @property
    def is_fragmented(self) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        """1D boolean array (`(N,)`) indicating whether each locus spans a contig break (`gap_bin == 14`)."""
        n_loci = len(self)
        if n_loci == 0:
            return np.empty(0, dtype=np.bool_)
        if len(self.edge_types) == 0:
            return np.zeros(n_loci, dtype=np.bool_)
        edge_breaks = IntergenicType.GAP_BIN.extract_array(self.edge_types) == 14
        out = np.zeros(n_loci, dtype=np.bool_)
        for i in range(n_loci):
            s = int(self.edge_offsets[i])
            e = int(self.edge_offsets[i + 1])
            if e > s and np.any(edge_breaks[s:e]):
                out[i] = True
        return out

    @property
    def num_frameshifts(self) -> np.ndarray[tuple[int], np.dtype[np.int32]]:
        """1D `int32` array (`(N,)`) of frameshifted gene counts per locus."""
        n_loci = len(self)
        if n_loci == 0:
            return np.empty(0, dtype=np.int32)
        if len(self.node_types) == 0:
            return np.zeros(n_loci, dtype=np.int32)
        fs_nodes = GeneNode.IS_FRAMESHIFT.extract_array(self.node_types).astype(np.int32)
        out = np.zeros(n_loci, dtype=np.int32)
        for i in range(n_loci):
            s = int(self.node_offsets[i])
            e = int(self.node_offsets[i + 1])
            if e > s:
                out[i] = int(np.sum(fs_nodes[s:e]))
        return out

    @property
    def frameshift_fractions(self) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
        """1D `float32` array (`(N,)`) of frameshifted gene fractions (`[0.0, 1.0]`) per locus."""
        n_loci = len(self)
        if n_loci == 0:
            return np.empty(0, dtype=np.float32)
        counts = (self.node_offsets[1:] - self.node_offsets[:-1]).astype(np.float32)
        fs = self.num_frameshifts.astype(np.float32)
        return np.where(counts > 0.0, fs / np.maximum(counts, 1.0), np.float32(0.0)).astype(np.float32)

    @property
    def has_mge(self) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        """1D boolean array (`(N,)`) indicating whether each locus contains `TRANSPOSASE` or `INTEGRASE` insertions."""
        if len(self) == 0:
            return np.empty(0, dtype=np.bool_)
        return np.array([loc.has_mobile_elements for loc in self], dtype=np.bool_)

    @property
    def is_intact(self) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
        """1D boolean array (`(N,)`) indicating whether each locus is single-contig, MGE-free, and frameshift-free."""
        n_loci = len(self)
        if n_loci == 0:
            return np.empty(0, dtype=np.bool_)
        return (~self.is_fragmented) & (~self.has_mge) & (self.num_frameshifts == 0)

    @property
    def categories(self) -> np.ndarray[tuple[int], np.dtype[np.str_]]:
        """1D string array (`(N,)`) of `ArchetypeCategory` values (`'Capsule'`, `'O-Antigen'`, etc.) per locus."""
        if len(self) == 0:
            return np.empty(0, dtype="U20")
        return get_archetype_tensors().categories[self.archetypes]

    def best(self, *, prefer_intact: bool = False) -> Locus | None:
        """Return the highest-scoring `Locus` in the batch (or `None` if empty).

        Args:
            prefer_intact: If True, prioritize intact loci without frameshifts or assembly breaks
                (e.g., an intact ectopic `CPS_SYNTHASE` over a frameshifted/cryptic `CPS_WZY` pseudogene cluster).
        """
        if len(self) == 0:
            return None
        if prefer_intact:
            fs_frac = self.frameshift_fractions
            frag = self.is_fragmented.astype(np.int32)
            # Lexsort ascending by (is_fragmented, frameshift_fraction, -score) and take the first index
            order = np.lexsort((-self.scores, fs_frac, frag))
            return self[int(order[0])]
        return self[int(np.argmax(self.scores))]

    def filter(
        self,
        *,
        names: str | Iterable[str] | None = None,
        categories: ArchetypeCategory | str | Iterable[ArchetypeCategory | str] | None = None,
        archetypes: LocusArchetype | str | int | Iterable[LocusArchetype | str | int] | None = None,
        min_completeness: float = 0.0,
        min_score: float = 0.0,
        max_frameshift_fraction: float = 1.0,
        only_intact: bool = False,
    ) -> Self:
        """Filter loci by `names`, ontology `categories`, specific `archetypes`, `min_completeness`, `min_score`, `max_frameshift_fraction`, and/or `only_intact`."""
        n = len(self)
        if n == 0:
            return self
        mask = np.ones(n, dtype=np.bool_)
        if names is not None:
            name_items = [names] if isinstance(names, str) else list(names)
            allowed_names = np.array([str(x) for x in name_items], dtype=str)
            mask &= np.isin(self.names, allowed_names)
        if min_completeness > 0.0:
            mask &= self.completeness >= min_completeness
        if min_score > 0.0:
            mask &= self.scores >= min_score
        if max_frameshift_fraction < 1.0:
            mask &= self.frameshift_fractions <= max_frameshift_fraction + 1e-6
        if only_intact:
            mask &= ~self.is_fragmented
            mask &= np.array([not loc.has_mobile_elements for loc in self], dtype=np.bool_)
        if categories is not None:
            cat_items = [categories] if isinstance(categories, (str, ArchetypeCategory)) else list(categories)
            allowed_cats = {
                (c.value if isinstance(c, ArchetypeCategory) else str(c)).strip().lower().replace("_", "-")
                for c in cat_items
            }
            t_cats = np.array(
                [str(c).strip().lower().replace("_", "-") for c in get_archetype_tensors().categories],
                dtype="U20",
            )
            arch_cat_ok = np.array([c in allowed_cats for c in t_cats], dtype=np.bool_)
            mask &= arch_cat_ok[self.archetypes]
        if archetypes is not None:
            arch_items: list[LocusArchetype | str | int] = (
                [archetypes] if isinstance(archetypes, (str, int, LocusArchetype)) else list(archetypes)
            )
            allowed_arch_ids = np.array([int(LocusArchetype.from_name(a)) for a in arch_items], dtype=np.int8)
            mask &= np.isin(self.archetypes, allowed_arch_ids)
        return self[mask]

    @classmethod
    def empty(cls) -> Self:
        """Create an empty 0-length `Loci` container."""
        return cls(
            node_types=np.empty(0, dtype=np.uint32),
            node_offsets=np.zeros(1, dtype=np.uint32),
            edge_sources=np.empty(0, dtype=np.uint32),
            edge_targets=np.empty(0, dtype=np.uint32),
            edge_types=np.empty(0, dtype=np.uint32),
            edge_offsets=np.zeros(1, dtype=np.uint32),
            genes=Genes.empty(),
            intervals=Intervals.empty(),
            contig_indices=np.empty(0, dtype=np.uint32),
            archetypes=np.empty(0, dtype=np.int8),
            completeness=np.empty(0, dtype=np.float32),
            scores=np.empty(0, dtype=np.float32),
            is_tandem_split=np.empty(0, dtype=np.bool_),
            names=np.empty(0, dtype="U1"),
        )

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        """Concatenate multiple `Loci` batches into a single contiguous `Loci` container."""
        batch_list = [b for b in batches if len(b) > 0]
        if not batch_list:
            return cls.empty()
        if len(batch_list) == 1:
            return batch_list[0]

        nodes, n_offs, sources, targets, types, e_offs = cls._concat_soa_base(batch_list)
        return cls(
            node_types=nodes,
            node_offsets=n_offs,
            edge_sources=sources,
            edge_targets=targets,
            edge_types=types,
            edge_offsets=e_offs,
            genes=Genes.concat([b.genes for b in batch_list]),
            intervals=Intervals.concat([b.intervals for b in batch_list]),
            contig_indices=np.concatenate([b.contig_indices for b in batch_list]),
            archetypes=np.concatenate([b.archetypes for b in batch_list]),
            completeness=np.concatenate([b.completeness for b in batch_list]),
            scores=np.concatenate([b.scores for b in batch_list]),
            is_tandem_split=np.concatenate([b.is_tandem_split for b in batch_list]),
            names=np.concatenate([b.names for b in batch_list]),
        )

    @classmethod
    def from_list(cls, loci: Sequence[Locus]) -> Self:
        """Compile a sequence of scalar `Locus` objects into a contiguous `Loci` batch."""
        n_loci = len(loci)
        if n_loci == 0:
            return cls.empty()

        node_offsets = np.empty(n_loci + 1, dtype=np.uint32)
        node_offsets[0] = 0
        edge_offsets = np.empty(n_loci + 1, dtype=np.uint32)
        edge_offsets[0] = 0

        tot_nodes = sum(len(loc.nodes) for loc in loci)
        tot_edges = sum(len(loc.edges) for loc in loci)

        node_types = np.empty(tot_nodes, dtype=np.uint32)
        edge_sources = np.empty(tot_edges, dtype=np.uint32)
        edge_targets = np.empty(tot_edges, dtype=np.uint32)
        edge_types = np.empty(tot_edges, dtype=np.uint32)

        starts = np.empty(n_loci, dtype=np.int32)
        ends = np.empty(n_loci, dtype=np.int32)
        strands = np.empty(n_loci, dtype=np.int8)
        contig_indices = np.empty(n_loci, dtype=np.uint32)
        archetypes = np.empty(n_loci, dtype=np.int8)
        completeness = np.empty(n_loci, dtype=np.float32)
        scores = np.empty(n_loci, dtype=np.float32)
        is_tandem_split = np.empty(n_loci, dtype=np.bool_)
        names_list: list[str] = []
        genes_batches: list[Genes] = []

        curr_node = 0
        curr_edge = 0

        for i, loc in enumerate(loci):
            n_n = len(loc.nodes)
            n_e = len(loc.edges)
            if n_n > 0:
                node_types[curr_node : curr_node + n_n] = [int(n) for n in loc.nodes]
            if n_e > 0:
                for j, e in enumerate(loc.edges):
                    edge_sources[curr_edge + j] = e.source + curr_node
                    edge_targets[curr_edge + j] = e.target + curr_node
                    edge_types[curr_edge + j] = int(e.edge_type)

            curr_node += n_n
            curr_edge += n_e
            node_offsets[i + 1] = curr_node
            edge_offsets[i + 1] = curr_edge

            starts[i] = loc.interval.start
            ends[i] = loc.interval.end
            strands[i] = loc.interval.strand
            contig_indices[i] = loc.contig_index
            archetypes[i] = int(loc.archetype)
            completeness[i] = loc.completeness
            scores[i] = loc.score
            is_tandem_split[i] = loc.is_tandem_split
            names_list.append(loc.name)
            genes_batches.append(loc.genes)

        return cls(
            node_types=node_types,
            node_offsets=node_offsets,
            edge_sources=edge_sources,
            edge_targets=edge_targets,
            edge_types=edge_types,
            edge_offsets=edge_offsets,
            genes=Genes.concat(genes_batches),
            intervals=Intervals(starts=starts, ends=ends, strands=strands),
            contig_indices=contig_indices,
            archetypes=archetypes,
            completeness=completeness,
            scores=scores,
            is_tandem_split=is_tandem_split,
            names=np.array(names_list, dtype=str),
        )

    def canonicalize_strands(self) -> Self:
        """Orient all loci in the batch into canonical 5' -> 3' operon order (`strand = +1`)."""
        if len(self) == 0 or bool(np.all(self.intervals.strands >= 0)):
            return self
        return self.from_list([loc.canonicalize_strand() for loc in self])

    def heal_mobile_elements(self, remove_nodes: bool = True) -> Self:
        """Excise mobile genetic element (`TRANSPOSASE`, `INTEGRASE`) insertions across all loci in the batch."""
        if not remove_nodes or len(self) == 0 or len(self.node_types) == 0:
            return self
        roles = self.roles
        has_mge = bool(np.any((roles == int(GeneRole.TRANSPOSASE)) | (roles == int(GeneRole.INTEGRASE))))
        if not has_mge:
            return self
        return self.from_list([loc.heal_mobile_elements(remove_nodes=True) for loc in self])

    def with_clusters(self, clusters: Clusters | np.ndarray) -> Self:
        """Return a new `Loci` batch with homology `cluster_id` bitfields updated from `Clusters` or an ID array."""
        c_ids = clusters.cluster_ids if isinstance(clusters, Clusters) else np.asarray(clusters, dtype=np.int32)
        if len(c_ids) != len(self.node_types):
            raise ValueError(
                f"Cluster IDs length ({len(c_ids)}) does not match total gene nodes ({len(self.node_types)})."
            )
        # In GeneNode, cluster_id = 0 means unclustered; shift 0-based Clusters IDs to 1-based if needed
        shifted = np.where(c_ids >= 0, c_ids + 1, 0).astype(np.uint32)
        cleared = self.node_types & ~np.uint32(GeneNode.CLUSTER_ID.mask)
        packed_cid = (shifted & np.uint32(GeneNode.CLUSTER_ID.max_val)) << np.uint32(GeneNode.CLUSTER_ID.shift)
        new_node_types = cleared | packed_cid
        return type(self)(
            node_types=new_node_types,
            node_offsets=self.node_offsets,
            edge_sources=self.edge_sources,
            edge_targets=self.edge_targets,
            edge_types=self.edge_types,
            edge_offsets=self.edge_offsets,
            genes=self.genes,
            intervals=self.intervals,
            contig_indices=self.contig_indices,
            archetypes=self.archetypes,
            completeness=self.completeness,
            scores=self.scores,
            is_tandem_split=self.is_tandem_split,
            names=self.names,
        )

    def cluster_genes(
        self,
        clusterer: Clusterer | None = None,
        *,
        by_role: bool = True,
        classifier: Any | None = None,
        embedding_layer: str = "conv_max",
        **kwargs: Any,
    ) -> Self:
        """Cluster translated locus proteins and return a new `Loci` batch with packed homology `CLUSTER_ID`s."""
        if len(self.node_types) == 0 or len(self.genes) != len(self.node_types):
            return self
        proteins = self.genes.translations
        if len(proteins) != len(self.node_types):
            return self
        engine = clusterer if clusterer is not None else Clusterer(**kwargs)
        roles = self.roles if by_role else None
        use_emb = classifier is not None or engine.embedding_weight > 0.0
        clf = classifier
        if use_emb and clf is None:
            from serogen.genes.cnn import GeneRoleClassifier

            clf = GeneRoleClassifier.pretrained()

        if len(proteins) > 500:
            seen_key_to_u: dict[tuple[int, bytes], int] = {}
            u_indices: list[int] = []
            inv_map = np.empty(len(proteins), dtype=np.int32)
            for g_i in range(len(proteins)):
                p_off = int(proteins.offsets[g_i])
                p_len = int(proteins.lengths[g_i])
                r_key = int(roles[g_i]) if roles is not None else 0
                key = (r_key, proteins.seqs[p_off : p_off + p_len].tobytes())
                u_idx = seen_key_to_u.get(key)
                if u_idx is None:
                    u_idx = len(u_indices)
                    seen_key_to_u[key] = u_idx
                    u_indices.append(g_i)
                inv_map[g_i] = u_idx
            u_idx_arr = np.asarray(u_indices, dtype=np.int32)
            u_prots = proteins[u_idx_arr]
            u_embs = clf.transform(u_prots, layer=embedding_layer) if clf is not None else None
            u_clusters = engine.cluster(
                u_prots,
                roles=roles[u_idx_arr] if roles is not None else None,
                embeddings=u_embs,
            )
            return self.with_clusters(u_clusters.cluster_ids[inv_map])
        embs = clf.transform(proteins, layer=embedding_layer) if clf is not None else None
        clusters = engine.cluster(proteins, roles=roles, embeddings=embs)
        return self.with_clusters(clusters)

    def as_pan_graph(
        self,
        node_mask: int | None = None,
        *,
        canonicalize_strands: bool = True,
        heal_mges: bool = False,
        cluster_genes: bool = False,
        clusterer: Clusterer | None = None,
    ) -> tuple[
        csr_array,
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
    ]:
        """Compress the `Loci` batch into a directed consensus pan-locus graph.

        Args:
            node_mask: Optional bitmask to apply to `GeneNode` values before merging
                (e.g., `GeneNode.mask("ROLE_ID", "CLUSTER_ID")`).
            canonicalize_strands: If True (default), orient `-1` strand loci into canonical
                5' -> 3' operon order prior to graph compression.
            heal_mges: If True, excise `TRANSPOSASE` and `INTEGRASE` insertions and heal
                synteny edges prior to graph compression.
            cluster_genes: If True (or if `clusterer` is provided), cluster translated gene
                proteins and pack homology `CLUSTER_ID`s into nodes before merging.
            clusterer: Optional pre-configured `Clusterer` instance to use when clustering proteins.

        Returns:
            A tuple of `(adj, pan_nodes, node_counts)`.
        """
        batch = self
        if heal_mges:
            batch = batch.heal_mobile_elements(remove_nodes=True)
        if canonicalize_strands:
            batch = batch.canonicalize_strands()
        if cluster_genes or clusterer is not None:
            batch = batch.cluster_genes(clusterer=clusterer)
        return super(Loci, batch).as_pan_graph(node_mask=node_mask)

    def match(
        self,
        query: Locus | Loci,
        *,
        top_k: int | None = None,
        min_score: float = 0.0,
        sort: bool = True,
        **matcher_kwargs: Any,
    ) -> Any:
        """Compare one or more `query` loci against this `Loci` batch acting as a reference database."""
        from serogen.loci.matcher import LocusMatcher

        matcher = LocusMatcher(self, **matcher_kwargs)
        return matcher.match(query, top_k=top_k, min_score=min_score, sort=sort)

    @overload
    def __getitem__(self, item: int | np.integer[Any] | str) -> Locus: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> Self: ...

    @overload
    def __getitem__(
        self, item: int | np.integer[Any] | str | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> Locus | Self: ...

    def __getitem__(
        self, item: int | np.integer[Any] | str | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> Locus | Self:
        if isinstance(item, str):
            matches = np.nonzero(self.names == item)[0]
            if len(matches) == 0:
                raise KeyError(f"Locus name {item!r} not found in Loci batch.")
            item = int(matches[0])

        if isinstance(item, (int, np.integer)):
            idx = int(item)
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Loci index {item} out of range [0, {len(self) - 1}]")

            m_start = int(self.node_offsets[idx])
            m_end = int(self.node_offsets[idx + 1])
            e_start = int(self.edge_offsets[idx])
            e_end = int(self.edge_offsets[idx + 1])

            nodes = tuple(GeneNode(int(x)) for x in self.node_types[m_start:m_end])
            edges = tuple(
                IntergenicEdge(
                    source=int(self.edge_sources[k]) - m_start,
                    target=int(self.edge_targets[k]) - m_start,
                    edge_type=IntergenicType(int(self.edge_types[k])),
                )
                for k in range(e_start, e_end)
            )
            sub_genes = self.genes[m_start:m_end] if len(self.genes) >= m_end else Genes.empty()
            return Locus(
                nodes=nodes,
                edges=edges,
                genes=sub_genes,
                interval=self.intervals[idx],
                contig_index=int(self.contig_indices[idx]) if len(self.contig_indices) > idx else 0,
                archetype=LocusArchetype(int(self.archetypes[idx]))
                if len(self.archetypes) > idx
                else LocusArchetype.UNKNOWN,
                completeness=float(self.completeness[idx]) if len(self.completeness) > idx else 0.0,
                score=float(self.scores[idx]) if len(self.scores) > idx else 0.0,
                is_tandem_split=bool(self.is_tandem_split[idx]) if len(self.is_tandem_split) > idx else False,
                name=str(self.names[idx]) if len(self.names) > idx else "",
            )

        kept_indices = self._normalize_indices(item)
        if len(kept_indices) == 0:
            return self.empty()

        new_nodes, new_node_offsets, new_sources, new_targets, new_types, new_edge_offsets = self._slice_soa_arrays(
            kept_indices
        )

        # Gather flat node-level `genes` matching the selected loci
        node_idx_slices = [
            np.arange(int(self.node_offsets[i]), int(self.node_offsets[i + 1]), dtype=np.int32) for i in kept_indices
        ]
        flat_node_indices = np.concatenate(node_idx_slices) if node_idx_slices else np.empty(0, dtype=np.int32)

        return type(self)(
            node_types=new_nodes,
            node_offsets=new_node_offsets,
            edge_sources=new_sources,
            edge_targets=new_targets,
            edge_types=new_types,
            edge_offsets=new_edge_offsets,
            genes=self.genes[flat_node_indices] if len(self.genes) > 0 else Genes.empty(),
            intervals=self.intervals[kept_indices],
            contig_indices=self.contig_indices[kept_indices],
            archetypes=self.archetypes[kept_indices],
            completeness=self.completeness[kept_indices],
            scores=self.scores[kept_indices],
            is_tandem_split=self.is_tandem_split[kept_indices],
            names=self.names[kept_indices],
        )
