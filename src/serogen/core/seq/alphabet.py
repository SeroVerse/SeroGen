r"""ASCII-based biological alphabets, translation tables, and genetic codes.

This module provides high-performance ASCII alphabet definitions ([`Alphabet`][serogen.core.seq.alphabet.Alphabet])
with pre-computed 256-byte lookup tables for instant $O(1)$ k-mer bit-packing and reduced alphabet
transformations, alongside translation tables ([`GeneticCode`][serogen.core.seq.alphabet.GeneticCode])
for vectorised parallel protein translation.

Classes:
    [`Alphabet`][serogen.core.seq.alphabet.Alphabet]: ASCII biological alphabet representation with dual LUTs.
    [`GeneticCode`][serogen.core.seq.alphabet.GeneticCode]: Translation table and genetic code representation.
    [`AlphabetError`][serogen.core.seq.alphabet.AlphabetError]: Base exception for alphabet operations.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self

import numpy as np
from numba import njit, prange


# Exceptions -----------------------------------------------------------------------------------------------------------
class AlphabetError(ValueError):
    r"""Raised when an alphabet definition is invalid or an operation is incompatible."""


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True, eq=False, init=False)
class Alphabet:
    r"""ASCII biological alphabet with pre-computed lookup tables and dynamic singleton caching.

    Provides pre-computed 256-byte lookup tables:
    - `kmer_lut`: Maps ASCII characters to dense $[0 \dots K-1]$ integers (with unknown characters mapped to $K$)
      for on-the-fly rolling k-mer and syncmer bit-packing in L1 cache.
    - `ascii_lut`: Maps ASCII characters to canonical uppercase / reduced group representative ASCII bytes.
    - `complement_lut`: 256-element byte translation table for nucleotide complementation.

    Attributes:
        symbols (bytes): Canonical uppercase ASCII symbols representing distinct alphabet states.
        complement (bytes | None): Complemented symbols corresponding to `symbols` (for nucleotide alphabets).
        bits_per_char (int): Exact bit width required to represent characters ($(\text{len}-1).\text{bit\_length}()$).
        default_k (int): Recommended default k-mer size for indexing.
        supports_canonical (bool): Whether reverse-complement canonicalization is valid.
        kmer_lut (np.ndarray): Read-only 256-byte array for $O(1)$ dense integer mapping.
        ascii_lut (np.ndarray): Read-only 256-byte array for $O(1)$ ASCII normalization / reduction.
        complement_lut (np.ndarray | None): Read-only 256-byte array for nucleotide complementation.
    """

    symbols: bytes
    complement: bytes | None = None
    aliases: dict[bytes, bytes] | None = None
    groups: dict[bytes, tuple[bytes, ...]] | None = None
    bits_per_char: int = 0
    default_k: int = 0
    supports_canonical: bool = False
    name: str | None = None
    kmer_lut: np.ndarray[tuple[int], np.dtype[np.uint8]] = field(
        default_factory=lambda: np.empty(0, dtype=np.uint8), repr=False
    )
    ascii_lut: np.ndarray[tuple[int], np.dtype[np.uint8]] = field(
        default_factory=lambda: np.empty(0, dtype=np.uint8), repr=False
    )
    complement_lut: np.ndarray[tuple[int], np.dtype[np.uint8]] = field(
        default_factory=lambda: np.empty(0, dtype=np.uint8), repr=False
    )

    _CACHE: ClassVar[dict[Any, Self]] = {}

    DNA: ClassVar[Alphabet]
    AMINO: ClassVar[Alphabet]
    MMSEQS12: ClassVar[Alphabet]

    def __new__(
        cls,
        symbols: bytes | str,
        complement: bytes | str | None = None,
        aliases: Mapping[bytes | str, bytes | str] | None = None,
        groups: Mapping[bytes | str, Iterable[bytes | str]] | None = None,
        default_k: int | None = None,
        supports_canonical: bool | None = None,
        name: str | None = None,
    ) -> Self:
        r"""Create or return a cached singleton instance of an Alphabet."""
        sym_bytes = (symbols.encode("ascii") if isinstance(symbols, str) else symbols).upper()
        comp_bytes = (
            (complement.encode("ascii") if isinstance(complement, str) else complement).upper()
            if complement is not None
            else None
        )

        norm_aliases: dict[bytes, bytes] | None = None
        if aliases:
            norm_aliases = {
                (k.encode("ascii") if isinstance(k, str) else k): (v.encode("ascii") if isinstance(v, str) else v)
                for k, v in aliases.items()
            }

        norm_groups: dict[bytes, tuple[bytes, ...]] | None = None
        if groups:
            norm_groups = {
                (k.encode("ascii") if isinstance(k, str) else k): tuple(
                    x.encode("ascii") if isinstance(x, str) else x for x in v
                )
                for k, v in groups.items()
            }

        bits_per_char = (len(sym_bytes) - 1).bit_length()
        resolved_k = default_k if default_k is not None else (21 if bits_per_char <= 2 else 7)
        resolved_canonical = supports_canonical if supports_canonical is not None else (comp_bytes is not None)

        cache_key = (
            sym_bytes,
            comp_bytes,
            tuple(sorted(norm_aliases.items())) if norm_aliases else None,
            tuple(sorted((k, tuple(v)) for k, v in norm_groups.items())) if norm_groups else None,
            resolved_k,
            resolved_canonical,
        )

        if cache_key in cls._CACHE:
            existing = cls._CACHE[cache_key]
            if name and not existing.name:
                object.__setattr__(existing, "name", name)
            return existing

        instance = object.__new__(cls)
        cls._CACHE[cache_key] = instance
        instance._init_alphabet(
            symbols=sym_bytes,
            complement=comp_bytes,
            aliases=norm_aliases,
            groups=norm_groups,
            default_k=resolved_k,
            supports_canonical=resolved_canonical,
            name=name,
        )
        return instance

    def _init_alphabet(
        self,
        symbols: bytes,
        complement: bytes | None,
        aliases: dict[bytes, bytes] | None,
        groups: dict[bytes, tuple[bytes, ...]] | None,
        default_k: int,
        supports_canonical: bool,
        name: str | None = None,
    ) -> None:
        r"""Internal initialization for building lookup tables and properties."""
        if not symbols.isascii():
            raise AlphabetError("Alphabet symbols must be valid ASCII characters.")

        n_syms = len(symbols)
        if len(set(symbols)) != n_syms:
            raise AlphabetError("Alphabet symbols must be unique.")

        # Bit depth
        bits_per_char = (n_syms - 1).bit_length()

        # 1. Build kmer_lut (ASCII byte -> 0..K-1 dense index; unknowns -> K)
        kmer_lut = np.full(256, n_syms, dtype=np.uint8)

        # 2. Build ascii_lut (ASCII byte -> representative ASCII uppercase byte)
        ascii_lut = np.full(256, ord("X"), dtype=np.uint8)

        if groups is not None:
            # Reduced alphabet groups mapping (e.g. MMseqs12)
            for idx, (rep, members) in enumerate(groups.items()):
                rep_byte = rep[0]
                for m in members:
                    m_byte = m[0]
                    kmer_lut[m_byte] = idx
                    kmer_lut[ord(chr(m_byte).lower())] = idx
                    ascii_lut[m_byte] = rep_byte
                    ascii_lut[ord(chr(m_byte).lower())] = rep_byte
        else:
            # Standard alphabet mapping
            for idx, b in enumerate(symbols):
                kmer_lut[b] = idx
                kmer_lut[ord(chr(b).lower())] = idx
                ascii_lut[b] = b
                ascii_lut[ord(chr(b).lower())] = b

        # Apply Aliases
        if aliases:
            for src, dst in aliases.items():
                src_b = src[0]
                dst_b = dst[0]
                target_kmer_idx = kmer_lut[dst_b]
                target_ascii = ascii_lut[dst_b]
                kmer_lut[src_b] = target_kmer_idx
                kmer_lut[ord(chr(src_b).lower())] = target_kmer_idx
                ascii_lut[src_b] = target_ascii
                ascii_lut[ord(chr(src_b).lower())] = target_ascii

        kmer_lut.flags.writeable = False
        ascii_lut.flags.writeable = False

        # 3. Build complement_lut
        if complement is not None:
            if len(complement) != n_syms:
                raise AlphabetError("Complement must have the exact same length as symbols.")
            comp_lut = np.full(256, ord("N"), dtype=np.uint8)
            comp_lut[ord("n")] = ord("n")
            for s, c in zip(symbols, complement, strict=False):
                comp_lut[s] = c
                comp_lut[ord(chr(s).lower())] = ord(chr(c).lower())
            if aliases:
                for src, dst in aliases.items():
                    src_b = src[0]
                    dst_b = dst[0]
                    comp_lut[src_b] = comp_lut[dst_b]
                    comp_lut[ord(chr(src_b).lower())] = comp_lut[ord(chr(dst_b).lower())]
        else:
            comp_lut = np.arange(256, dtype=np.uint8)
        comp_lut.flags.writeable = False

        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "complement", complement)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "bits_per_char", bits_per_char)
        object.__setattr__(self, "default_k", default_k)
        object.__setattr__(self, "supports_canonical", supports_canonical)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kmer_lut", kmer_lut)
        object.__setattr__(self, "ascii_lut", ascii_lut)
        object.__setattr__(self, "complement_lut", comp_lut)

    def __len__(self) -> int:
        r"""Return the number of primary symbols in the alphabet."""
        return len(self.symbols)

    def __contains__(self, item: bytes | str | int) -> bool:
        r"""Check if a character or byte value is a valid member of this alphabet."""
        if isinstance(item, int):
            return 0 <= item < 256 and self.kmer_lut[item] < len(self.symbols)
        if isinstance(item, str):
            if len(item) != 1:
                return False
            item_val = ord(item)
            return item_val < 256 and self.kmer_lut[item_val] < len(self.symbols)
        if isinstance(item, bytes):
            if len(item) != 1:
                return False
            return self.kmer_lut[item[0]] < len(self.symbols)
        return False

    def __repr__(self) -> str:
        r"""Return a clean string representation of the Alphabet."""
        if self.name:
            return f"Alphabet.{self.name}"
        return f"Alphabet({self.symbols!r}, bits={self.bits_per_char})"

    def __eq__(self, other: object) -> bool:
        r"""Check equality based on alphabet identity or specifications."""
        if self is other:
            return True
        if not isinstance(other, Alphabet):
            return False
        return (
            self.symbols == other.symbols
            and self.complement == other.complement
            and self.bits_per_char == other.bits_per_char
            and self.default_k == other.default_k
            and self.supports_canonical == other.supports_canonical
        )

    def __hash__(self) -> int:
        r"""Return hash for using Alphabet instances in sets or dictionary keys."""
        return hash((self.symbols, self.complement, self.bits_per_char, self.default_k, self.supports_canonical))

    def masker(self, k: int) -> tuple[int, int, type[np.uint32 | np.uint64]]:
        r"""Calculate k-mer bit-shift parameters and word mask for sequence hashing.

        Args:
            k (int): K-mer length.

        Returns:
            tuple[int, int, type[np.uint32 | np.uint64]]: Tuple of (bits_per_char, bitmask, integer_dtype).
        """
        bps = self.bits_per_char
        total_bits = k * bps
        if total_bits <= 32:
            dtype = np.uint32
        elif total_bits <= 64:
            dtype = np.uint64
        else:
            raise ValueError(f"K={k} is too large for 64-bit word packing with {self}")
        mask = (1 << (bps * (k - 1))) - 1
        return bps, mask, dtype

    def encode(
        self, seq: bytes | bytearray | memoryview | np.ndarray[tuple[int], np.dtype[np.uint8]]
    ) -> np.ndarray[tuple[int], np.dtype[np.uint8]]:
        r"""Encode ASCII sequence bytes into dense integer values [0..K-1].

        Args:
            seq (bytes | bytearray | memoryview | np.ndarray): Input sequence bytes or 1D array.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.uint8]]: Array of dense integer indices.
        """
        arr = seq if isinstance(seq, np.ndarray) else np.frombuffer(seq, dtype=np.uint8)
        if arr.ndim == 1 and arr.dtype == np.uint8:
            if len(arr) >= 65_536:
                return _lut_map_parallel(arr, self.kmer_lut)
            return _lut_map_serial(arr, self.kmer_lut)
        return self.kmer_lut[arr]

    def reduce(
        self, seq: bytes | bytearray | memoryview | np.ndarray[tuple[int], np.dtype[np.uint8]]
    ) -> bytes | np.ndarray[tuple[int], np.dtype[np.uint8]]:
        r"""Perform vectorized reduction of ASCII sequence bytes into reduced representative ASCII characters.

        Args:
            seq (bytes | bytearray | memoryview | np.ndarray): Input sequence bytes or 1D array.

        Returns:
            bytes | np.ndarray[tuple[int], np.dtype[np.uint8]]: Reduced sequence(s) in ASCII format.
        """
        if isinstance(seq, np.ndarray):
            if seq.ndim == 1 and seq.dtype == np.uint8:
                if len(seq) >= 65_536:
                    return _lut_map_parallel(seq, self.ascii_lut)
                return _lut_map_serial(seq, self.ascii_lut)
            return self.ascii_lut[seq]
        arr = np.frombuffer(seq, dtype=np.uint8)
        if len(arr) >= 65_536:
            return _lut_map_parallel(arr, self.ascii_lut).tobytes()
        return _lut_map_serial(arr, self.ascii_lut).tobytes()


# Predefined Alphabet Singletons ---------------------------------------------------------------------------------------
Alphabet.DNA = Alphabet(
    symbols=b"ACGT",
    complement=b"TGCA",
    aliases={b"U": b"T"},
    default_k=21,
    supports_canonical=True,
    name="DNA",
)

Alphabet.AMINO = Alphabet(
    symbols=b"ACDEFGHIKLMNPQRSTVWYUO",
    aliases={b"X": b"A", b"B": b"D", b"Z": b"E", b"J": b"L"},
    default_k=7,
    supports_canonical=False,
    name="AMINO",
)

Alphabet.MMSEQS12 = Alphabet(
    symbols=b"ALIKENFCGHPW",
    groups={
        b"A": (b"A", b"S", b"T"),
        b"L": (b"L", b"M"),
        b"I": (b"I", b"V"),
        b"K": (b"K", b"R"),
        b"E": (b"E", b"Q"),
        b"N": (b"N", b"D"),
        b"F": (b"F", b"Y"),
        b"C": (b"C",),
        b"G": (b"G",),
        b"H": (b"H",),
        b"P": (b"P",),
        b"W": (b"W",),
    },
    aliases={
        b"X": b"A",
        b"B": b"N",
        b"Z": b"E",
        b"J": b"L",
        b"U": b"C",
        b"O": b"K",
    },
    default_k=7,
    supports_canonical=False,
    name="MMSEQS12",
)


# Genetic Code Class ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True, eq=False, init=False)
class GeneticCode:
    r"""NCBI Translation Table representation for codon translation and complementation.

    Attributes:
        table_id (int): NCBI translation table integer identifier.
        name (str): Descriptive name of the genetic code.
        start_codons (frozenset[bytes]): Valid 3-base start codon byte triplets.
        stop_codons (frozenset[bytes]): Valid 3-base stop codon byte triplets.
    """

    table_id: int
    name: str
    start_codons: frozenset[bytes]
    stop_codons: frozenset[bytes]
    _char_map: np.ndarray[tuple[int], np.dtype[np.uint8]] = field(
        default_factory=lambda: np.empty(0, dtype=np.uint8), repr=False
    )
    _codon_map: np.ndarray[tuple[int], np.dtype[np.uint8]] = field(
        default_factory=lambda: np.empty(0, dtype=np.uint8), repr=False
    )
    _starts_mask: np.ndarray[tuple[int], np.dtype[np.bool_]] = field(
        default_factory=lambda: np.empty(0, dtype=np.bool_), repr=False
    )
    _stops_mask: np.ndarray[tuple[int], np.dtype[np.bool_]] = field(
        default_factory=lambda: np.empty(0, dtype=np.bool_), repr=False
    )
    start_mask_64: np.ndarray[tuple[int], np.dtype[np.uint8]] = field(
        default_factory=lambda: np.empty(0, dtype=np.uint8), repr=False
    )
    stop_mask_64: np.ndarray[tuple[int], np.dtype[np.uint8]] = field(
        default_factory=lambda: np.empty(0, dtype=np.uint8), repr=False
    )
    rev_start_mask_64: np.ndarray[tuple[int], np.dtype[np.uint8]] = field(
        default_factory=lambda: np.empty(0, dtype=np.uint8), repr=False
    )
    rev_stop_mask_64: np.ndarray[tuple[int], np.dtype[np.uint8]] = field(
        default_factory=lambda: np.empty(0, dtype=np.uint8), repr=False
    )
    start_log_priors_64: np.ndarray[tuple[int], np.dtype[np.float64]] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64), repr=False
    )
    rev_start_log_priors_64: np.ndarray[tuple[int], np.dtype[np.float64]] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64), repr=False
    )
    stop_log_priors_64: np.ndarray[tuple[int], np.dtype[np.float64]] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64), repr=False
    )
    rev_stop_log_priors_64: np.ndarray[tuple[int], np.dtype[np.float64]] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64), repr=False
    )

    _CACHE: ClassVar[dict[Any, Self]] = {}
    standard: ClassVar[GeneticCode]
    bacterial: ClassVar[GeneticCode]

    def __new__(
        cls,
        table_id: int,
        name: str,
        mapping: Mapping[bytes, bytes],
        start_codons: Iterable[bytes],
        stop_codons: Iterable[bytes],
    ) -> Self:
        r"""Create or return a cached singleton instance of a GeneticCode."""
        norm_starts = frozenset(s.upper() for s in start_codons)
        norm_stops = frozenset(s.upper() for s in stop_codons)
        cache_key = (table_id, name, tuple(sorted(mapping.items())), norm_starts, norm_stops)

        if cache_key in cls._CACHE:
            return cls._CACHE[cache_key]

        instance = object.__new__(cls)
        cls._CACHE[cache_key] = instance
        instance._init_code(table_id, name, mapping, norm_starts, norm_stops)
        return instance

    def _init_code(
        self,
        table_id: int,
        name: str,
        mapping: Mapping[bytes, bytes],
        start_codons: frozenset[bytes],
        stop_codons: frozenset[bytes],
    ) -> None:
        r"""Internal initialization for building translation tables."""
        object.__setattr__(self, "table_id", table_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "start_codons", start_codons)
        object.__setattr__(self, "stop_codons", stop_codons)

        # Reuse Alphabet.DNA.kmer_lut directly (A=0, C=1, G=2, T/U=3, others=4)
        char_map = Alphabet.DNA.kmer_lut
        object.__setattr__(self, "_char_map", char_map)

        # Build _codon_map: 5*5*5 = 125 entries (88 = ord('X'))
        codon_map = np.full(125, ord("X"), dtype=np.uint8)
        starts_mask = np.zeros(125, dtype=bool)
        stops_mask = np.zeros(125, dtype=bool)

        for codon, aa in mapping.items():
            idx = int(char_map[codon[0]]) * 25 + int(char_map[codon[1]]) * 5 + int(char_map[codon[2]])
            codon_map[idx] = aa[0]

        for sc in start_codons:
            idx = int(char_map[sc[0]]) * 25 + int(char_map[sc[1]]) * 5 + int(char_map[sc[2]])
            starts_mask[idx] = True

        for st in stop_codons:
            idx = int(char_map[st[0]]) * 25 + int(char_map[st[1]]) * 5 + int(char_map[st[2]])
            stops_mask[idx] = True

        # Build 64-entry (4x4x4) trinucleotide LUTs for Viterbi & gene extraction
        start_mask_64 = np.zeros(64, dtype=np.uint8)
        stop_mask_64 = np.zeros(64, dtype=np.uint8)
        rev_start_mask_64 = np.zeros(64, dtype=np.uint8)
        rev_stop_mask_64 = np.zeros(64, dtype=np.uint8)
        start_log_priors_64 = np.full(64, -1e9, dtype=np.float64)
        rev_start_log_priors_64 = np.full(64, -1e9, dtype=np.float64)
        stop_log_priors_64 = np.full(64, -1e9, dtype=np.float64)
        rev_stop_log_priors_64 = np.full(64, -1e9, dtype=np.float64)

        comp_lut = Alphabet.DNA.complement_lut
        default_start_priors = {
            b"ATG": float(np.log(0.83)),
            b"GTG": float(np.log(0.10)),
            b"TTG": float(np.log(0.07)),
        }
        default_stop_priors = {
            b"TAA": float(np.log(0.54)),
            b"TAG": float(np.log(0.16)),
            b"TGA": float(np.log(0.30)),
        }

        # Primary Viterbi start codons (ATG/GTG/TTG subset of start_codons, or all start_codons if none of those)
        primary_starts = {sc for sc in start_codons if sc in default_start_priors}
        if not primary_starts:
            primary_starts = set(start_codons)

        for sc in primary_starts:
            c0, c1, c2 = int(char_map[sc[0]]), int(char_map[sc[1]]), int(char_map[sc[2]])
            if c0 < 4 and c1 < 4 and c2 < 4:
                idx64 = c0 * 16 + c1 * 4 + c2
                start_mask_64[idx64] = 1
                lp = default_start_priors.get(sc, float(np.log(0.05)))
                start_log_priors_64[idx64] = lp

                # Reverse-complement codon on forward strand: (comp(sc[2]), comp(sc[1]), comp(sc[0]))
                rc0 = int(char_map[comp_lut[sc[2]]])
                rc1 = int(char_map[comp_lut[sc[1]]])
                rc2 = int(char_map[comp_lut[sc[0]]])
                r_idx64 = rc0 * 16 + rc1 * 4 + rc2
                rev_start_mask_64[r_idx64] = 1
                rev_start_log_priors_64[r_idx64] = lp

        for st in stop_codons:
            c0, c1, c2 = int(char_map[st[0]]), int(char_map[st[1]]), int(char_map[st[2]])
            if c0 < 4 and c1 < 4 and c2 < 4:
                idx64 = c0 * 16 + c1 * 4 + c2
                stop_mask_64[idx64] = 1
                lp = default_stop_priors.get(st, float(np.log(1.0 / max(len(stop_codons), 1))))
                stop_log_priors_64[idx64] = lp

                rc0 = int(char_map[comp_lut[st[2]]])
                rc1 = int(char_map[comp_lut[st[1]]])
                rc2 = int(char_map[comp_lut[st[0]]])
                r_idx64 = rc0 * 16 + rc1 * 4 + rc2
                rev_stop_mask_64[r_idx64] = 1
                rev_stop_log_priors_64[r_idx64] = lp

        for arr in (
            codon_map,
            starts_mask,
            stops_mask,
            start_mask_64,
            stop_mask_64,
            rev_start_mask_64,
            rev_stop_mask_64,
            start_log_priors_64,
            rev_start_log_priors_64,
            stop_log_priors_64,
            rev_stop_log_priors_64,
        ):
            arr.flags.writeable = False

        object.__setattr__(self, "_codon_map", codon_map)
        object.__setattr__(self, "_starts_mask", starts_mask)
        object.__setattr__(self, "_stops_mask", stops_mask)
        object.__setattr__(self, "start_mask_64", start_mask_64)
        object.__setattr__(self, "stop_mask_64", stop_mask_64)
        object.__setattr__(self, "rev_start_mask_64", rev_start_mask_64)
        object.__setattr__(self, "rev_stop_mask_64", rev_stop_mask_64)
        object.__setattr__(self, "start_log_priors_64", start_log_priors_64)
        object.__setattr__(self, "rev_start_log_priors_64", rev_start_log_priors_64)
        object.__setattr__(self, "stop_log_priors_64", stop_log_priors_64)
        object.__setattr__(self, "rev_stop_log_priors_64", rev_stop_log_priors_64)

    @classmethod
    def from_table(cls, table: int | str | GeneticCode) -> GeneticCode:
        r"""Resolve a `GeneticCode` instance by NCBI translation table ID (`1, 11`) or name."""
        if isinstance(table, GeneticCode):
            return table
        if isinstance(table, int):
            by_id = {
                1: cls.standard,
                11: cls.bacterial,
            }
            if table in by_id:
                return by_id[table]
            raise AlphabetError(f"Unknown NCBI translation table ID: {table}. Supported: {sorted(by_id)}.")
        key = table.strip().lower().replace("-", "_").replace(" ", "_")
        by_name = {
            "standard": cls.standard,
            "1": cls.standard,
            "bacterial": cls.bacterial,
            "11": cls.bacterial,
        }
        if key in by_name:
            return by_name[key]
        raise AlphabetError(f"Unknown genetic code name: {table!r}.")

    def __repr__(self) -> str:
        r"""Return a clean string representation of the GeneticCode."""
        return f"GeneticCode(id={self.table_id}, name={self.name!r})"

    def __eq__(self, other: object) -> bool:
        r"""Check equality based on genetic code table identity or parameters."""
        if self is other:
            return True
        if not isinstance(other, GeneticCode):
            return False
        return (
            self.table_id == other.table_id
            and self.name == other.name
            and self.start_codons == other.start_codons
            and self.stop_codons == other.stop_codons
        )

    def __hash__(self) -> int:
        r"""Return hash based on table ID, name, and start/stop codons."""
        return hash((self.table_id, self.name, self.start_codons, self.stop_codons))

    def is_coding(self, seq: bytes) -> bool:
        r"""Check if a nucleotide sequence starts with a valid start codon and ends with a stop codon.

        Args:
            seq (bytes): Nucleotide sequence to check.

        Returns:
            bool: True if length >= 3 and starts with a valid start codon and ends with a stop codon.
        """
        if len(seq) < 3:
            return False
        return seq[:3].upper() in self.start_codons and seq[-3:].upper() in self.stop_codons

    def translate(
        self,
        seq: bytes | bytearray | memoryview | np.ndarray[tuple[int], np.dtype[np.uint8]],
        frame: int = 0,
        to_stop: bool = False,
    ) -> np.ndarray[tuple[int], np.dtype[np.uint8]]:
        r"""Translate a single nucleotide sequence into an ASCII protein sequence.

        Args:
            seq (bytes | bytearray | memoryview | np.ndarray): Input nucleotide sequence.
            frame (int): Reading frame offset (0, 1, or 2). Defaults to 0.
            to_stop (bool): If True, truncates translation at the first stop codon (*). Defaults to False.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.uint8]]: Translated protein sequence in ASCII format.
        """
        if not isinstance(seq, np.ndarray):
            arr = np.ascontiguousarray(np.frombuffer(seq, np.uint8))
        else:
            arr = np.ascontiguousarray(seq)

        if len(arr) < 3:
            return np.empty(0, dtype=np.uint8)

        return _translate_kernel(arr, frame, self._char_map, self._codon_map, to_stop)

    def translate_ragged(
        self,
        seqs: np.ndarray[tuple[int], np.dtype[np.uint8]],
        offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
        lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
        frames: np.ndarray[tuple[int], np.dtype[np.int8]] | None = None,
        to_stop: bool = False,
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.uint8]],
        np.ndarray[tuple[int], np.dtype[np.int32]],
        np.ndarray[tuple[int], np.dtype[np.int32]],
    ]:
        r"""Translate a ragged batch of nucleotide sequences in parallel.

        Args:
            seqs (np.ndarray[tuple[int], np.dtype[np.uint8]]): Concatenated nucleotide sequence bytes.
            offsets (np.ndarray[tuple[int], np.dtype[np.int32]]): Start offsets per sequence.
            lengths (np.ndarray[tuple[int], np.dtype[np.int32]]): Sequence lengths.
            frames (np.ndarray[tuple[int], np.dtype[np.int8]] | None): Reading frame offsets (0, 1, or 2) per sequence.
            to_stop (bool): If True, truncates translation at the first stop codon (*). Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of (out_seqs, out_offsets, out_lengths) arrays.
        """
        n_seqs = len(offsets)
        if n_seqs == 0:
            return (
                np.empty(0, dtype=np.uint8),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )

        return _translate_ragged_kernel(
            seqs,
            offsets,
            lengths,
            frames,
            self._char_map,
            self._codon_map,
            to_stop,
        )


# Initialize Genetic Code Singletons (Tables 1, 11) --------------------------------------------------------------------
_STANDARD_CODON_MAPPING: dict[bytes, bytes] = {
    b"TTT": b"F",
    b"TTC": b"F",
    b"TTA": b"L",
    b"TTG": b"L",
    b"TCT": b"S",
    b"TCC": b"S",
    b"TCA": b"S",
    b"TCG": b"S",
    b"TAT": b"Y",
    b"TAC": b"Y",
    b"TAA": b"*",
    b"TAG": b"*",
    b"TGT": b"C",
    b"TGC": b"C",
    b"TGA": b"*",
    b"TGG": b"W",
    b"CTT": b"L",
    b"CTC": b"L",
    b"CTA": b"L",
    b"CTG": b"L",
    b"CCT": b"P",
    b"CCC": b"P",
    b"CCA": b"P",
    b"CCG": b"P",
    b"CAT": b"H",
    b"CAC": b"H",
    b"CAA": b"Q",
    b"CAG": b"Q",
    b"CGT": b"R",
    b"CGC": b"R",
    b"CGA": b"R",
    b"CGG": b"R",
    b"ATT": b"I",
    b"ATC": b"I",
    b"ATA": b"I",
    b"ATG": b"M",
    b"ACT": b"T",
    b"ACC": b"T",
    b"ACA": b"T",
    b"ACG": b"T",
    b"AAT": b"N",
    b"AAC": b"N",
    b"AAA": b"K",
    b"AAG": b"K",
    b"AGT": b"S",
    b"AGC": b"S",
    b"AGA": b"R",
    b"AGG": b"R",
    b"GTT": b"V",
    b"GTC": b"V",
    b"GTA": b"V",
    b"GTG": b"V",
    b"GCT": b"A",
    b"GCC": b"A",
    b"GCA": b"A",
    b"GCG": b"A",
    b"GAT": b"D",
    b"GAC": b"D",
    b"GAA": b"E",
    b"GAG": b"E",
    b"GGT": b"G",
    b"GGC": b"G",
    b"GGA": b"G",
    b"GGG": b"G",
}

GeneticCode.standard = GeneticCode(
    table_id=1,
    name="The Standard Code",
    mapping=_STANDARD_CODON_MAPPING,
    start_codons=(b"TTG", b"CTG", b"ATG"),
    stop_codons=(b"TAA", b"TAG", b"TGA"),
)

GeneticCode.bacterial = GeneticCode(
    table_id=11,
    name="Bacterial, Archaeal, and Plant Plastid",
    mapping=_STANDARD_CODON_MAPPING,
    start_codons=(b"TTG", b"CTG", b"ATT", b"ATC", b"ATA", b"ATG", b"GTG"),
    stop_codons=(b"TAA", b"TAG", b"TGA"),
)


# Numba Kernels --------------------------------------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def _lut_map_serial(
    arr: np.ndarray[tuple[int], np.dtype[np.uint8]],
    lut: np.ndarray[tuple[int], np.dtype[np.uint8]],
) -> np.ndarray[tuple[int], np.dtype[np.uint8]]:
    r"""Map a 1D `uint8` array through a 256-byte lookup table on a single thread."""
    n = len(arr)
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        out[i] = lut[arr[i]]
    return out


@njit(parallel=True, cache=True, nogil=True)
def _lut_map_parallel(
    arr: np.ndarray[tuple[int], np.dtype[np.uint8]],
    lut: np.ndarray[tuple[int], np.dtype[np.uint8]],
) -> np.ndarray[tuple[int], np.dtype[np.uint8]]:
    r"""Map a 1D `uint8` array through a 256-byte lookup table in parallel."""
    n = len(arr)
    out = np.empty(n, dtype=np.uint8)
    for i in prange(n):  # type: ignore[not-iterable]
        out[i] = lut[arr[i]]
    return out


@njit(cache=True, nogil=True)
def _translate_kernel(
    seq: np.ndarray[tuple[int], np.dtype[np.uint8]],
    frame: int,
    char_map: np.ndarray[tuple[int], np.dtype[np.uint8]],
    codon_map: np.ndarray[tuple[int], np.dtype[np.uint8]],
    to_stop: bool,
) -> np.ndarray[tuple[int], np.dtype[np.uint8]]:
    r"""Translate a single sequence array starting from a given reading frame."""
    n_codons = (len(seq) - frame) // 3
    if n_codons <= 0:
        return np.empty(0, dtype=np.uint8)

    max_len = n_codons
    out = np.empty(max_len, dtype=np.uint8)
    out_idx = 0

    for i in range(n_codons):
        pos = frame + i * 3
        c1 = char_map[seq[pos]]
        c2 = char_map[seq[pos + 1]]
        c3 = char_map[seq[pos + 2]]
        codon_idx = c1 * 25 + c2 * 5 + c3
        aa = codon_map[codon_idx]
        if to_stop and aa == 42:  # 42 = ord('*')
            break
        out[out_idx] = aa
        out_idx += 1

    return out[:out_idx].copy()


@njit(parallel=True, cache=True, nogil=True)
def _translate_ragged_kernel(
    seqs: np.ndarray[tuple[int], np.dtype[np.uint8]],
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    frames: np.ndarray[tuple[int], np.dtype[np.int8]] | None,
    char_map: np.ndarray[tuple[int], np.dtype[np.uint8]],
    codon_map: np.ndarray[tuple[int], np.dtype[np.uint8]],
    to_stop: bool,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.uint8]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
]:
    r"""Parallel Numba kernel translating a batch of ragged sequences without duplicate codon decoding."""
    n_seqs = len(offsets)
    out_lengths = np.empty(n_seqs, dtype=np.int32)
    has_frames = frames is not None

    if not to_stop:
        out_offsets = np.empty(n_seqs, dtype=np.int32)
        total_len = 0
        for i in range(n_seqs):
            frame = frames[i] if has_frames else 0
            n_codons = (lengths[i] - frame) // 3
            nc = n_codons if n_codons > 0 else 0
            out_lengths[i] = nc
            out_offsets[i] = total_len
            total_len += nc

        out_seqs = np.empty(total_len, dtype=np.uint8)
        for i in prange(n_seqs):  # type: ignore[not-iterable]
            n_codons = out_lengths[i]
            if n_codons == 0:
                continue
            frame = frames[i] if has_frames else 0
            in_s = offsets[i] + frame
            out_s = out_offsets[i]
            for c in range(n_codons):
                pos = in_s + c * 3
                c1 = char_map[seqs[pos]]
                c2 = char_map[seqs[pos + 1]]
                c3 = char_map[seqs[pos + 2]]
                out_seqs[out_s + c] = codon_map[c1 * 25 + c2 * 5 + c3]
        return out_seqs, out_offsets, out_lengths

    # When to_stop=True, translate each sequence once into an upper-bound buffer and compact slices
    max_offsets = np.empty(n_seqs, dtype=np.int32)
    max_total = 0
    for i in range(n_seqs):
        frame = frames[i] if has_frames else 0
        n_codons = (lengths[i] - frame) // 3
        max_offsets[i] = max_total
        if n_codons > 0:
            max_total += n_codons

    tmp_seqs = np.empty(max_total, dtype=np.uint8)
    for i in prange(n_seqs):  # type: ignore[not-iterable]
        frame = frames[i] if has_frames else 0
        n_codons = (lengths[i] - frame) // 3
        if n_codons <= 0:
            out_lengths[i] = 0
            continue
        in_s = offsets[i] + frame
        tmp_s = max_offsets[i]
        count = 0
        for c in range(n_codons):
            pos = in_s + c * 3
            c1 = char_map[seqs[pos]]
            c2 = char_map[seqs[pos + 1]]
            c3 = char_map[seqs[pos + 2]]
            aa = codon_map[c1 * 25 + c2 * 5 + c3]
            if aa == 42:  # ord('*')
                break
            tmp_seqs[tmp_s + count] = aa
            count += 1
        out_lengths[i] = count

    out_offsets = np.empty(n_seqs, dtype=np.int32)
    total_len = 0
    for i in range(n_seqs):
        out_offsets[i] = total_len
        total_len += out_lengths[i]

    out_seqs = np.empty(total_len, dtype=np.uint8)
    for i in prange(n_seqs):  # type: ignore[not-iterable]
        count = out_lengths[i]
        if count == 0:
            continue
        out_s = out_offsets[i]
        tmp_s = max_offsets[i]
        out_seqs[out_s : out_s + count] = tmp_seqs[tmp_s : tmp_s + count]

    return out_seqs, out_offsets, out_lengths
