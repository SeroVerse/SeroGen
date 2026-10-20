# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "altair>=5.5.0",
#     "cassetter>=0.1.0",
#     "httpx2[http2]>=2.13.0",
#     "numpy>=2.4.4",
#     "scipy>=1.17.1",
#     "numba>=0.65.0",
#     "serogen",
# ]
#
# [tool.uv.sources]
# serogen = { path = "..", editable = true }
# ///

"""Sandboxed Marimo notebook benchmarking S. pneumoniae cps reference database construction and whole-genome serotyping."""

import marimo

__generated_with = "0.11.0"
app = marimo.App(
    width="full",
    app_title="SeroGen — S. pneumoniae Serotype Reference Database & Whole-Genome Benchmark",
)


@app.cell
def _():
    import asyncio
    import gzip
    import os
    import re
    import time
    import xml.etree.ElementTree as ET
    from dataclasses import dataclass
    from pathlib import Path

    import altair as alt
    import httpx2
    import marimo as mo
    import numpy as np
    from cassetter import Cassetter

    from serogen.core.graph import LayoutEngine
    from serogen.core.io.fasta import FastaReader
    from serogen.core.seq.clustering import Clusterer
    from serogen.core.seq.seq import Sequence, Sequences
    from serogen.genes import GeneRole, GeneRoleClassifier
    from serogen.loci import (
        BENIGN_VARIANT_MASK,
        ArchetypeCategory,
        GeneNode,
        Loci,
        Locus,
        LocusArchetype,
        LocusMatcher,
        LocusPredictor,
        MatchType,
        SerotypeDatabase,
    )

    alt.data_transformers.disable_max_rows()

    return (
        ArchetypeCategory,
        BENIGN_VARIANT_MASK,
        Cassetter,
        Clusterer,
        ET,
        FastaReader,
        GeneNode,
        GeneRole,
        GeneRoleClassifier,
        LayoutEngine,
        Loci,
        Locus,
        LocusArchetype,
        LocusMatcher,
        LocusPredictor,
        MatchType,
        Path,
        Sequence,
        Sequences,
        SerotypeDatabase,
        alt,
        asyncio,
        dataclass,
        gzip,
        httpx2,
        mo,
        np,
        os,
        re,
        time,
    )


@app.cell
def _(dataclass):
    # ------------------------------------------------------------------------------------------------------------------
    # Immutable Reference & Whole-Genome Benchmark Specifications
    # ------------------------------------------------------------------------------------------------------------------
    @dataclass(frozen=True, slots=True)
    class CpsReferenceSpec:
        """Metadata specification for an NCBI S. pneumoniae reference cps locus sequence."""

        serotype: str
        serogroup: str
        accession: str
        strain: str
        expected_mechanism: str

    @dataclass(frozen=True, slots=True)
    class WgsStrainSpec:
        """Metadata specification for a complete closed S. pneumoniae RefSeq chromosome."""

        strain: str
        true_serotype: str
        serogroup: str
        accession: str
        refseq_assembly: str
        expected_mechanism: str
        is_negative_control: bool = False

    # Curated 18-serotype reference panel spanning major PCV vaccine serotypes, cross-reacting serogroups
    # (6A/6B/6C/6D, 19A/19F, 23A/23B/23F), and both Wzy-dependent (CPS_WZY) & Synthase-dependent (CPS_SYNTHASE) capsules
    CURATED_CPS_SPECS: tuple[CpsReferenceSpec, ...] = (
        CpsReferenceSpec("1", "1", "CR931632.1", "519/43", "CPS_WZY"),
        CpsReferenceSpec("2", "2", "CR931633.1", "pn2L", "CPS_WZY"),
        CpsReferenceSpec("3", "3", "CR931634.2", "524/62", "CPS_SYNTHASE"),
        CpsReferenceSpec("4", "4", "CR931635.1", "600/62", "CPS_WZY"),
        CpsReferenceSpec("6A", "6", "CR931638.1", "34351 (Rodriques)", "CPS_WZY"),
        CpsReferenceSpec("6B", "6", "CR931639.1", "2616/39", "CPS_WZY"),
        CpsReferenceSpec("6C", "6", "EF538714.1", "CHPA388", "CPS_WZY"),
        CpsReferenceSpec("6D", "6", "HM171374.1", "MNZ21", "CPS_WZY"),
        CpsReferenceSpec("9V", "9", "CR931648.1", "980/68", "CPS_WZY"),
        CpsReferenceSpec("14", "14", "CR931662.1", "34359", "CPS_WZY"),
        CpsReferenceSpec("18C", "18", "CR931673.1", "4593/40", "CPS_WZY"),
        CpsReferenceSpec("19A", "19", "CR931675.1", "141/68", "CPS_WZY"),
        CpsReferenceSpec("19F", "19", "CR931678.1", "485/61", "CPS_WZY"),
        CpsReferenceSpec("23A", "23", "CR931683.1", "1196/45", "CPS_WZY"),
        CpsReferenceSpec("23B", "23", "CR931684.1", "1039/41", "CPS_WZY"),
        CpsReferenceSpec("23F", "23", "CR931685.1", "Dr. Melchior", "CPS_WZY"),
        CpsReferenceSpec("33F", "33", "CR931702.1", "3084/37", "CPS_WZY"),
        CpsReferenceSpec("37", "37", "CR931709.1", "264/73 (dexB-aliA)", "CPS_WZY"),
    )

    # Complete closed chromosome validation panel (10 encapsulated reference strains + 1 unencapsulated R6 control)
    WGS_VALIDATION_SPECS: tuple[WgsStrainSpec, ...] = (
        WgsStrainSpec("NCTC 7465", "1", "1", "LN831051.1", "GCF_001457635.1", "CPS_WZY"),
        WgsStrainSpec("D39", "2", "2", "CP000410.1", "GCF_000014365.1", "CPS_WZY"),
        WgsStrainSpec("OXC141", "3", "3", "FQ312027.1", "GCF_000210975.1", "CPS_SYNTHASE"),
        WgsStrainSpec("SPN994038", "3", "3", "FQ312041.2", "GCF_000211035.1", "CPS_SYNTHASE"),
        WgsStrainSpec("TIGR4", "4", "4", "AE005672.3", "GCF_000006885.1", "CPS_WZY"),
        WgsStrainSpec("670-6B", "6B", "6", "CP002176.1", "GCF_000147095.1", "CPS_WZY"),
        WgsStrainSpec("CGSP14", "14", "14", "CP001033.1", "GCF_000019825.1", "CPS_WZY"),
        WgsStrainSpec("Hungary19A-6", "19A", "19", "CP000936.1", "GCF_000019265.1", "CPS_WZY"),
        WgsStrainSpec("ATCC 49619", "19F", "19", "AP018938.1", "GCF_003966485.1", "CPS_WZY"),
        WgsStrainSpec("ATCC 700669", "23F", "23", "FM211187.1", "GCF_000026665.1", "CPS_WZY"),
        WgsStrainSpec(
            "R6 (Rough)",
            "None",
            "None",
            "AE007317.1",
            "GCF_000007045.1",
            "Unencapsulated",
            is_negative_control=True,
        ),
    )

    return CURATED_CPS_SPECS, CpsReferenceSpec, WGS_VALIDATION_SPECS, WgsStrainSpec


@app.cell
def _(
    ArchetypeCategory,
    BENIGN_VARIANT_MASK,
    Cassetter,
    CpsReferenceSpec,
    ET,
    FastaReader,
    LayoutEngine,
    Loci,
    Locus,
    LocusMatcher,
    LocusPredictor,
    MatchType,
    Path,
    Sequence,
    Sequences,
    SerotypeDatabase,
    WgsStrainSpec,
    asyncio,
    dataclass,
    gzip,
    httpx2,
    np,
    os,
    re,
    time,
):

    # ------------------------------------------------------------------------------------------------------------------
    # OOP Async HTTP/2 NCBI Downloader, Reference Database Builder & 4-Stage Validation Engine
    # ------------------------------------------------------------------------------------------------------------------
    class NcbiAsyncFetcher:
        """High-throughput asynchronous HTTP/2 NCBI E-utilities sequence fetcher powered by `httpx2` and `cassetter`."""

        EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

        def __init__(
            self,
            api_key: str | None = None,
            *,
            cassette_dir: Path | None = None,
            cassette_name: str = "spneumo_ncbi.toml",
            record_mode: str = "new_episodes",
            max_concurrency: int = 6,
            timeout: float = 90.0,
            max_retries: int = 3,
        ) -> None:
            self.api_key = api_key if api_key is not None else self.resolve_api_key()
            self.cassette_dir = (
                cassette_dir if cassette_dir is not None else (Path(__file__).resolve().parent / "cassettes")
            )
            self.cassette_dir.mkdir(parents=True, exist_ok=True)
            self.cassette_name = cassette_name
            self.recorder = Cassetter(
                cassette_library_dir=str(self.cassette_dir),
                record_mode=record_mode,
            )
            self.max_concurrency = max_concurrency
            self.timeout = timeout
            self.max_retries = max_retries

        @staticmethod
        def resolve_api_key() -> str | None:
            """Resolve `NCBI_API_KEY` from environment, `~/.env`, or `~/.zshrc` without logging secrets."""
            env_val = os.environ.get("NCBI_API_KEY")
            if env_val:
                return env_val.strip()
            for candidate in (Path.home() / ".env", Path.home() / ".zshrc", Path.home() / ".zshenv"):
                if candidate.is_file():
                    for line in candidate.read_text(errors="ignore").splitlines():
                        stripped = line.strip()
                        if stripped.startswith("#") or "NCBI_API_KEY" not in stripped:
                            continue
                        match = re.search(r"NCBI_API_KEY\s*=\s*[\"\']?([A-Za-z0-9_]+)[\"\']?", stripped)
                        if match:
                            return match.group(1).strip()
            return None

        async def _get_bytes(
            self,
            client: httpx2.AsyncClient,
            url: str,
            params: dict[str, str],
            sem: asyncio.Semaphore,
        ) -> bytes:
            """Execute a single rate-limited HTTP/2 GET request with exponential backoff."""
            req_params = dict(params)
            if self.api_key:
                req_params["api_key"] = self.api_key

            backoff = 0.8
            for attempt in range(self.max_retries):
                async with sem:
                    try:
                        resp = await client.get(url, params=req_params)
                        if resp.status_code == 200 and resp.content:
                            data = resp.content
                            if data.startswith(b"\x1f\x8b"):
                                data = gzip.decompress(data)
                            return data
                        if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                            await asyncio.sleep(backoff)
                            backoff *= 2.0
                            continue
                    except Exception:
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(backoff)
                            backoff *= 2.0
                            continue
            raise RuntimeError(f"Failed to fetch NCBI E-utilities resource from {url} with params {params}.")

        async def discover_full_bentley_panel(self) -> tuple[CpsReferenceSpec, ...]:
            """Query NCBI `esummary` across `CR931632..CR931722` + post-2006 loci to build the 92-serotype registry."""
            # Note: CR931636.1 is a 5.4 kb partial fragment of Serotype 5 (CR931637.1 is the full 20 kb locus);
            # CR931634.2 is the corrected Serotype 3 synthase locus.
            accs = ["CR931634.2" if idx == 634 else f"CR931{idx}.1" for idx in range(632, 723) if idx != 636] + [
                "EF538714.1",
                "HM171374.1",
            ]

            sem = asyncio.Semaphore(self.max_concurrency)
            headers = {"User-Agent": "SeroGen/0.1 (Async HTTP/2 Pneumo Benchmark)"}
            with self.recorder.use_cassette(self.cassette_name):
                async with httpx2.AsyncClient(http2=True, timeout=self.timeout, headers=headers) as client:
                    raw_xml = await self._get_bytes(
                        client,
                        f"{self.EUTILS_BASE}/esummary.fcgi",
                        {"db": "nuccore", "id": ",".join(accs), "retmode": "xml"},
                        sem,
                    )

            root = ET.fromstring(raw_xml)
            specs: list[CpsReferenceSpec] = []
            for doc in root.findall(".//DocSum"):
                acc_ver = doc.findtext("Item[@Name='AccessionVersion']") or ""
                title = doc.findtext("Item[@Name='Title']") or ""
                if acc_ver == "EF538714.1":
                    sero, strain = "6C", "CHPA388"
                elif acc_ver == "HM171374.1":
                    sero, strain = "6D", "MNZ21"
                elif acc_ver == "CR931687.1":
                    sero, strain = "24B", "2236/42"
                else:
                    m_sero = re.search(r"\(serotype\s+([0-9A-Za-z]+)\)", title, flags=re.IGNORECASE)
                    sero = m_sero.group(1).upper() if m_sero else acc_ver
                    m_strain = re.search(r"strain\s+(.+?)\s*\(serotype", title, flags=re.IGNORECASE)
                    strain = m_strain.group(1).strip() if m_strain else "SSI Ref"

                m_grp = re.match(r"^(\d+)", sero)
                serogroup = m_grp.group(1) if m_grp else sero
                mech = "CPS_SYNTHASE" if sero == "3" else "CPS_WZY"
                specs.append(CpsReferenceSpec(sero, serogroup, acc_ver, strain, mech))
            return tuple(specs)

        async def fetch_panel_async(
            self,
            cps_specs: tuple[CpsReferenceSpec, ...],
            wgs_specs: tuple[WgsStrainSpec, ...],
        ) -> tuple[Sequences, list[Sequence], float]:
            """Concurrently download all reference `cps` FASTAs and complete `WGS` chromosomes via HTTP/2."""
            t0 = time.perf_counter()
            sem = asyncio.Semaphore(self.max_concurrency)
            headers = {"User-Agent": "SeroGen/0.1 (Async HTTP/2 Pneumo Benchmark)"}

            with self.recorder.use_cassette(self.cassette_name):
                async with httpx2.AsyncClient(http2=True, timeout=self.timeout, headers=headers) as client:
                    # Batch-fetch reference cps loci in chunks of 30 accessions
                    cps_Accessions = [s.accession for s in cps_specs]
                    cps_chunks = [cps_Accessions[i : i + 30] for i in range(0, len(cps_Accessions), 30)]

                    async def _fetch_cps_Batch(chunk: list[str]) -> Sequences:
                        raw = await self._get_bytes(
                            client,
                            f"{self.EUTILS_BASE}/efetch.fcgi",
                            {"db": "nuccore", "id": ",".join(chunk), "rettype": "fasta", "retmode": "text"},
                            sem,
                        )
                        return FastaReader.from_bytes(raw).load()

                    async def _fetch_wgs_single(acc: str) -> Sequence:
                        raw = await self._get_bytes(
                            client,
                            f"{self.EUTILS_BASE}/efetch.fcgi",
                            {"db": "nuccore", "id": acc, "rettype": "fasta", "retmode": "text"},
                            sem,
                        )
                        return FastaReader.from_bytes(raw).load()[0]

                    cps_task = asyncio.gather(*(_fetch_cps_Batch(c) for c in cps_chunks))
                    wgs_task = asyncio.gather(*(_fetch_wgs_single(w.accession) for w in wgs_specs))
                    cps_batches, wgs_list = await asyncio.gather(cps_task, wgs_task)

            cps_seqs = Sequences.concat(cps_batches)
            elapsed = time.perf_counter() - t0
            return cps_seqs, list(wgs_list), elapsed

    @dataclass(slots=True)
    class PneumoSerotypeDatabase:
        """Indexed S. pneumoniae capsular reference database wrapping `SerotypeDatabase` + benchmark matrices."""

        ref_db: SerotypeDatabase
        specs: tuple[CpsReferenceSpec, ...]
        loci: Loci
        matcher: LocusMatcher
        pan_adj: object
        pan_nodes: np.ndarray
        pan_counts: np.ndarray
        pan_coords: np.ndarray
        all_vs_all_scores: np.ndarray
        all_vs_all_jaccards: np.ndarray
        all_vs_all_flags: np.ndarray
        wwl_distances: np.ndarray
        build_elapsed_s: float

        @classmethod
        def build(
            cls,
            specs: tuple[CpsReferenceSpec, ...],
            cps_seqs: Sequences,
            *,
            predictor: LocusPredictor,
            cluster_mode: str = "mcl",
            min_identity: float = 80.0,
            min_coverage: float = 0.80,
            novel_jaccard_threshold: float | str = "auto",
        ) -> PneumoSerotypeDatabase:
            """Segment reference cps sequences, build `SerotypeDatabase`, and compute 2D `LayoutEngine` coordinates."""
            t0 = time.perf_counter()
            extracted_loci: list[Locus] = []
            kept_specs: list[CpsReferenceSpec] = []

            for spec, seq in zip(specs, cps_seqs, strict=True):
                best_loc = predictor.predict(seq, categories=ArchetypeCategory.CAPSULE).best()
                if best_loc is None:
                    best_loc = predictor.predict(seq).best()
                if best_loc is None:
                    continue
                extracted_loci.append(best_loc.with_name(spec.serotype))
                kept_specs.append(spec)

            ref_db = SerotypeDatabase.build(
                extracted_loci,
                names=[s.serotype for s in kept_specs],
                serogroups=[s.serogroup for s in kept_specs],
                accessions=[s.accession for s in kept_specs],
                strains=[s.strain for s in kept_specs],
                cluster_mode=cluster_mode,  # type: ignore[arg-type]
                min_identity=min_identity,
                min_coverage=min_coverage,
                novel_jaccard_threshold=novel_jaccard_threshold,  # type: ignore[arg-type]
            )

            layout_adj = ref_db.pan_adj.tocsr().copy()
            if layout_adj.nnz > 0:
                layout_adj.data = 1.0 + 0.25 * np.log1p(layout_adj.data.astype(np.float64))
            layout_engine = LayoutEngine(
                mode="fr",
                dim=2,
                iterations=200,
                gravity=0.35,
                seed=42,
            )
            pan_coords = layout_engine.layout(layout_adj)

            n_db = len(ref_db)
            ava_matches = ref_db.match(ref_db.loci, sort=False)
            scores_mat = ava_matches.scores.reshape(n_db, n_db).astype(np.float64)
            jaccards_mat = ava_matches.jaccards.reshape(n_db, n_db).astype(np.float64)
            flags_mat = ava_matches.match_types.reshape(n_db, n_db)
            wwl_mat = ref_db.loci.wwl_distance(iterations=2)

            elapsed = time.perf_counter() - t0
            return cls(
                ref_db=ref_db,
                specs=tuple(kept_specs),
                loci=ref_db.loci,
                matcher=ref_db.matcher,
                pan_adj=ref_db.pan_adj,
                pan_nodes=ref_db.pan_nodes,
                pan_counts=ref_db.pan_counts,
                pan_coords=pan_coords,
                all_vs_all_scores=scores_mat,
                all_vs_all_jaccards=jaccards_mat,
                all_vs_all_flags=flags_mat,
                wwl_distances=wwl_mat,
                build_elapsed_s=elapsed,
            )

    class PneumoBenchmarkSuite:
        """Executes the 4-stage S. pneumoniae database and whole-genome validation suite."""

        def __init__(
            self,
            db: PneumoSerotypeDatabase,
            wgs_specs: tuple[WgsStrainSpec, ...],
            wgs_seqs: list[Sequence],
            *,
            wgs_predictor: LocusPredictor,
        ) -> None:
            self.db = db
            self.wgs_specs = wgs_specs
            self.wgs_seqs = wgs_seqs
            self.wgs_predictor = wgs_predictor

        @staticmethod
        def _select_capsule_locus(pred_loci: Loci) -> Locus | None:
            """Select the primary capsular locus (`ArchetypeCategory.CAPSULE`) from a whole-genome prediction."""
            return pred_loci.filter(categories=ArchetypeCategory.CAPSULE).best()

        @staticmethod
        def format_match_flags(flag_val: int | MatchType) -> str:
            """Format a `MatchType` bitflag into a compact human-readable label."""
            mt = MatchType(int(flag_val))
            if mt == MatchType.EXACT:
                return "EXACT"
            names = [f.name for f in MatchType if f != MatchType.EXACT and (mt & f) and f.name is not None]
            return " | ".join(names)

        def run_wgs_serotyping(self) -> tuple[list[dict[str, object]], dict[str, Locus]]:
            """Stage 2: Segment complete RefSeq chromosomes and match against the reference cps database."""
            records: list[dict[str, object]] = []
            intact_capsule_loci: dict[str, Locus] = {}

            for spec, seq in zip(self.wgs_specs, self.wgs_seqs, strict=True):
                t0 = time.perf_counter()
                pred_loci = self.wgs_predictor.predict(seq)
                cps_loc = self._select_capsule_locus(pred_loci)
                pred_ms = (time.perf_counter() - t0) * 1000.0

                if spec.is_negative_control:
                    correct = cps_loc is None
                    records.append(
                        {
                            "strain": spec.strain,
                            "accession": spec.accession,
                            "genome_bp": len(seq),
                            "true_serotype": spec.true_serotype,
                            "predicted_serotype": "None" if cps_loc is None else "FalsePositive",
                            "runner_up_serotype": "—",
                            "archetype": "None" if cps_loc is None else cps_loc.archetype.name,
                            "locus_interval": "—",
                            "num_genes": 0 if cps_loc is None else cps_loc.num_genes,
                            "completeness": 0.0 if cps_loc is None else float(cps_loc.completeness),
                            "top1_score": 0.0,
                            "top2_score": 0.0,
                            "margin": 0.0,
                            "top1_jaccard": 0.0,
                            "match_flags": "NEGATIVE_CONTROL (0 CPS Loci)",
                            "is_correct": correct,
                            "runtime_ms": pred_ms,
                        }
                    )
                    continue

                if cps_loc is None:
                    records.append(
                        {
                            "strain": spec.strain,
                            "accession": spec.accession,
                            "genome_bp": len(seq),
                            "true_serotype": spec.true_serotype,
                            "predicted_serotype": "Missed",
                            "runner_up_serotype": "—",
                            "archetype": "None",
                            "locus_interval": "—",
                            "num_genes": 0,
                            "completeness": 0.0,
                            "top1_score": 0.0,
                            "top2_score": 0.0,
                            "margin": 0.0,
                            "top1_jaccard": 0.0,
                            "match_flags": "MISSED",
                            "is_correct": False,
                            "runtime_ms": pred_ms,
                        }
                    )
                    continue

                intact_capsule_loci[spec.strain] = cps_loc
                matches = self.db.ref_db.match(cps_loc, top_k=3)
                top1 = matches[0]
                top2 = matches[1] if len(matches) > 1 else None
                pred_sero = top1.target_name
                runner_sero = top2.target_name if top2 is not None else "—"
                top2_score = float(top2.score) if top2 is not None else 0.0

                records.append(
                    {
                        "strain": spec.strain,
                        "accession": spec.accession,
                        "genome_bp": len(seq),
                        "true_serotype": spec.true_serotype,
                        "predicted_serotype": pred_sero,
                        "runner_up_serotype": runner_sero,
                        "archetype": cps_loc.archetype.name,
                        "locus_interval": f"{cps_loc.start:,}..{cps_loc.end:,} ({cps_loc.strand:+d})",
                        "num_genes": cps_loc.num_genes,
                        "completeness": float(cps_loc.completeness),
                        "top1_score": float(top1.score),
                        "top2_score": top2_score,
                        "margin": float(top1.score) - top2_score,
                        "top1_jaccard": float(top1.jaccard),
                        "match_flags": self.format_match_flags(top1.match_type),
                        "is_correct": pred_sero == spec.true_serotype,
                        "runtime_ms": pred_ms,
                    }
                )

            return records, intact_capsule_loci

        def run_fragmentation_stress_test(
            self,
            intact_capsule_loci: dict[str, Locus],
        ) -> list[dict[str, object]]:
            """Stage 3: Artificially split chromosomes inside the cps locus (intergenic & mid-gene) to test stitching."""
            frag_records: list[dict[str, object]] = []
            rng = np.random.default_rng(2026)

            rc_tab = bytes.maketrans(b"ACGTacgtNn", b"TGCAtgcaNn")

            for spec, seq in zip(self.wgs_specs, self.wgs_seqs, strict=True):
                if spec.is_negative_control or spec.strain not in intact_capsule_loci:
                    continue
                intact_loc = intact_capsule_loci[spec.strain]
                n_genes = intact_loc.num_genes
                if n_genes < 3:
                    continue

                # Focus on a 120 kb chromosomal window around the cps locus for rapid multi-contig simulation
                win_pad = 45_000
                win_start = max(0, int(intact_loc.start) - win_pad)
                win_end = min(len(seq), int(intact_loc.end) + win_pad)
                raw_win = seq[win_start:win_end]

                mid_gene_idx = n_genes // 2
                g_starts = intact_loc.genes.intervals.starts
                g_ends = intact_loc.genes.intervals.ends

                # Mode A: Intergenic contig break between middle genes of the cps operon
                intergenic_cut_abs = int((g_ends[mid_gene_idx - 1] + g_starts[mid_gene_idx]) // 2)
                intergenic_cut_rel = intergenic_cut_abs - win_start

                # Mode B: Mid-ORF contig break directly through the middle of an internal cps gene
                mid_orf_cut_abs = int((g_starts[mid_gene_idx] + g_ends[mid_gene_idx]) // 2)
                mid_orf_cut_rel = mid_orf_cut_abs - win_start

                for mode_label, cut_rel in (
                    ("Intact Control", None),
                    ("Intergenic Contig Split", intergenic_cut_rel),
                    ("Mid-ORF Split-Gene Break", mid_orf_cut_rel),
                ):
                    if cut_rel is None:
                        assembly = Sequences.from_bytes([raw_win])
                    else:
                        # Add 2 additional random background chromosome breaks outside the locus
                        bg_cut_left = int(rng.integers(5_000, max(6_000, int(intact_loc.start) - win_start - 5_000)))
                        bg_cut_right = int(
                            rng.integers(
                                min(len(raw_win) - 6_000, int(intact_loc.end) - win_start + 5_000),
                                len(raw_win) - 4_000,
                            )
                        )
                        cuts = sorted({bg_cut_left, cut_rel, bg_cut_right})
                        pieces: list[bytes] = []
                        prev = 0
                        for idx_c, c_pos in enumerate(cuts):
                            chunk_bytes = raw_win[prev:c_pos]
                            # Reverse-complement every odd contig to test orientation-agnostic stitching
                            if idx_c % 2 == 1:
                                pieces.append(chunk_bytes.translate(rc_tab)[::-1])
                            else:
                                pieces.append(chunk_bytes)
                            prev = c_pos
                        pieces.append(raw_win[prev:])
                        assembly = Sequences.from_bytes(pieces)

                    pred_loci = self.wgs_predictor.predict(assembly)
                    stitched_loc = self._select_capsule_locus(pred_loci)
                    if stitched_loc is None:
                        frag_records.append(
                            {
                                "strain": spec.strain,
                                "true_serotype": spec.true_serotype,
                                "mode": mode_label,
                                "predicted_serotype": "Missed",
                                "score": 0.0,
                                "jaccard": 0.0,
                                "completeness": 0.0,
                                "is_fragmented": False,
                                "has_fragmented_flag": False,
                                "num_contigs": 0,
                                "is_correct": False,
                            }
                        )
                        continue

                    top_hit = self.db.ref_db.match(stitched_loc, top_k=1)[0]
                    pred_sero = top_hit.target_name
                    has_frag_flag = bool(top_hit.match_type & MatchType.FRAGMENTED_VARIANT)
                    frag_records.append(
                        {
                            "strain": spec.strain,
                            "true_serotype": spec.true_serotype,
                            "mode": mode_label,
                            "predicted_serotype": pred_sero,
                            "score": float(top_hit.score),
                            "jaccard": float(top_hit.jaccard),
                            "completeness": float(stitched_loc.completeness),
                            "is_fragmented": bool(stitched_loc.is_fragmented),
                            "has_fragmented_flag": has_frag_flag,
                            "num_contigs": int(stitched_loc.num_contigs),
                            "is_correct": pred_sero == spec.true_serotype,
                        }
                    )

            return frag_records

        def run_leave_one_out_novelty(
            self,
            intact_capsule_loci: dict[str, Locus],
        ) -> list[dict[str, object]]:
            """Stage 4: Hold out each query genome's true serotype from the DB via `db.without()` to test novelty detection."""
            loo_records: list[dict[str, object]] = []
            seen_serotypes: set[str] = set()

            for spec in self.wgs_specs:
                if spec.is_negative_control or spec.strain not in intact_capsule_loci:
                    continue
                if spec.true_serotype in seen_serotypes:
                    continue
                seen_serotypes.add(spec.true_serotype)

                q_loc = intact_capsule_loci[spec.strain]
                full_hit = self.db.ref_db.match(q_loc, top_k=1)[0]

                loo_db = self.db.ref_db.without(spec.true_serotype)
                loo_hit = loo_db.match(q_loc, top_k=1)[0]
                closest_rec = loo_db[loo_hit.target_index]

                # A held-out serotype is novel if `is_novel` is set or if it differs from the closest relative
                is_novel_flagged = bool(loo_hit.is_novel)
                same_serogroup = closest_rec.serogroup == spec.serogroup

                loo_records.append(
                    {
                        "strain": spec.strain,
                        "true_serotype": spec.true_serotype,
                        "serogroup": spec.serogroup,
                        "full_db_score": float(full_hit.score),
                        "full_db_jaccard": float(full_hit.jaccard),
                        "held_out_closest_serotype": closest_rec.name,
                        "held_out_same_serogroup": same_serogroup,
                        "held_out_score": float(loo_hit.score),
                        "held_out_jaccard": float(loo_hit.jaccard),
                        "score_drop": float(full_hit.score) - float(loo_hit.score),
                        "held_out_missing_genes": int(loo_hit.missing_genes),
                        "held_out_extra_genes": int(loo_hit.extra_genes),
                        "held_out_flags": self.format_match_flags(loo_hit.match_type),
                        "is_novel_detected": is_novel_flagged,
                        "only_benign_flags": bool((loo_hit.match_type & ~BENIGN_VARIANT_MASK) == MatchType.EXACT),
                    }
                )

            return loo_records

    return NcbiAsyncFetcher, PneumoBenchmarkSuite, PneumoSerotypeDatabase


@app.cell
def _(mo):
    # ------------------------------------------------------------------------------------------------------------------
    # Interactive Marimo Controls (Defaults Execute Automatically in Both Interactive & Script Mode)
    # ------------------------------------------------------------------------------------------------------------------
    is_script_mode = mo.app_meta().mode == "script"

    panel_selector = mo.ui.dropdown(
        options={
            "Curated 18-Serotype Benchmark Panel (Fast, ~5s)": "curated_18",
            "Full 92-Serotype Bentley + Post-2006 Panel (~15s)": "full_92",
        },
        value="Curated 18-Serotype Benchmark Panel (Fast, ~5s)",
        label="Reference Database Scope",
    )
    cluster_mode_selector = mo.ui.dropdown(
        options={
            "MCL (Bitscore >= 40, I = 2.0, Role-Stratified)": "mcl",
            "CD-HIT (Greedy 80% AA Identity)": "cdhit",
        },
        value="MCL (Bitscore >= 40, I = 2.0, Role-Stratified)",
        label="Protein Clustering Mode",
    )
    min_identity_slider = mo.ui.slider(
        start=70.0,
        stop=95.0,
        step=5.0,
        value=80.0,
        label="Clusterer / Matcher Min AA Identity (%)",
    )
    min_coverage_slider = mo.ui.slider(
        start=0.50,
        stop=0.90,
        step=0.05,
        value=0.80,
        label="Clusterer / Matcher Min Coverage",
    )

    mo.output.replace(
        mo.md(
            f"""
            # SeroGen — *Streptococcus pneumoniae* Reference Database & Whole-Genome Serotyping Benchmark
            This self-contained notebook validates **in-memory NCBI HTTP/2 sequence fetching (`httpx2`)**,
            **reference *cps* database construction (`Loci.cluster_genes` + `Loci.as_pan_graph`)**,
            **whole-genome serotyping (`LocusPredictor` + `LocusMatcher`)**, **contig-split stitching (`FRAGMENTED_VARIANT`)**,
            and **leave-one-out novelty detection (`MatchType.is_novel`)**.

            {mo.hstack([panel_selector, cluster_mode_selector, min_identity_slider, min_coverage_slider], justify="start", gap=2.0)}
            """
        )
    )
    return cluster_mode_selector, is_script_mode, min_coverage_slider, min_identity_slider, panel_selector


@app.cell
async def _(
    CURATED_CPS_SPECS,
    LocusPredictor,
    NcbiAsyncFetcher,
    PneumoBenchmarkSuite,
    PneumoSerotypeDatabase,
    WGS_VALIDATION_SPECS,
    cluster_mode_selector,
    min_coverage_slider,
    min_identity_slider,
    panel_selector,
):
    # ------------------------------------------------------------------------------------------------------------------
    # Execute Async HTTP/2 Download, Reference DB Construction & 4-Stage Validation Suite
    # ------------------------------------------------------------------------------------------------------------------
    fetcher = NcbiAsyncFetcher()
    if panel_selector.value == "full_92":
        active_cps_specs = await fetcher.discover_full_bentley_panel()
    else:
        active_cps_specs = CURATED_CPS_SPECS

    cps_seqs, wgs_seqs, download_elapsed_s = await fetcher.fetch_panel_async(
        active_cps_specs,
        WGS_VALIDATION_SPECS,
    )

    cps_predictor = LocusPredictor(whole_genome=False)
    wgs_predictor = LocusPredictor(whole_genome=True, stitch_contig_splits=True)

    pneumo_db = PneumoSerotypeDatabase.build(
        active_cps_specs,
        cps_seqs,
        predictor=cps_predictor,
        cluster_mode=str(cluster_mode_selector.value),
        min_identity=float(min_identity_slider.value),
        min_coverage=float(min_coverage_slider.value),
        novel_jaccard_threshold="auto",
    )

    suite = PneumoBenchmarkSuite(
        pneumo_db,
        WGS_VALIDATION_SPECS,
        wgs_seqs,
        wgs_predictor=wgs_predictor,
    )
    wgs_records, intact_wgs_loci = suite.run_wgs_serotyping()
    frag_records = suite.run_fragmentation_stress_test(intact_wgs_loci)
    loo_records = suite.run_leave_one_out_novelty(intact_wgs_loci)

    return (
        active_cps_specs,
        cps_predictor,
        cps_seqs,
        download_elapsed_s,
        fetcher,
        frag_records,
        intact_wgs_loci,
        loo_records,
        pneumo_db,
        suite,
        wgs_predictor,
        wgs_records,
        wgs_seqs,
    )


@app.cell
def _(
    GeneNode,
    GeneRole,
    alt,
    download_elapsed_s,
    frag_records,
    loo_records,
    mo,
    np,
    pneumo_db,
    wgs_records,
    wgs_seqs,
):
    # ------------------------------------------------------------------------------------------------------------------
    # Compile Summary Metrics & Interactive 4-Panel Altair / Vega-Lite Dashboard
    # ------------------------------------------------------------------------------------------------------------------
    n_db_loci = len(pneumo_db.loci)
    total_db_genes = int(pneumo_db.loci.total_nodes)
    unique_clusters = int(len(np.unique(pneumo_db.loci.cluster_ids[pneumo_db.loci.cluster_ids > 0])))
    n_pan_nodes = int(len(pneumo_db.pan_nodes))
    n_pan_edges = int(pneumo_db.pan_adj.nnz)
    compression_ratio = total_db_genes / max(n_pan_nodes, 1)

    wgs_correct_count = sum(1 for r in wgs_records if r["is_correct"])
    wgs_total_count = len(wgs_records)
    wgs_accuracy_pct = 100.0 * wgs_correct_count / max(wgs_total_count, 1)

    frag_split_only = [r for r in frag_records if r["mode"] != "Intact Control"]
    frag_correct_count = sum(1 for r in frag_split_only if r["is_correct"])
    frag_accuracy_pct = 100.0 * frag_correct_count / max(len(frag_split_only), 1)

    loo_novel_count = sum(1 for r in loo_records if r["is_novel_detected"])
    loo_accuracy_pct = 100.0 * loo_novel_count / max(len(loo_records), 1)

    total_downloaded_mb = (sum(len(s) for s in wgs_seqs) + int(np.sum(pneumo_db.loci.intervals.lengths))) / 1_000_000.0

    # 1. All-vs-All Reference Serotype Similarity Heatmap Data
    ava_rows: list[dict[str, object]] = []
    for _i, _spec_i in enumerate(pneumo_db.specs):
        for _j, _spec_j in enumerate(pneumo_db.specs):
            ava_rows.append(
                {
                    "Query Serotype": _spec_i.serotype,
                    "Target Serotype": _spec_j.serotype,
                    "Query Serogroup": _spec_i.serogroup,
                    "Target Serogroup": _spec_j.serogroup,
                    "Similarity Score": round(float(pneumo_db.all_vs_all_scores[_i, _j]), 4),
                    "Jaccard": round(float(pneumo_db.all_vs_all_jaccards[_i, _j]), 4),
                    "WWL Distance": round(float(pneumo_db.wwl_distances[_i, _j]), 4),
                }
            )

    serotype_order = [s.serotype for s in pneumo_db.specs]
    heatmap_dim = 360 if len(serotype_order) <= 25 else 520
    chart_ava = (
        alt.Chart(alt.Data(values=ava_rows))
        .mark_rect()
        .encode(
            x=alt.X("Target Serotype:N", sort=serotype_order, title="Reference Target Serotype"),
            y=alt.Y("Query Serotype:N", sort=serotype_order, title="Reference Query Serotype"),
            color=alt.Color(
                "Similarity Score:Q",
                scale=alt.Scale(scheme="viridis", domain=[0.0, 1.0]),
                title="Matcher Score",
            ),
            tooltip=[
                "Query Serotype:N",
                "Target Serotype:N",
                "Similarity Score:Q",
                "Jaccard:Q",
                "WWL Distance:Q",
            ],
        )
        .properties(
            width=heatmap_dim,
            height=heatmap_dim,
            title="Stage 1: Reference cps All-vs-All Similarity Matrix",
        )
    )

    # 2. Whole-Genome Top-1 vs Runner-Up Score Comparison
    wgs_bar_rows: list[dict[str, object]] = []
    for _r in wgs_records:
        if _r["true_serotype"] == "None":
            continue
        _label = f"{_r['strain']} (True: {_r['true_serotype']})"
        wgs_bar_rows.append(
            {
                "Strain": _label,
                "Rank": "Top-1 Hit (Predicted)",
                "Score": round(float(_r["top1_score"]), 4),
                "Called Serotype": str(_r["predicted_serotype"]),
            }
        )
        wgs_bar_rows.append(
            {
                "Strain": _label,
                "Rank": "Top-2 Runner-Up",
                "Score": round(float(_r["top2_score"]), 4),
                "Called Serotype": str(_r["runner_up_serotype"]),
            }
        )

    chart_wgs = (
        alt.Chart(alt.Data(values=wgs_bar_rows))
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y("Strain:N", sort=None, title="Whole-Genome RefSeq Strain"),
            x=alt.X("Score:Q", scale=alt.Scale(domain=[0.0, 1.0]), title="LocusMatcher Score"),
            yOffset=alt.YOffset("Rank:N"),
            color=alt.Color(
                "Rank:N",
                scale=alt.Scale(
                    domain=["Top-1 Hit (Predicted)", "Top-2 Runner-Up"],
                    range=["#2563eb", "#94a3b8"],
                ),
                title="Match Rank",
            ),
            tooltip=["Strain:N", "Rank:N", "Called Serotype:N", "Score:Q"],
        )
        .properties(
            width=360,
            height=360,
            title="Stage 2: Whole-Genome Top-1 vs Runner-Up Separation",
        )
    )

    # 3. Simulated Contig-Split Stress-Test Chart
    chart_frag = (
        alt.Chart(alt.Data(values=frag_records))
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("strain:N", sort=None, title="Fragmented Whole-Genome Strain"),
            y=alt.Y("score:Q", scale=alt.Scale(domain=[0.0, 1.0]), title="Stitched Locus Match Score"),
            xOffset=alt.XOffset("mode:N"),
            color=alt.Color(
                "mode:N",
                scale=alt.Scale(
                    domain=["Intact Control", "Intergenic Contig Split", "Mid-ORF Split-Gene Break"],
                    range=["#059669", "#d97706", "#7c3aed"],
                ),
                title="Assembly Condition",
            ),
            tooltip=[
                alt.Tooltip("strain:N", title="Strain"),
                alt.Tooltip("true_serotype:N", title="True Serotype"),
                alt.Tooltip("mode:N", title="Assembly Condition"),
                alt.Tooltip("predicted_serotype:N", title="Called Serotype"),
                alt.Tooltip("score:Q", title="Score", format=".3f"),
                alt.Tooltip("num_contigs:Q", title="Spanned Contigs"),
                alt.Tooltip("has_fragmented_flag:N", title="FRAGMENTED_VARIANT Flag"),
            ],
        )
        .properties(
            width=360,
            height=280,
            title="Stage 3: Contig-Split Stitching & Mid-ORF Break Recovery",
        )
    )

    # 4. Leave-One-Out Novelty Detection Chart
    loo_chart_rows: list[dict[str, object]] = []
    for _r in loo_records:
        _label = f"{_r['strain']} ({_r['true_serotype']})"
        loo_chart_rows.append(
            {
                "Query": _label,
                "Database State": "Full Reference DB",
                "Score": round(float(_r["full_db_score"]), 4),
                "Best Hit": str(_r["true_serotype"]),
                "Flags": "True Serotype Present",
            }
        )
        loo_chart_rows.append(
            {
                "Query": _label,
                "Database State": "Leave-One-Out DB (True Serotype Held Out)",
                "Score": round(float(_r["held_out_score"]), 4),
                "Best Hit": str(_r["held_out_closest_serotype"]),
                "Flags": str(_r["held_out_flags"]),
            }
        )

    chart_loo = (
        alt.Chart(alt.Data(values=loo_chart_rows))
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("Query:N", sort=None, title="Query Strain (Held-Out Serotype)"),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0.0, 1.0]), title="Top Match Score"),
            xOffset=alt.XOffset("Database State:N"),
            color=alt.Color(
                "Database State:N",
                scale=alt.Scale(
                    domain=["Full Reference DB", "Leave-One-Out DB (True Serotype Held Out)"],
                    range=["#0d9488", "#e11d48"],
                ),
                title="Reference DB State",
            ),
            tooltip=["Query:N", "Database State:N", "Best Hit:N", "Score:Q", "Flags:N"],
        )
        .properties(
            width=360,
            height=280,
            title="Stage 4: Leave-One-Out Novelty Detection & Serogroup Fallback",
        )
    )

    full_dashboard_chart = alt.vconcat(
        alt.hconcat(chart_ava, chart_wgs).resolve_scale(color="independent"),
        alt.hconcat(chart_frag, chart_loo).resolve_scale(color="independent"),
    ).configure_axis(labelFontSize=11, titleFontSize=12)

    # Top Conserved Core Gene Clusters & LayoutEngine 2D Pan-Locus Network Data
    pan_order = np.argsort(-pneumo_db.pan_counts)
    top_core_idx_set = {int(idx) for idx in pan_order[:14]}
    top_pan_rows: list[dict[str, object]] = []
    for _idx in pan_order[:12]:
        _node_val = int(pneumo_db.pan_nodes[_idx])
        _role_id = int(GeneNode.ROLE_ID.extract(_node_val))
        _cid = int(GeneNode.CLUSTER_ID.extract(_node_val))
        _cnt = int(pneumo_db.pan_counts[_idx])
        _role_obj = GeneRole(_role_id)
        top_pan_rows.append(
            {
                "cluster_id": _cid,
                "role_name": _role_obj.name,
                "role_abbrev": _role_obj.abbreviation,
                "category": _role_obj.category.value,
                "locus_count": _cnt,
                "frequency_pct": 100.0 * _cnt / max(n_db_loci, 1),
            }
        )

    pan_csr = pneumo_db.pan_adj.tocsr()
    node_max_edge = np.zeros(n_pan_nodes, dtype=np.int32)
    node_in_deg = np.zeros(n_pan_nodes, dtype=np.int32)
    node_out_deg = np.zeros(n_pan_nodes, dtype=np.int32)
    pan_edge_rows: list[dict[str, object]] = []

    for _u in range(n_pan_nodes):
        _u_val = int(pneumo_db.pan_nodes[_u])
        _u_role = GeneRole(int(GeneNode.ROLE_ID.extract(_u_val)))
        _u_cid = int(GeneNode.CLUSTER_ID.extract(_u_val))
        _row_start = int(pan_csr.indptr[_u])
        _row_end = int(pan_csr.indptr[_u + 1])
        for _k in range(_row_start, _row_end):
            _v = int(pan_csr.indices[_k])
            if _u == _v:
                continue
            _w = int(pan_csr.data[_k])
            if _w <= 0:
                continue
            node_out_deg[_u] += 1
            node_in_deg[_v] += 1
            if _w > node_max_edge[_u]:
                node_max_edge[_u] = _w
            if _w > node_max_edge[_v]:
                node_max_edge[_v] = _w

            _v_val = int(pneumo_db.pan_nodes[_v])
            _v_role = GeneRole(int(GeneNode.ROLE_ID.extract(_v_val)))
            _v_cid = int(GeneNode.CLUSTER_ID.extract(_v_val))
            pan_edge_rows.append(
                {
                    "x": round(float(pneumo_db.pan_coords[_u, 0]), 5),
                    "y": round(float(pneumo_db.pan_coords[_u, 1]), 5),
                    "x2": round(float(pneumo_db.pan_coords[_v, 0]), 5),
                    "y2": round(float(pneumo_db.pan_coords[_v, 1]), 5),
                    "weight": _w,
                    "weight_pct": round(100.0 * _w / max(n_db_loci, 1), 1),
                    "source_node": f"{_u_role.abbreviation} (CID-{_u_cid:03d})",
                    "target_node": f"{_v_role.abbreviation} (CID-{_v_cid:03d})",
                    "transition": f"{_u_role.abbreviation} (C{_u_cid}) -> {_v_role.abbreviation} (C{_v_cid})",
                }
            )

    pan_node_rows: list[dict[str, object]] = []
    for _idx in range(n_pan_nodes):
        _node_val = int(pneumo_db.pan_nodes[_idx])
        _role_id = int(GeneNode.ROLE_ID.extract(_node_val))
        _cid = int(GeneNode.CLUSTER_ID.extract(_node_val))
        _cnt = int(pneumo_db.pan_counts[_idx])
        _freq_pct = round(100.0 * _cnt / max(n_db_loci, 1), 1)
        _role_obj = GeneRole(_role_id)
        _max_sup = int(node_max_edge[_idx]) if int(node_max_edge[_idx]) > 0 else 1
        _core_label = f"{_role_obj.abbreviation} (C{_cid})" if _idx in top_core_idx_set else ""
        pan_node_rows.append(
            {
                "node_idx": _idx,
                "cluster_id": f"CID-{_cid:03d}",
                "role_abbrev": _role_obj.abbreviation,
                "role_name": _role_obj.name,
                "category": _role_obj.category.value,
                "locus_count": _cnt,
                "frequency_pct": _freq_pct,
                "degree": int(node_in_deg[_idx] + node_out_deg[_idx]),
                "max_edge_support": _max_sup,
                "x": round(float(pneumo_db.pan_coords[_idx, 0]), 5),
                "y": round(float(pneumo_db.pan_coords[_idx, 1]), 5),
                "core_label": _core_label,
            }
        )

    max_edge_weight = max((int(r["weight"]) for r in pan_edge_rows), default=1)
    slider_max = max(2, min(max_edge_weight, max(2, n_db_loci // 2)))
    min_edge_slider = alt.binding_range(
        min=1,
        max=slider_max,
        step=1,
        name="Min Synteny Edge Support (Loci): ",
    )
    min_edge_param = alt.param(name="min_edge_support", value=1, bind=min_edge_slider)
    cat_selection = alt.selection_point(fields=["category"], bind="legend")

    pan_edges_layer = (
        alt.Chart(alt.Data(values=pan_edge_rows))
        .mark_rule(strokeCap="round")
        .transform_filter(alt.datum.weight >= min_edge_param)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-1.18, 1.18]), axis=None),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[-1.18, 1.18]), axis=None),
            x2="x2:Q",
            y2="y2:Q",
            strokeWidth=alt.StrokeWidth(
                "weight:Q",
                scale=alt.Scale(type="sqrt", range=[0.75, 5.5]),
                title="Shared Loci (Edge)",
            ),
            opacity=alt.Opacity(
                "weight:Q",
                scale=alt.Scale(type="sqrt", range=[0.22, 0.85]),
                legend=None,
            ),
            color=alt.value("#64748b"),
            tooltip=[
                alt.Tooltip("transition:N", title="Synteny Transition"),
                alt.Tooltip("source_node:N", title="Source Cluster"),
                alt.Tooltip("target_node:N", title="Target Cluster"),
                alt.Tooltip("weight:Q", title="Shared Loci"),
                alt.Tooltip("weight_pct:Q", title="Locus Share (%)", format=".1f"),
            ],
        )
    )

    pan_nodes_layer = (
        alt.Chart(alt.Data(values=pan_node_rows))
        .mark_circle(stroke="#0f172a", strokeWidth=1.1)
        .transform_filter(alt.datum.max_edge_support >= min_edge_param)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-1.18, 1.18]), axis=None),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[-1.18, 1.18]), axis=None),
            size=alt.Size(
                "locus_count:Q",
                scale=alt.Scale(range=[45, 540]),
                title="Locus Presence",
            ),
            color=alt.Color(
                "category:N",
                scale=alt.Scale(scheme="tableau10"),
                title="Functional Category",
            ),
            opacity=alt.condition(cat_selection, alt.value(0.92), alt.value(0.18)),
            tooltip=[
                alt.Tooltip("cluster_id:N", title="Cluster ID"),
                alt.Tooltip("role_abbrev:N", title="Gene Role"),
                alt.Tooltip("role_name:N", title="Full Role Name"),
                alt.Tooltip("category:N", title="Functional Category"),
                alt.Tooltip("locus_count:Q", title="Locus Count"),
                alt.Tooltip("frequency_pct:Q", title="Core Frequency (%)", format=".1f"),
                alt.Tooltip("degree:Q", title="Synteny Degree"),
            ],
        )
        .add_params(cat_selection)
    )

    pan_labels_layer = (
        alt.Chart(alt.Data(values=[r for r in pan_node_rows if r["core_label"]]))
        .mark_text(dy=-12, fontSize=10, fontWeight="bold", color="#0f172a")
        .transform_filter(alt.datum.max_edge_support >= min_edge_param)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-1.18, 1.18]), axis=None),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[-1.18, 1.18]), axis=None),
            text="core_label:N",
        )
    )

    chart_pan_graph = (
        alt.layer(pan_edges_layer, pan_nodes_layer, pan_labels_layer)
        .add_params(min_edge_param)
        .properties(
            width=820,
            height=460,
            title="Stage 1B: S. pneumoniae Consensus Pan-K-Locus Synteny Network (Fruchterman-Reingold LayoutEngine)",
        )
        .configure_view(stroke=None)
    )

    mo.output.replace(
        mo.vstack(
            [
                mo.md(
                    f"""
                    ## Benchmark Summary KPIs
                    - **HTTP/2 Download (`httpx2` + `cassetter`)**: `{n_db_loci}` reference *cps* loci + `{len(wgs_seqs)}` complete chromosomes (`{total_downloaded_mb:.1f} MB` in `{download_elapsed_s:.2f}s`)
                    - **Reference Database & Pan-Graph (`LayoutEngine`)**: `{total_db_genes}` total genes clustered into `{unique_clusters}` homology groups (`{n_pan_nodes}` pan-nodes, `{n_pan_edges}` directed synteny edges, `{compression_ratio:.2f}x` graph compression)
                    - **Stage 2 Whole-Genome Serotyping Accuracy**: **`{wgs_correct_count} / {wgs_total_count}` (`{wgs_accuracy_pct:.1f}%`)** (including `R6` rough negative control with `0` capsule loci)
                    - **Stage 3 Contig-Split Stitching Recovery**: **`{frag_correct_count} / {len(frag_split_only)}` (`{frag_accuracy_pct:.1f}%`)** across intergenic & mid-ORF contig breaks
                    - **Stage 4 Leave-One-Out Novelty Detection**: **`{loo_novel_count} / {len(loo_records)}` (`{loo_accuracy_pct:.1f}%`)** held-out serotypes flagged with `Match.is_novel == True`
                    """
                ),
                chart_pan_graph,
                full_dashboard_chart,
            ]
        )
    )

    return (
        ava_rows,
        chart_ava,
        chart_frag,
        chart_loo,
        chart_pan_graph,
        chart_wgs,
        compression_ratio,
        frag_accuracy_pct,
        frag_correct_count,
        frag_split_only,
        full_dashboard_chart,
        loo_accuracy_pct,
        loo_chart_rows,
        loo_novel_count,
        n_db_loci,
        n_pan_edges,
        n_pan_nodes,
        pan_edge_rows,
        pan_node_rows,
        top_pan_rows,
        total_db_genes,
        total_downloaded_mb,
        unique_clusters,
        wgs_accuracy_pct,
        wgs_bar_rows,
        wgs_correct_count,
        wgs_total_count,
    )


@app.cell
def _(
    Path,
    chart_pan_graph,
    compression_ratio,
    download_elapsed_s,
    frag_accuracy_pct,
    frag_correct_count,
    frag_records,
    frag_split_only,
    full_dashboard_chart,
    is_script_mode,
    loo_accuracy_pct,
    loo_novel_count,
    loo_records,
    mo,
    n_db_loci,
    n_pan_edges,
    n_pan_nodes,
    pneumo_db,
    top_pan_rows,
    total_db_genes,
    total_downloaded_mb,
    unique_clusters,
    wgs_accuracy_pct,
    wgs_correct_count,
    wgs_records,
    wgs_total_count,
):
    # ------------------------------------------------------------------------------------------------------------------
    # Standalone HTML Report Generator (scripts/benchmark_spneumo_db_report.html)
    # ------------------------------------------------------------------------------------------------------------------
    report_path = Path(__file__).resolve().parent / "benchmark_spneumo_db_report.html"
    pan_graph_html = chart_pan_graph.to_html(fullhtml=False, output_div="vis-pan-graph")
    charts_html = full_dashboard_chart.to_html(fullhtml=False, output_div="vis-dashboard")

    db_rows_html = ""
    for _idx, (_spec, _loc) in enumerate(zip(pneumo_db.specs, pneumo_db.loci, strict=True)):
        db_rows_html += (
            f"<tr>"
            f"<td>{_idx + 1}</td>"
            f"<td><strong>{_spec.serotype}</strong></td>"
            f"<td>{_spec.serogroup}</td>"
            f"<td><code>{_spec.accession}</code></td>"
            f"<td>{_spec.strain}</td>"
            f"<td><code>{_loc.archetype.name}</code></td>"
            f"<td>{_loc.num_genes}</td>"
            f"<td>{_loc.start:,}..{_loc.end:,} ({_loc.strand:+d})</td>"
            f"<td>{_loc.completeness * 100:.1f}%</td>"
            f"<td>{_loc.score:.1f}</td>"
            f"</tr>\n"
        )

    wgs_rows_html = ""
    for _r in wgs_records:
        _badge = "✅ PASS" if _r["is_correct"] else "❌ FAIL"
        wgs_rows_html += (
            f"<tr>"
            f"<td><strong>{_r['strain']}</strong></td>"
            f"<td><code>{_r['accession']}</code></td>"
            f"<td>{int(_r['genome_bp']):,} bp</td>"
            f"<td><strong>{_r['true_serotype']}</strong></td>"
            f"<td><strong>{_r['predicted_serotype']}</strong></td>"
            f"<td>{_r['runner_up_serotype']} ({float(_r['top2_score']):.3f})</td>"
            f"<td><code>{_r['archetype']}</code></td>"
            f"<td>{_r['num_genes']}</td>"
            f"<td>{float(_r['top1_score']):.3f}</td>"
            f"<td>{float(_r['top1_jaccard']):.3f}</td>"
            f"<td><code>{_r['match_flags']}</code></td>"
            f"<td>{_badge}</td>"
            f"</tr>\n"
        )

    frag_rows_html = ""
    for _r in frag_records:
        _badge = "✅ PASS" if _r["is_correct"] else "❌ FAIL"
        frag_rows_html += (
            f"<tr>"
            f"<td><strong>{_r['strain']}</strong></td>"
            f"<td>{_r['true_serotype']}</td>"
            f"<td>{_r['mode']}</td>"
            f"<td><strong>{_r['predicted_serotype']}</strong></td>"
            f"<td>{float(_r['score']):.3f}</td>"
            f"<td>{float(_r['jaccard']):.3f}</td>"
            f"<td>{float(_r['completeness']) * 100:.1f}%</td>"
            f"<td>{_r['num_contigs']}</td>"
            f"<td><code>{_r['has_fragmented_flag']}</code></td>"
            f"<td>{_badge}</td>"
            f"</tr>\n"
        )

    loo_rows_html = ""
    for _r in loo_records:
        _badge = "✅ NOVEL FLAGGED" if _r["is_novel_detected"] else "⚠️ UNFLAGGED"
        loo_rows_html += (
            f"<tr>"
            f"<td><strong>{_r['strain']}</strong></td>"
            f"<td><strong>{_r['true_serotype']}</strong></td>"
            f"<td>{float(_r['full_db_score']):.3f}</td>"
            f"<td><strong>{_r['held_out_closest_serotype']}</strong> (Serogroup {_r['serogroup']}: {_r['held_out_same_serogroup']})</td>"
            f"<td>{float(_r['held_out_score']):.3f}</td>"
            f"<td>-{float(_r['score_drop']):.3f}</td>"
            f"<td>-{_r['held_out_missing_genes']} / +{_r['held_out_extra_genes']}</td>"
            f"<td><code>{_r['held_out_flags']}</code></td>"
            f"<td>{_badge}</td>"
            f"</tr>\n"
        )

    pan_rows_html = ""
    for _r in top_pan_rows:
        pan_rows_html += (
            f"<tr>"
            f"<td><code>CID-{_r['cluster_id']:03d}</code></td>"
            f"<td><code>{_r['role_abbrev']}</code></td>"
            f"<td>{_r['role_name']}</td>"
            f"<td>{_r['category']}</td>"
            f"<td>{_r['locus_count']} / {n_db_loci}</td>"
            f"<td><strong>{float(_r['frequency_pct']):.1f}%</strong></td>"
            f"</tr>\n"
        )

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>SeroGen — S. pneumoniae Reference Database &amp; Whole-Genome Serotyping Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 24px auto;
      max-width: 1240px;
      padding: 0 20px;
      color: #1f2937;
      background: #f9fafb;
    }}
    h1, h2 {{ color: #111827; }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 14px;
      margin: 18px 0 28px 0;
    }}
    .kpi-card {{
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 14px 16px;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }}
    .kpi-label {{ font-size: 12px; text-transform: uppercase; color: #6b7280; font-weight: 600; }}
    .kpi-value {{ font-size: 24px; font-weight: 700; color: #111827; margin: 4px 0; }}
    .kpi-sub {{ font-size: 12px; color: #4b5563; }}
    .chart-box {{
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 28px;
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      margin-bottom: 28px;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 12px;
      border-bottom: 1px solid #e5e7eb;
      text-align: left;
    }}
    th {{ background: #f3f4f6; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>SeroGen — <em>Streptococcus pneumoniae</em> Reference Database &amp; Whole-Genome Benchmark Report</h1>
  <p>
    End-to-end validation of <code>LocusPredictor</code>, <code>Loci.cluster_genes()</code>,
    <code>Loci.as_pan_graph()</code>, <code>LayoutEngine</code>, and <code>LocusMatcher</code> across NCBI
    <em>cps</em> reference loci (Bentley et al. 2006 + post-2006 expansions, cached via <code>cassetter</code> +
    <code>httpx2</code>) and closed RefSeq chromosomes.
  </p>

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">HTTP/2 NCBI Fetch</div>
      <div class="kpi-value">{download_elapsed_s:.2f}s</div>
      <div class="kpi-sub">{n_db_loci} cps loci + {wgs_total_count} WGS ({total_downloaded_mb:.1f} MB)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Pan-Locus Graph DB</div>
      <div class="kpi-value">{unique_clusters} clusters</div>
      <div class="kpi-sub">{total_db_genes} genes &rarr; {n_pan_nodes} pan-nodes ({compression_ratio:.2f}x)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">WGS Serotyping Top-1</div>
      <div class="kpi-value">{wgs_accuracy_pct:.1f}%</div>
      <div class="kpi-sub">{wgs_correct_count} / {wgs_total_count} chromosomes (incl. R6 control)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Contig-Split Stitching</div>
      <div class="kpi-value">{frag_accuracy_pct:.1f}%</div>
      <div class="kpi-sub">{frag_correct_count} / {len(frag_split_only)} intergenic &amp; mid-ORF splits</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">LOO Novelty Detection</div>
      <div class="kpi-value">{loo_accuracy_pct:.1f}%</div>
      <div class="kpi-sub">{loo_novel_count} / {len(loo_records)} held-out serotypes flagged</div>
    </div>
  </div>

  <h2>Stage 1B: Interactive Consensus Pan-K-Locus Synteny Network (Fruchterman-Reingold LayoutEngine)</h2>
  <div class="chart-box">
    {pan_graph_html}
  </div>

  <h2>Interactive 4-Stage Diagnostic Dashboard (Altair / Vega-Lite)</h2>
  <div class="chart-box">
    {charts_html}
  </div>

  <h2>Stage 2: End-to-End Whole-Genome Serotyping &amp; Unencapsulated R6 Negative Control</h2>
  <table>
    <thead>
      <tr>
        <th>Strain</th><th>RefSeq Chr</th><th>Genome Size</th>
        <th>True Serotype</th><th>Predicted (Top-1)</th><th>Runner-Up (Top-2)</th>
        <th>Archetype</th><th>Genes</th><th>Score</th><th>Jaccard</th>
        <th>MatchType Bitflags</th><th>Status</th>
      </tr>
    </thead>
    <tbody>{wgs_rows_html}</tbody>
  </table>

  <h2>Stage 3: Simulated Fragmented-Assembly Stress-Test (Intergenic &amp; Mid-ORF Contig Breaks)</h2>
  <table>
    <thead>
      <tr>
        <th>Strain</th><th>True Serotype</th><th>Assembly Condition</th>
        <th>Predicted Serotype</th><th>Match Score</th><th>Jaccard</th>
        <th>Completeness</th><th>Contigs Spanned</th><th>FRAGMENTED_VARIANT</th><th>Status</th>
      </tr>
    </thead>
    <tbody>{frag_rows_html}</tbody>
  </table>

  <h2>Stage 4: Leave-One-Out (LOO) Novelty Detection &amp; Serogroup Fallback</h2>
  <table>
    <thead>
      <tr>
        <th>Query Strain</th><th>Held-Out True Serotype</th><th>Full DB Score</th>
        <th>Closest Remaining DB Hit</th><th>Held-Out Score</th><th>Score Drop</th>
        <th>Missing / Extra Genes</th><th>Held-Out MatchType Flags</th><th>Novelty Detection</th>
      </tr>
    </thead>
    <tbody>{loo_rows_html}</tbody>
  </table>

  <h2>Stage 1A: Indexed S. pneumoniae Reference cps Database ({n_db_loci} Serotypes, Built in {pneumo_db.build_elapsed_s:.2f}s)</h2>
  <table>
    <thead>
      <tr>
        <th>#</th><th>Serotype</th><th>Serogroup</th><th>NCBI Accession</th>
        <th>Reference Strain</th><th>Archetype</th><th>Genes</th>
        <th>Segmented Interval</th><th>Completeness</th><th>DP Score</th>
      </tr>
    </thead>
    <tbody>{db_rows_html}</tbody>
  </table>

  <h2>Stage 1B Summary: Top Conserved Core Gene Clusters in Consensus Pan-Locus Graph ({n_pan_nodes} Nodes, {n_pan_edges} Edges)</h2>
  <table>
    <thead>
      <tr>
        <th>Cluster ID</th><th>Abbrev</th><th>Predicted Gene Role</th>
        <th>Category</th><th>Locus Presence</th><th>Core Frequency</th>
      </tr>
    </thead>
    <tbody>{pan_rows_html}</tbody>
  </table>
</body>
</html>
"""
    report_path.write_text(report_html, encoding="utf-8")
    report_kb = report_path.stat().st_size / 1024.0

    if is_script_mode:
        print(
            f"[SeroGen] Downloaded {n_db_loci} cps reference loci + {wgs_total_count} complete chromosomes "
            f"({total_downloaded_mb:.1f} MB) via HTTP/2 in {download_elapsed_s:.2f}s"
        )
        print(
            f"[SeroGen] Built PneumoSerotypeDatabase in {pneumo_db.build_elapsed_s:.2f}s: "
            f"{total_db_genes} genes -> {unique_clusters} clusters -> "
            f"{n_pan_nodes} pan-nodes & {n_pan_edges} edges ({compression_ratio:.2f}x compression)"
        )
        print(
            f"[SeroGen] Stage 2 Whole-Genome Serotyping Accuracy: "
            f"{wgs_correct_count}/{wgs_total_count} ({wgs_accuracy_pct:.1f}%)"
        )
        print(
            f"[SeroGen] Stage 3 Contig-Split Stitching Recovery: "
            f"{frag_correct_count}/{len(frag_split_only)} ({frag_accuracy_pct:.1f}%)"
        )
        print(
            f"[SeroGen] Stage 4 Leave-One-Out Novelty Detection: "
            f"{loo_novel_count}/{len(loo_records)} ({loo_accuracy_pct:.1f}%)"
        )
        print(f"[SeroGen] Exported standalone HTML report -> {report_path} ({report_kb:.1f} KB)")

    mo.output.replace(mo.md(f"**Exported standalone HTML report** to `{report_path}` (`{report_kb:.1f} KB`)."))
    return (
        charts_html,
        db_rows_html,
        frag_rows_html,
        loo_rows_html,
        pan_graph_html,
        pan_rows_html,
        report_html,
        report_kb,
        report_path,
        wgs_rows_html,
    )


if __name__ == "__main__":
    app.run()
