"""Tests for serogen.loci and serogen.genes classification, segmentation, and graph operations."""

from pathlib import Path

import numpy as np
import pytest

from serogen.core.genome import GenomeAssembly
from serogen.core.graph.arborescence import (
    k_best_arborescences,
    maximum_spanning_arborescence,
    minimum_spanning_arborescence,
)
from serogen.core.graph.gnn import (
    RelationalMPNN,
    build_multihop_csr,
    sinkhorn_bipartite_matching,
)
from serogen.core.interval import Intervals
from serogen.core.seq.clustering import Clusterer
from serogen.core.seq.seq import Sequences
from serogen.genes.cnn import GeneRoleClassifier
from serogen.genes.models import Genes
from serogen.genes.roles import GeneRole
from serogen.loci.archetypes import (
    ArchetypeCategory,
    ArchetypeTensors,
    LocusArchetype,
    get_archetype_tensors,
    mask_to_roles,
    roles_to_mask,
)
from serogen.loci.classifier import ArchetypeClassifier
from serogen.loci.matcher import LocusMatcher, Match, Matches, MatchType
from serogen.loci.models import GeneNode, Loci, Locus
from serogen.loci.predictor import LocusPredictor
from serogen.loci.segmenter import LocusSegmenter


def test_archetype_ontology_and_bitmasks() -> None:
    """Verify LocusArchetype metadata, ArchetypeTensors, and discrete bitmask classification."""
    cps_roles = [
        GeneRole.TRANSLOCON,
        GeneRole.KINASE,
        GeneRole.FLIPPASE,
        GeneRole.POLYMERASE,
        GeneRole.INITIATOR,
        GeneRole.GLYCOSYLTRANSFERASE,
        GeneRole.EPIMERASE,
    ]
    mask = roles_to_mask(cps_roles)
    unpacked = mask_to_roles(mask)
    assert set(unpacked) == set(cps_roles)

    tensors = get_archetype_tensors()
    assert isinstance(tensors, ArchetypeTensors)
    assert tensors.module_indicators.shape[0] == len(LocusArchetype)
    assert tensors.module_indicators.shape[2] == len(GeneRole)

    clf = ArchetypeClassifier()
    complete = clf.is_complete(mask)
    assert bool(complete[LocusArchetype.CPS_WZY]) is True
    assert bool(complete[LocusArchetype.CPS_ABC]) is False

    arch, comp = clf.classify_mask(mask)
    assert arch == LocusArchetype.CPS_WZY
    assert comp == 1.0
    assert LocusArchetype.from_name("K_locus") == LocusArchetype.CPS_WZY
    assert LocusArchetype.CPS_WZY.category == ArchetypeCategory.CAPSULE


def test_archetype_classifier_interface() -> None:
    """Verify ArchetypeClassifier works unfitted and supports optional fit() and score()."""
    cps_roles = np.array(
        [
            GeneRole.TRANSLOCON,
            GeneRole.KINASE,
            GeneRole.FLIPPASE,
            GeneRole.POLYMERASE,
            GeneRole.INITIATOR,
            GeneRole.GLYCOSYLTRANSFERASE,
        ],
        dtype=np.int32,
    )
    lps_abc_roles = np.array(
        [
            GeneRole.ABC_PERMEASE,
            GeneRole.ABC_ATPASE,
            GeneRole.GLYCOSYLTRANSFERASE,
            GeneRole.EPIMERASE,
        ],
        dtype=np.int32,
    )

    clf = ArchetypeClassifier(min_completeness=0.8)

    batch_X = [cps_roles, lps_abc_roles]
    scores, comps, incomps = clf.score_components(batch_X)
    assert scores.shape == (2, len(LocusArchetype))
    assert comps.shape == (2, len(LocusArchetype))
    assert incomps.shape == (2, len(LocusArchetype))

    preds = clf.predict(batch_X)
    assert list(preds) == [int(LocusArchetype.CPS_WZY), int(LocusArchetype.LPS_ABC)]
    assert clf.score(batch_X, [LocusArchetype.CPS_WZY, LocusArchetype.LPS_ABC]) == 1.0

    probs = clf.predict_proba(batch_X)
    assert probs.shape == (2, len(LocusArchetype))
    assert np.allclose(probs.sum(axis=1), 1.0)

    clf.fit(batch_X, [LocusArchetype.CPS_WZY, LocusArchetype.LPS_ABC])
    assert hasattr(clf, "w_comp_")
    assert hasattr(clf, "w_incomp_")
    assert list(clf.predict(batch_X)) == [int(LocusArchetype.CPS_WZY), int(LocusArchetype.LPS_ABC)]


def test_gene_role_classifier_train_predict_and_io(tmp_path: Path) -> None:
    """Verify GeneRoleClassifier training, prediction, serialization, and pretrained loading."""
    seqs = Sequences.from_bytes(
        [
            b"MKKLLVLLPLLAASAAAWZAWZAWZAWZAWZAWZA",
            b"MKKLLVLLPLLAASAAAWZAWZAWZAWZAWZAWZAKK",
            b"MRVLVTGGAGFIGSHVLRLLEAGHEVVVLDNLCNSKR",
            b"MRVLVTGGAGFIGSHVLRLLEAGHEVVVLDNLCNSKRVV",
        ]
    )
    y = np.array(
        [
            int(GeneRole.TRANSLOCON),
            int(GeneRole.TRANSLOCON),
            int(GeneRole.EPIMERASE),
            int(GeneRole.EPIMERASE),
        ],
        dtype=np.int64,
    )

    clf = GeneRoleClassifier(
        embed_dim=8,
        num_filters=12,
        kernel_size=(3, 5, 7),
        learning_rate=0.02,
        epochs=15,
        batch_size=2,
        random_state=42,
    )
    assert not clf.is_fitted
    clf.fit(seqs, y)
    assert clf.is_fitted

    logits = clf.decision_function(seqs)
    assert logits.shape == (4, 2)
    probs = clf.predict_proba(seqs)
    assert probs.shape == (4, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    preds = clf.predict(seqs)
    assert preds.shape == (4,)
    assert clf.score(seqs, y) == 1.0

    model_path = tmp_path / "test_model.npz"
    clf.save(model_path)
    loaded = GeneRoleClassifier.load(model_path)
    assert loaded.is_fitted
    assert loaded.kernel_size == (3, 5, 7)
    assert np.allclose(loaded.decision_function(seqs), logits, atol=1e-5)

    pretrained = GeneRoleClassifier.pretrained()
    assert pretrained.is_fitted
    pre_probs = pretrained.predict_proba(seqs)
    assert pre_probs.shape == (4, pretrained.num_classes)


def test_gene_role_classifier_transform_and_int64_labels(tmp_path: Path) -> None:
    """Verify int64/32-bit labels, sample_weight, transform, and init_from."""
    seqs = Sequences.from_bytes(
        [
            b"MKKLLVLLPLLAASAAAWZAWZAWZAWZAWZAWZA",
            b"MKKLLVLLPLLAASAAAWZAWZAWZAWZAWZAWZAKK",
            b"MRVLVTGGAGFIGSHVLRLLEAGHEVVVLDNLCNSKR",
            b"MRVLVTGGAGFIGSHVLRLLEAGHEVVVLDNLCNSKRVV",
        ]
    )
    tag_a = (1 << 31) | 197
    tag_b = (1 << 31) | 262
    y_large = np.array([tag_a, tag_a, tag_b, tag_b], dtype=np.int64)
    sample_w = np.array([1.0, 0.9, 1.0, 0.8], dtype=np.float32)

    base_clf = GeneRoleClassifier(
        embed_dim=8,
        num_filters=12,
        kernel_size=(3, 5),
        learning_rate=0.02,
        epochs=15,
        batch_size=2,
        random_state=42,
    )
    base_clf.fit(seqs, y_large, sample_weight=sample_w, y_source=[0, 0, 1, 1])
    assert base_clf.classes_.dtype == np.int64
    assert list(base_clf.classes_) == [tag_a, tag_b]
    assert base_clf.score(seqs, y_large) == 1.0
    assert base_clf.feature_dim == 2 * base_clf.num_filters

    feats = base_clf.transform(seqs)
    assert feats.shape == (4, base_clf.feature_dim)
    embed_mean = base_clf.transform(seqs, layer="embed_mean")
    assert embed_mean.shape == (4, base_clf.embed_dim)
    manual_logits = feats @ base_clf.dense_w_ + base_clf.dense_b_
    assert np.allclose(manual_logits, base_clf.decision_function(seqs), atol=1e-5)

    src_probs = base_clf.predict_source_proba(seqs)
    assert src_probs.shape == (4,)
    assert list(base_clf.predict_source(seqs)) == [0, 0, 1, 1]

    fine_clf = GeneRoleClassifier(
        embed_dim=8,
        num_filters=12,
        kernel_size=(3, 5),
        learning_rate=0.02,
        epochs=10,
        batch_size=2,
        random_state=7,
    )
    fine_clf.fit(seqs, y_large, init_from=base_clf, backbone_lr_scale=0.0)
    assert np.allclose(fine_clf.embedding_, base_clf.embedding_)
    assert np.allclose(fine_clf.conv_w_, base_clf.conv_w_)

    p = tmp_path / "large_classes.npz"
    base_clf.save(p)
    reloaded = GeneRoleClassifier.load(p)
    assert reloaded.classes_.dtype == np.int64
    assert list(reloaded.predict(seqs)) == list(y_large)
    assert np.allclose(reloaded.predict_source_proba(seqs), src_probs, atol=1e-5)


def test_locus_and_loci_soa_and_graph_operations() -> None:
    """Verify unified Locus and Loci SoA and GraphBatch methods."""
    loci = Loci.from_list(
        [
            Locus.from_genes(["wza", "wzb", "wzc", "wzx", "wzy", "wbaP"], strands=[1, 1, 1, 1, 1, 1]),
            Locus.from_genes(["wzm", "wzt", "rmlA", "rmlB"], strands=[-1, -1, -1, -1]),
            Locus.from_genes(["wzx", "wzy", "wzz", "wbaP", "tnpA", "rmlC"], strands=[1, 1, 1, 1, 1, 1]),
        ]
    )
    assert len(loci) == 3
    assert len(loci.genes) == 16

    loc0 = loci[0]
    assert isinstance(loc0, Locus)
    assert loc0.num_genes == 6
    assert len(loc0.genes) == 6

    loc2 = loci[2]
    assert loc2.has_mobile_elements is True
    healed = loc2.heal_mobile_elements(remove_nodes=True)
    assert healed.has_mobile_elements is False
    assert healed.num_genes == 5
    assert len(healed.genes) == 5

    sub = loci[[2, 0]]
    assert len(sub) == 2
    assert sub[0] == loc2
    assert sub[1] == loc0
    assert len(sub.genes) == 12

    concat_loci = Loci.concat([loci[:1], loci[1:]])
    assert len(concat_loci) == 3
    assert concat_loci[1] == loci[1]

    pan_adj, pan_nodes, _ = loci.as_pan_graph()
    assert pan_adj.shape == (len(pan_nodes), len(pan_nodes))


def test_locus_segmenter_tandem_and_gap_bridging() -> None:
    """Verify LocusSegmenter disambiguates tandem K+O loci, bridges internal gaps, and respects contigs."""
    roles_contig0 = [
        GeneRole.UNKNOWN,
        GeneRole.UNKNOWN,
        GeneRole.UNKNOWN,
        GeneRole.TRANSLOCON,
        GeneRole.KINASE,
        GeneRole.FLIPPASE,
        GeneRole.UNKNOWN,
        GeneRole.POLYMERASE,
        GeneRole.INITIATOR,
        GeneRole.GLYCOSYLTRANSFERASE,
        GeneRole.PYRUVYLTRANSFERASE,
        GeneRole.ABC_PERMEASE,
        GeneRole.ABC_ATPASE,
        GeneRole.GLYCOSYLTRANSFERASE,
        GeneRole.OXIDOREDUCTASE,
        GeneRole.EPIMERASE,
    ]
    n0 = len(roles_contig0)
    starts0 = np.arange(n0, dtype=np.int32) * 1200
    ends0 = starts0 + 1000
    strands0 = np.ones(n0, dtype=np.int8)

    roles_contig1 = [
        GeneRole.FLIPPASE,
        GeneRole.POLYMERASE,
        GeneRole.COPOLYMERASE,
        GeneRole.INITIATOR,
        GeneRole.GLYCOSYLTRANSFERASE,
        GeneRole.NUCLEOTIDYLTRANSFERASE,
    ]
    n1 = len(roles_contig1)
    starts1 = np.arange(n1, dtype=np.int32) * 1100
    ends1 = starts1 + 950
    strands1 = -np.ones(n1, dtype=np.int8)

    all_roles = np.array(roles_contig0 + roles_contig1, dtype=np.int32)
    intervals = Intervals(
        starts=np.concatenate([starts0, starts1]),
        ends=np.concatenate([ends0, ends1]),
        strands=np.concatenate([strands0, strands1]),
    )
    contig_indices = np.concatenate(
        [
            np.zeros(n0, dtype=np.uint32),
            np.ones(n1, dtype=np.uint32),
        ]
    )
    genes = Genes(intervals=intervals, contig_indices=contig_indices)

    seg = LocusSegmenter(min_completeness=0.80, max_gap_genes=2)
    loci = seg.segment(genes, all_roles)

    assert isinstance(loci, Loci)
    assert len(loci) == 3
    assert list(loci.archetypes) == [
        int(LocusArchetype.CPS_WZY),
        int(LocusArchetype.LPS_ABC),
        int(LocusArchetype.LPS_WZY),
    ]

    assert list(loci.contig_indices) == [0, 0, 1]
    assert list(loci.is_tandem_split) == [True, True, False]

    assert loci[0].num_genes == 8
    assert len(loci[0].genes) == 8
    assert loci[0].archetype == LocusArchetype.CPS_WZY
    assert loci[1].archetype == LocusArchetype.LPS_ABC
    assert loci[2].archetype == LocusArchetype.LPS_WZY
    assert loci[2].strand == -1
    assert len(loci[:1].genes) == 8


def test_locus_predictor_mge_bridging_and_healing() -> None:
    """Verify LocusSegmenter and LocusPredictor bridge IS transposase insertions and support heal_mobile_elements."""
    # KL106-like CPS_WZY locus interrupted mid-operon by an IS5 transposase + disrupted fragment + second transposase
    roles = [
        GeneRole.NUCLEOTIDYLTRANSFERASE,
        GeneRole.CAPSULE_ANCHOR,
        GeneRole.TRANSLOCON,
        GeneRole.PHOSPHATASE,
        GeneRole.KINASE,
        GeneRole.TRANSPOSASE,
        GeneRole.GLYCOSYLTRANSFERASE,
        GeneRole.GLYCOSYLTRANSFERASE,
        GeneRole.UNKNOWN,
        GeneRole.UNKNOWN,
        GeneRole.TRANSPOSASE,
        GeneRole.GLYCOSYLTRANSFERASE,
        GeneRole.FLIPPASE,
        GeneRole.GLYCOSYLTRANSFERASE,
        GeneRole.INITIATOR,
        GeneRole.OXIDOREDUCTASE,
        GeneRole.POLYMERASE,
        GeneRole.OXIDOREDUCTASE,
    ]
    n = len(roles)
    starts = np.arange(n, dtype=np.int32) * 1100
    ends = starts + 950
    strands = np.ones(n, dtype=np.int8)
    strands[9] = -1  # opposite-strand disrupted stub adjacent to TRANSPOSASE
    genes = Genes(intervals=Intervals(starts=starts, ends=ends, strands=strands))

    seg = LocusSegmenter(min_completeness=0.80, stitch_contig_splits=False)
    loci_default = seg.segment(genes, roles)
    assert len(loci_default) == 1
    loc = loci_default[0]
    assert loc.archetype == LocusArchetype.CPS_WZY
    assert loc.num_genes == n
    assert loc.has_mobile_elements is True
    assert bool(loci_default.has_mge[0]) is True

    loci_healed = loci_default.heal_mobile_elements(remove_nodes=True)
    assert len(loci_healed) == 1
    assert loci_healed[0].num_genes == n - 2
    assert loci_healed[0].has_mobile_elements is False
    assert bool(loci_healed.has_mge[0]) is False


def test_sgh10_tandem_k1_o1() -> None:
    """Verify exact K1 Capsule + O1 O-Antigen tandem segmentation on SGH10."""
    sgh10_path = Path(__file__).parent / "data" / "SGH10_chromosome.fna.zst"
    if sgh10_path.exists():
        asm = GenomeAssembly.from_file(sgh10_path)
        predictor = LocusPredictor()
        all_genes = predictor.orf_predictor.predict(asm)
        region_mask = (all_genes.intervals.starts >= 1_700_000) & (all_genes.intervals.ends <= 1_750_000)
        region_loci = predictor.predict(all_genes[region_mask])

        assert len(region_loci) == 2
        k1, o1 = region_loci[0], region_loci[1]
        assert k1.archetype == LocusArchetype.CPS_WZY
        assert k1.start == 1_706_482
        assert k1.end == 1_731_461
        assert k1.strand == 1
        assert k1.is_tandem_split is True

        assert o1.archetype == LocusArchetype.LPS_ABC
        assert o1.start == 1_734_456
        assert o1.end == 1_742_380
        assert o1.strand == 1
        assert o1.is_tandem_split is True


def test_locus_predictor_pipeline() -> None:
    """Verify LocusPredictor role probability projection and empty/Genes handling."""
    predictor = LocusPredictor(min_completeness=0.50)
    assert predictor.archetype_classifier is predictor.segmenter.classifier
    assert len(predictor.predict(Genes.empty())) == 0

    translations = Sequences.from_bytes(
        [
            b"MKKLLVLLPLLAASAAAWZAWZAWZAWZAWZAWZA",
            b"MRVLVTGGAGFIGSHVLRLLEAGHEVVVLDNLCNSKR",
            b"MKKLLVLLPLLAASAAAWZAWZAWZAWZAWZAWZAKK",
        ]
    )
    intervals = Intervals(
        starts=np.array([0, 1200, 2400], dtype=np.int32),
        ends=np.array([1000, 2200, 3400], dtype=np.int32),
        strands=np.array([1, 1, 1], dtype=np.int8),
    )
    genes = Genes(intervals=intervals, translations=translations, contig_indices=np.zeros(3, dtype=np.uint32))
    probs = predictor.predict_role_probabilities(genes)
    assert probs.shape == (3, len(GeneRole))
    assert np.allclose(probs.sum(axis=1), 1.0)
    loci = predictor.predict(genes)
    assert isinstance(loci, Loci)


def test_pan_locus_graph_pipeline() -> None:
    """Verify strand canonicalization, MGE healing commutativity, protein clustering, and pan-locus graph."""
    loc_fwd = Locus.from_genes(
        ["wza", "wzb", "wzc", "wbaP"],
        strands=[1, 1, 1, 1],
    )
    loc_rev_mge = Locus.from_genes(
        ["wbaP", "tnpA", "wzc", "wzb", "wza"],
        strands=[-1, -1, -1, -1, -1],
    )
    assert loc_rev_mge.strand == -1

    c_then_h = loc_rev_mge.canonicalize_strand().heal_mobile_elements()
    h_then_c = loc_rev_mge.heal_mobile_elements().canonicalize_strand()
    assert c_then_h == h_then_c
    assert c_then_h.strand == 1
    assert list(c_then_h.roles) == list(loc_fwd.roles)
    assert c_then_h.canonicalize_strand() == c_then_h

    prot_seqs = [
        b"MKKLLVLLPLLAASAAAWZAWZAWZAWZAWZAWZA" * 3,
        b"MKLNIIVCTGGNRSIGKSVLALLQERGYQVRSMP" * 3,
        b"MNTQVLSVVLAALLLLVAGAWZCWZCWZCWZCWZC" * 3,
        b"MKRIALLFLLPLLVLAALWBAPWBAPWBAPWBAP" * 3,
    ]
    tnp_seq = b"MPRRSKKEAAQRLLEAGYTVRTVRRATNPTNPTNP" * 3

    genes_fwd = Genes(
        intervals=loc_fwd.genes.intervals,
        translations=Sequences.from_bytes(prot_seqs),
    )
    genes_rev = Genes(
        intervals=loc_rev_mge.genes.intervals,
        translations=Sequences.from_bytes([prot_seqs[3], tnp_seq, prot_seqs[2], prot_seqs[1], prot_seqs[0]]),
    )
    loc_a = Locus.from_genes(genes_fwd, roles=loc_fwd.roles)
    loc_b = Locus.from_genes(genes_rev, roles=loc_rev_mge.roles)

    batch = Loci.from_list([loc_a, loc_b])
    adj, pan_nodes, node_counts = batch.as_pan_graph(
        node_mask=GeneNode.mask("ROLE_ID", "CLUSTER_ID"),
        canonicalize_strands=True,
        heal_mges=True,
        cluster_genes=True,
    )
    assert len(pan_nodes) == 4
    assert list(node_counts) == [2, 2, 2, 2]
    assert int(adj.sum()) == 6
    assert list(adj.data) == [2, 2, 2]
    for pn in pan_nodes:
        assert GeneNode(int(pn)).is_clustered is True


def test_locus_matcher_variants_and_novelty() -> None:
    """Verify MatchType bitflags, Matches SoA methods, and LocusMatcher novelty detection."""
    ref_kl1 = Locus.from_genes(
        ["wza", "wzb", "wzc", "wbaP", "wzx", "wzy"],
        strands=[1, 1, 1, 1, 1, 1],
        clusters=[1, 2, 3, 4, 5, 6],
        archetype=LocusArchetype.CPS_WZY,
    )
    ref_kl2 = Locus.from_genes(
        ["wza", "wzb", "wzc", "wcaJ", "wzx", "wzy"],
        strands=[1, 1, 1, 1, 1, 1],
        clusters=[1, 2, 3, 10, 11, 12],
        archetype=LocusArchetype.CPS_WZY,
    )
    db = Loci.from_list([ref_kl1, ref_kl2])
    matcher = LocusMatcher(db)

    m_exact = ref_kl1.match(ref_kl1)
    assert isinstance(m_exact, Match)
    assert m_exact.match_type == MatchType.EXACT
    assert m_exact.is_exact is True
    assert m_exact.is_novel is False
    assert np.isclose(m_exact.score, 1.0)
    assert m_exact.shared_genes == 6
    assert m_exact.missing_genes == 0
    assert m_exact.extra_genes == 0

    q_strand = Locus.from_genes(
        ["wzy", "wzx", "wbaP", "wzc", "wzb", "wza"],
        strands=[-1, -1, -1, -1, -1, -1],
        clusters=[6, 5, 4, 3, 2, 1],
        archetype=LocusArchetype.CPS_WZY,
    )
    matches_strand = matcher.match(q_strand)
    best_strand = matches_strand.best()
    assert best_strand is not None
    assert best_strand.target_index == 0
    assert best_strand.match_type == MatchType.STRAND_VARIANT
    assert best_strand.is_novel is False

    q_mge = Locus.from_genes(
        ["wza", "wzb", "wzc", "wbaP", "tnpA", "wzx", "wzy"],
        strands=[1, 1, 1, 1, 1, 1, 1],
        clusters=[1, 2, 3, 4, 99, 5, 6],
        archetype=LocusArchetype.CPS_WZY,
    )
    best_mge = matcher.match(q_mge).best()
    assert best_mge is not None
    assert best_mge.target_index == 0
    assert best_mge.match_type == MatchType.MGE_VARIANT
    assert best_mge.is_novel is False

    q_fs = Locus.from_genes(
        ["wza", "wzb", "wzc", "wbaP", "wzx", "wzy"],
        strands=[1, 1, 1, 1, 1, 1],
        clusters=[1, 2, 3, 4, 5, 6],
        frameshifts=[False, False, True, False, False, False],
        archetype=LocusArchetype.CPS_WZY,
    )
    best_fs = matcher.match(q_fs).best()
    assert best_fs is not None
    assert best_fs.match_type == MatchType.FRAMESHIFT_VARIANT
    assert best_fs.is_novel is False

    q_combo = Locus.from_genes(
        ["wzy", "wzx", "tnpA", "wbaP", "wzc", "wzb", "wza"],
        strands=[-1, -1, -1, -1, -1, -1, -1],
        clusters=[6, 5, 99, 4, 3, 2, 1],
        frameshifts=[False, False, False, False, True, False, False],
        archetype=LocusArchetype.CPS_WZY,
    )
    best_combo = matcher.match(q_combo).best()
    assert best_combo is not None
    expected_combo = MatchType.STRAND_VARIANT | MatchType.MGE_VARIANT | MatchType.FRAMESHIFT_VARIANT
    assert best_combo.match_type == expected_combo
    assert best_combo.match_type.flag_count == 3
    assert best_combo.is_novel is False

    q_synteny = Locus.from_genes(
        ["wza", "wzb", "wzc", "wbaP", "wzy", "wzx"],
        strands=[1, 1, 1, 1, 1, 1],
        clusters=[1, 2, 3, 4, 6, 5],
        archetype=LocusArchetype.CPS_WZY,
    )
    best_syn = matcher.match(q_synteny).best()
    assert best_syn is not None
    assert best_syn.match_type == MatchType.SYNTENY_VARIANT
    assert best_syn.is_novel is True

    q_gene_var = Locus.from_genes(
        ["wza", "wzb", "wzc", "wbaZ", "wzx", "wzy"],
        strands=[1, 1, 1, 1, 1, 1],
        clusters=[1, 2, 3, 50, 5, 6],
        archetype=LocusArchetype.CPS_WZY,
    )
    best_gv = matcher.match(q_gene_var).best()
    assert best_gv is not None
    assert bool(best_gv.match_type & MatchType.GENE_VARIANT)
    assert best_gv.shared_genes == 5
    assert best_gv.missing_genes == 1
    assert best_gv.extra_genes == 1
    assert best_gv.is_novel is True

    q_batch = Loci.from_list([ref_kl1, q_combo, q_gene_var])
    all_matches = db.match(q_batch)
    assert isinstance(all_matches, Matches)
    assert len(all_matches) == 6
    top_per_q = all_matches.best_per_query()
    assert len(top_per_q) == 3
    assert list(top_per_q.target_indices) == [0, 0, 0]
    assert list(top_per_q.is_novel) == [False, False, True]

    re_concat = Matches.concat([top_per_q[:1], top_per_q[1:]])
    assert len(re_concat) == 3
    assert re_concat[1] == top_per_q[1]


def test_locus_matcher_protein_sequences_and_align() -> None:
    """Verify LocusMatcher automatic protein indexing, on-the-fly query cluster assignment, and GraphAligner."""
    prot_seqs = [
        b"MKKLLVLLPLLAASAAAWZAWZAWZAWZAWZAWZA" * 3,
        b"MKLNIIVCTGGNRSIGKSVLALLQERGYQVRSMP" * 3,
        b"MNTQVLSVVLAALLLLVAGAWZCWZCWZCWZCWZC" * 3,
        b"MKRIALLFLLPLLVLAALWBAPWBAPWBAPWBAP" * 3,
    ]
    novel_seq = b"MYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY" * 3

    base_loc = Locus.from_genes(["wza", "wzb", "wzc", "wbaP"], strands=[1, 1, 1, 1])
    db_genes = Genes(intervals=base_loc.genes.intervals, translations=Sequences.from_bytes(prot_seqs))
    db_loc = Locus.from_genes(db_genes, roles=base_loc.roles, archetype=LocusArchetype.CPS_WZY)

    matcher = LocusMatcher(db_loc)
    assert matcher.use_clusters is True

    rev_base = Locus.from_genes(["wbaP", "wzc", "wzb", "wza"], strands=[-1, -1, -1, -1])
    q_rev_genes = Genes(intervals=rev_base.genes.intervals, translations=Sequences.from_bytes(prot_seqs[::-1]))
    q_rev_loc = Locus.from_genes(q_rev_genes, roles=rev_base.roles, archetype=LocusArchetype.CPS_WZY)

    m1 = matcher.match(q_rev_loc).best()
    assert m1 is not None
    assert m1.match_type == MatchType.STRAND_VARIANT
    assert m1.shared_genes == 4
    assert m1.is_novel is False

    q_nov_genes = Genes(
        intervals=base_loc.genes.intervals,
        translations=Sequences.from_bytes([prot_seqs[0], prot_seqs[1], prot_seqs[2], novel_seq]),
    )
    q_nov_loc = Locus.from_genes(q_nov_genes, roles=base_loc.roles, archetype=LocusArchetype.CPS_WZY)
    m2 = matcher.match(q_nov_loc).best()
    assert m2 is not None
    assert m2.match_type == MatchType.GENE_VARIANT
    assert m2.shared_genes == 3
    assert m2.missing_genes == 1
    assert m2.extra_genes == 1
    assert m2.is_novel is True

    alignment = matcher.align(q_rev_loc, 0)
    assert len(alignment.unmatched_a) == 0
    assert len(alignment.unmatched_b) == 0


def test_locus_segmenter_contig_split_stitching() -> None:
    """Verify LocusSegmenter rescues and stitches loci split across contigs (including strand inversions)."""
    intact_roles = [
        GeneRole.TRANSLOCON,
        GeneRole.KINASE,
        GeneRole.COPOLYMERASE,
        GeneRole.INITIATOR,
        GeneRole.FLIPPASE,
        GeneRole.POLYMERASE,
        GeneRole.OXIDOREDUCTASE,
    ]
    ref_loc = Locus.from_genes(
        intact_roles,
        strands=[1] * 7,
        clusters=[1, 2, 3, 4, 5, 6, 7],
        archetype=LocusArchetype.CPS_WZY,
    )
    matcher = LocusMatcher(ref_loc)

    c0_roles = [GeneRole.UNKNOWN, GeneRole.UNKNOWN, GeneRole.TRANSLOCON, GeneRole.KINASE, GeneRole.COPOLYMERASE]
    c0_starts = np.arange(len(c0_roles), dtype=np.int32) * 1200
    c0_ends = c0_starts + 1000
    c0_strands = np.ones(len(c0_roles), dtype=np.int8)
    c0_cids = [0, 0, 1, 2, 3]

    c1_roles = [
        GeneRole.UNKNOWN,
        GeneRole.UNKNOWN,
        GeneRole.OXIDOREDUCTASE,
        GeneRole.POLYMERASE,
        GeneRole.FLIPPASE,
        GeneRole.INITIATOR,
    ]
    c1_starts = np.arange(len(c1_roles), dtype=np.int32) * 1200
    c1_ends = c1_starts + 1000
    c1_strands = -np.ones(len(c1_roles), dtype=np.int8)
    c1_cids = [0, 0, 7, 6, 5, 4]

    all_roles = np.array(c0_roles + c1_roles, dtype=np.int32)
    all_cids = np.array(c0_cids + c1_cids, dtype=np.int32)
    intervals = Intervals(
        starts=np.concatenate([c0_starts, c1_starts]),
        ends=np.concatenate([c0_ends, c1_ends]),
        strands=np.concatenate([c0_strands, c1_strands]),
    )
    contig_indices = np.concatenate(
        [
            np.zeros(len(c0_roles), dtype=np.uint32),
            np.ones(len(c1_roles), dtype=np.uint32),
        ]
    )
    genes = Genes(intervals=intervals, contig_indices=contig_indices)

    seg_no_stitch = LocusSegmenter(min_completeness=0.80, stitch_contig_splits=False)
    unstitched = seg_no_stitch.segment(genes, all_roles, clusters=all_cids)
    assert len(unstitched) == 1
    assert unstitched[0].archetype == LocusArchetype.LPS_WZY
    assert len(unstitched[0].nodes) == 4

    seg_stitch = LocusSegmenter(min_completeness=0.80, stitch_contig_splits=True)
    stitched_loci = seg_stitch.segment(genes, all_roles, clusters=all_cids)
    assert len(stitched_loci) == 1
    assert list(stitched_loci.is_fragmented) == [True]

    loc = stitched_loci[0]
    assert loc.archetype == LocusArchetype.CPS_WZY
    assert loc.is_fragmented is True
    assert loc.num_contigs == 2
    assert loc.completeness == pytest.approx(1.0)
    assert list(loc.roles) == [int(r) for r in intact_roles]
    assert list(loc.cluster_ids) == [1, 2, 3, 4, 5, 6, 7]
    assert sum(1 for e in loc.edges if e.edge_type.is_contig_break) == 1

    match = matcher.match(loc).best()
    assert match is not None
    assert match.match_type == MatchType.FRAGMENTED_VARIANT
    assert match.jaccard == pytest.approx(1.0)
    assert match.shared_genes == 7
    assert match.is_novel is False


def test_loci_filter_and_allelic_identity() -> None:
    """Verify Loci.filter(categories=...) and allelic AA identity tie-breaking."""
    cps_loc = Locus.from_genes(
        ["wza", "wzb", "wzc", "wbaP", "wzx", "wzy"],
        strands=[1] * 6,
        archetype=LocusArchetype.CPS_WZY,
        completeness=1.0,
        score=42.0,
    )
    lps_loc = Locus.from_genes(
        ["wzm", "wzt", "wecA", "wbbl"],
        strands=[1] * 4,
        archetype=LocusArchetype.LPS_ABC,
        completeness=0.90,
        score=28.0,
    )
    batch = Loci.from_list([cps_loc, lps_loc])
    assert cps_loc.category == ArchetypeCategory.CAPSULE
    assert lps_loc.category == ArchetypeCategory.O_ANTIGEN
    assert list(batch.categories) == [str(ArchetypeCategory.CAPSULE), str(ArchetypeCategory.O_ANTIGEN)]

    only_cps = batch.filter(categories=ArchetypeCategory.CAPSULE)
    assert len(only_cps) == 1
    assert only_cps[0].archetype == LocusArchetype.CPS_WZY
    best_loc = batch.best()
    assert best_loc is not None
    assert best_loc.archetype == LocusArchetype.CPS_WZY

    only_lps = batch.filter(archetypes=[LocusArchetype.LPS_ABC], min_completeness=0.85)
    assert len(only_lps) == 1
    assert only_lps[0].archetype == LocusArchetype.LPS_ABC

    prot_6d = [
        b"MKKLLVLLPLLAASAAAWZAWZAWZAWZAWZAWZA" * 3,
        b"MKLNIIVCTGGNRSIGKSVLALLQERGYQVRSMP" * 3,
        b"MNTQVLSVVLAALLLLVAGAWZCWZCWZCWZCWZC" * 3,
        b"MKRIALLFLLPLLVLAALWBAPWBAPWBAPWBAP" * 3,
    ]
    prot_6c = [
        prot_6d[0],
        prot_6d[1],
        prot_6d[2],
        b"MKRIALLFLLPLLVLAALWBAPWBAPWBAPYYYY" * 3,
    ]
    base_4 = Locus.from_genes(["wza", "wzb", "wzc", "wbaP"], strands=[1, 1, 1, 1])
    loc_6d = Locus.from_genes(
        Genes(intervals=base_4.genes.intervals, translations=Sequences.from_bytes(prot_6d)),
        roles=base_4.roles,
        archetype=LocusArchetype.CPS_WZY,
    )
    loc_6c = Locus.from_genes(
        Genes(intervals=base_4.genes.intervals, translations=Sequences.from_bytes(prot_6c)),
        roles=base_4.roles,
        archetype=LocusArchetype.CPS_WZY,
    )
    db_6 = Loci.from_list([loc_6d, loc_6c]).cluster_genes()
    assert list(db_6[0].cluster_ids) == list(db_6[1].cluster_ids)

    matcher_6 = LocusMatcher(db_6)
    matches_6c = matcher_6.match(loc_6c)
    best_6c = matches_6c[0]
    runner_6c = matches_6c[1]
    assert best_6c.target_index == 1
    assert best_6c.mean_aa_identity == pytest.approx(1.0)
    assert runner_6c.target_index == 0
    assert runner_6c.mean_aa_identity < 1.0
    assert best_6c.score > runner_6c.score


def test_wzg_capsule_anchor_operon_snapping_teichoic_acid_and_predictor_filters() -> None:
    """Verify Wzg/CpsA CAPSULE_ANCHOR classification, outward operon boundary snapping, TEICHOIC_ACID, and predictor filters."""
    assert GeneRole.label("wzg", "integral membrane regulatory protein Wzg") == GeneRole.CAPSULE_ANCHOR
    assert GeneRole.label("cpsA", "capsular polysaccharide biosynthesis protein CpsA") == GeneRole.CAPSULE_ANCHOR
    assert GeneRole.label("cps2A", "capsule regulatory protein") == GeneRole.CAPSULE_ANCHOR
    assert GeneRole.label("capA", "capsule attachment protein") == GeneRole.CAPSULE_ANCHOR
    assert GeneRole.label("", "LytR-CpsA-Psr family capsule attachment protein") == GeneRole.CAPSULE_ANCHOR

    spn23f_wzg_seq = (
        b"MSRRFKKSRSQKVKRSVNIVLLTIYLLLVCFLLFLIFKYNILAFRYLNLVVTALVLLVALVGLLLIIYKKAEKFTIFLLVFSILVSSVSLFA"
        b"VQQFVGLTNRLNATSNYSEYSISVAVLADSDIENVTQLTSVTAPTGTDNENIQKLLADIKSSQNTDLTVDQSSSYLAAYKSLIAGETKAIVL"
        b"NSVFENIIESEYPDYASKIKKIYTKGFTKKVEAPKTSKNQSFNIYVSGIDTYGPISSVSRSDVNILMTVNRDTKKILLTTTPRDAYVPIADG"
        b"GNNQKDKLTHAGIYGVDSSIHTLENLYGVDINYYVRLNFTSFLKMIDLLGGVDVHNDQEFSALHGKFHFPVGNVHLDSEQALGFVRERYSLA"
        b"DGDRDRGRNQQKVIVAILQKLTSTEALKNYSTIIDSLQDSIQTNMPLETMINLVNAQLESGGNYKVNSQDLKGTGRTDLPSYAMPDSNLYVM"
        b"EIDDSSLAVVKAAIQDVMEGR"
    )
    GeneRoleClassifier.pretrained.cache_clear()
    clf = GeneRoleClassifier.pretrained()
    wzg_seq_batch = Sequences.from_bytes([spn23f_wzg_seq])
    wzg_pred = clf.predict(wzg_seq_batch)
    wzg_probs = clf.predict_proba(wzg_seq_batch)
    anchor_col = int(np.where(clf.classes_ == int(GeneRole.CAPSULE_ANCHOR))[0][0])
    assert int(wzg_pred[0]) == int(GeneRole.CAPSULE_ANCHOR)
    assert float(wzg_probs[0, anchor_col]) > 0.75

    n_roles = len(GeneRole)
    probs = np.zeros((11, n_roles), dtype=np.float32)
    probs[0, GeneRole.UNKNOWN] = 0.98
    probs[0, GeneRole.OXIDOREDUCTASE] = 0.02

    probs[1, GeneRole.UNKNOWN] = 0.58
    probs[1, GeneRole.CAPSULE_ANCHOR] = 0.42

    probs[2, GeneRole.UNKNOWN] = 0.58
    probs[2, GeneRole.CAPSULE_ANCHOR] = 0.42

    core_roles = [
        GeneRole.PHOSPHATASE,
        GeneRole.COPOLYMERASE,
        GeneRole.KINASE,
        GeneRole.INITIATOR,
        GeneRole.FLIPPASE,
        GeneRole.POLYMERASE,
    ]
    for offset, r in enumerate(core_roles, start=3):
        probs[offset, GeneRole.UNKNOWN] = 0.04
        probs[offset, r] = 0.96

    probs[9, GeneRole.UNKNOWN] = 0.60
    probs[9, GeneRole.OXIDOREDUCTASE] = 0.40

    probs[10, GeneRole.UNKNOWN] = 0.56
    probs[10, GeneRole.GLYCOSYLTRANSFERASE] = 0.44

    starts = np.array([0, 2000, 3400, 4445, 5500, 6550, 7600, 8650, 9700, 10730, 11755], dtype=np.int32)
    ends = np.array([1000, 3000, 4400, 5445, 6500, 7550, 8600, 9650, 10700, 11730, 12755], dtype=np.int32)
    strands = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1], dtype=np.int8)
    genes = Genes(intervals=Intervals(starts=starts, ends=ends, strands=strands))

    seg_snap = LocusSegmenter(max_operon_snap_bp=150)
    snapped_loci = seg_snap.segment(genes, probs)
    assert len(snapped_loci) == 1
    snapped_loc = snapped_loci[0]
    assert snapped_loc.archetype == LocusArchetype.CPS_WZY
    assert snapped_loc.num_genes == 8
    assert snapped_loc.start == 3400
    assert snapped_loc.end == 11730

    seg_no_snap = LocusSegmenter(max_operon_snap_bp=0)
    unsnapped_loci = seg_no_snap.segment(genes, probs)
    assert len(unsnapped_loci) == 1
    assert unsnapped_loci[0].num_genes == 6
    assert unsnapped_loci[0].start == 4445
    assert unsnapped_loci[0].end == 10700

    assert LocusArchetype.TEICHOIC_ACID.category == ArchetypeCategory.OTHER
    assert LocusArchetype.from_name("teichoic_acid") == LocusArchetype.TEICHOIC_ACID
    assert LocusArchetype.from_name("wta") == LocusArchetype.TEICHOIC_ACID

    wta_roles = [
        GeneRole.ABC_PERMEASE,
        GeneRole.ABC_ATPASE,
        GeneRole.PHOSPHOTRANSFERASE,
        GeneRole.ALANINE_TRANSFERASE,
        GeneRole.NUCLEOTIDYLTRANSFERASE,
        GeneRole.OXIDOREDUCTASE,
        GeneRole.GLYCOSYLTRANSFERASE,
    ]
    arch_clf = ArchetypeClassifier()
    wta_arch, wta_comp = arch_clf.classify_mask(roles_to_mask(wta_roles))
    assert wta_arch == LocusArchetype.TEICHOIC_ACID
    assert wta_comp == pytest.approx(1.0)


def test_gene_role_classifier_layer_sweep_transform_and_topk() -> None:
    """Verify multi-layer feature extraction transform(X, layer=...) and top_k_accuracy."""
    seqs = Sequences.from_bytes(
        [
            b"MKKLLVLLPLLAASAAAWZAWZAWZAWZAWZAWZA",
            b"MKKLLVLLPLLAASAAAWZAWZAWZAWZAWZAWZAKK",
            b"MRVLVTGGAGFIGSHVLRLLEAGHEVVVLDNLCNSKR",
            b"MRVLVTGGAGFIGSHVLRLLEAGHEVVVLDNLCNSKRVV",
        ]
    )
    y = np.array([1, 1, 2, 2], dtype=np.int64)
    clf = GeneRoleClassifier(
        embed_dim=8,
        num_filters=12,
        kernel_size=(3, 5),
        learning_rate=0.02,
        epochs=12,
        batch_size=2,
        random_state=42,
    ).fit(seqs, y)

    assert clf.top_k_accuracy(seqs, y, k=1) == 1.0
    assert clf.top_k_accuracy(seqs, y, k=2) == 1.0

    h_pen = clf.transform(seqs, layer="penultimate")
    h_max = clf.transform(seqs, layer="conv_max")
    h_avg = clf.transform(seqs, layer="conv_avg")
    h_tw = clf.transform(seqs, layer="task_weighted")
    h_tw_max = clf.transform(seqs, layer="task_weighted_conv_max")
    h_emb = clf.transform(seqs, layer="embed_mean")
    h_k5 = clf.transform(seqs, layer="conv_k5")
    h_logits = clf.transform(seqs, layer="logits")
    h_proba = clf.transform(seqs, layer="proba")

    assert h_pen.shape == (4, clf.feature_dim)
    assert h_max.shape == (4, 12)
    assert h_avg.shape == (4, 12)
    assert np.allclose(h_max, h_pen[:, :12])
    assert np.allclose(h_avg, h_pen[:, 12:24])
    assert h_tw.shape == (4, clf.feature_dim)
    assert h_tw_max.shape == (4, 12)
    assert h_emb.shape == (4, 8)
    assert h_k5.shape == (4, 12)
    assert np.allclose(h_logits, clf.decision_function(seqs))
    assert np.allclose(h_proba, clf.predict_proba(seqs))


def test_clusterer_motif_rescue_and_mcl_embedding_blending() -> None:
    """Verify role-gated twilight-zone homology rescue and MCL active-site embedding edge blending."""
    motif = b"DEDDVSVIVTGGAGFIGWYL"
    flank_a = b"MKKILVTG" * 6 + motif + b"ALRLLKAGHEVVVLD" * 6
    flank_b = b"MNRVFIAA" * 6 + motif + b"SIQYYQNNTRIIIFG" * 6
    unrelated = b"MWKLLVLLPLLAASAAAWZAWZAWZAWZAWZAWZA" * 4

    seqs = Sequences.from_bytes([flank_a, flank_b, unrelated])
    roles = np.array(
        [int(GeneRole.GLYCOSYLTRANSFERASE), int(GeneRole.GLYCOSYLTRANSFERASE), int(GeneRole.CAPSULE_ANCHOR)],
        dtype=np.int32,
    )

    c_no_emb = Clusterer(mode="mcl", min_identity=25.0, min_bitscore=45.0).cluster(seqs, roles=roles)
    assert c_no_emb.cluster_ids[0] != c_no_emb.cluster_ids[1]

    embs = np.array(
        [
            [1.0, 0.9, 0.8, 0.0],
            [0.98, 0.92, 0.78, 0.02],
            [0.0, 0.0, 0.1, 1.0],
        ],
        dtype=np.float32,
    )
    c_rescued = Clusterer(
        mode="mcl",
        min_identity=25.0,
        min_bitscore=45.0,
        embedding_weight=0.35,
        motif_rescue_threshold=0.75,
    ).cluster(seqs, roles=roles, embeddings=embs)
    assert c_rescued.cluster_ids[0] == c_rescued.cluster_ids[1]
    assert c_rescued.cluster_ids[0] != c_rescued.cluster_ids[2]


def test_arborescence_edmonds_slot_exclusivity_and_k_best() -> None:
    """Verify Chu-Liu/Edmonds cycle contraction, 3D slot exclusivity, and k-best arborescences."""
    w_2d = np.full((4, 4), -1e9, dtype=np.float64)
    w_2d[0, 1] = 5.0
    w_2d[0, 2] = 4.0
    w_2d[0, 3] = 2.0
    w_2d[1, 2] = 10.0
    w_2d[2, 3] = 10.0
    w_2d[3, 1] = 10.0
    res = maximum_spanning_arborescence(w_2d, root=0)
    parents, score = res
    assert score == pytest.approx(25.0)
    assert list(parents) == [-1, 0, 1, 2]

    min_res = minimum_spanning_arborescence(-w_2d, root=0)
    assert min_res.score == pytest.approx(-25.0)
    assert list(min_res.parents) == [-1, 0, 1, 2]

    w_3d = np.full((3, 3, 2), -1e9, dtype=np.float64)
    w_3d[0, 1, 0] = 10.0
    w_3d[0, 1, 1] = 6.0
    w_3d[0, 2, 0] = 9.0
    w_3d[0, 2, 1] = 8.0
    w_3d[1, 2, 0] = 5.0
    w_3d[2, 1, 0] = 4.0

    res_unconstrained = maximum_spanning_arborescence(w_3d, root=0, slot_exclusivity=False)
    assert res_unconstrained.score == pytest.approx(19.0)
    assert list(res_unconstrained.slots) == [-1, 0, 0]

    res_constrained = maximum_spanning_arborescence(w_3d, root=0, slot_exclusivity=True)
    assert res_constrained.score == pytest.approx(18.0)
    assert list(res_constrained.parents) == [-1, 0, 0]
    assert list(res_constrained.slots) == [-1, 0, 1]

    top3 = k_best_arborescences(w_3d, root=0, k=3, slot_exclusivity=True)
    assert len(top3) == 3
    assert [round(t.score, 4) for t in top3] == [18.0, 15.0, 15.0]


def test_relational_mpnn_forward_backward_clamped_and_sinkhorn(tmp_path: Path) -> None:
    """Verify RelationalMPNN CSR message passing, clamped anchors, tree edge scoring, training, and Sinkhorn."""
    log_alpha = np.array([[2.0, -1.0, 0.0], [-1.0, 3.0, 0.5], [0.0, 0.0, 2.5]], dtype=np.float32)
    p_mat = sinkhorn_bipartite_matching(log_alpha, n_iters=25, tau=0.5)
    assert p_mat.shape == (3, 3)
    assert np.allclose(p_mat.sum(axis=0), 1.0, atol=1e-2)
    assert np.allclose(p_mat.sum(axis=1), 1.0, atol=1e-2)
    assert list(np.argmax(p_mat, axis=1)) == [0, 1, 2]

    node_offsets = np.array([0, 4], dtype=np.int32)
    edge_sources = np.array([0, 1, 2], dtype=np.int32)
    edge_targets = np.array([1, 2, 3], dtype=np.int32)
    csr = build_multihop_csr(node_offsets, edge_sources, edge_targets, include_2hop=True, add_virtual_pool=True)
    assert csr.n_nodes == 5
    assert csr.n_relations == 8

    rng = np.random.default_rng(123)
    x = rng.normal(0.0, 1.0, size=(4, 16)).astype(np.float32)
    node_targets = np.array([0, 1, 2, 3], dtype=np.int32)
    clamped_mask = np.array([1, 0, 0, 0], dtype=np.uint8)
    edge_nodes = np.array([0, 1, 2, 3], dtype=np.int32)
    parent_targets = np.array([-1, 0, 1, 1], dtype=np.int32)
    slot_targets = np.array([-1, 0, 0, 2], dtype=np.int32)

    mpnn = RelationalMPNN(
        in_dim=16,
        hidden_dim=24,
        num_node_classes=5,
        num_edge_channels=4,
        num_relations=8,
        num_layers=2,
        bilinear_rank=12,
        learning_rate=0.015,
        random_state=42,
    )

    loss_initial = mpnn.fit_step(
        x,
        csr,
        node_targets=node_targets,
        edge_node_indices=edge_nodes,
        edge_parent_targets=parent_targets,
        edge_slot_targets=slot_targets,
        clamped_mask=clamped_mask,
    )
    loss_final = loss_initial
    for _ in range(35):
        loss_final = mpnn.fit_step(
            x,
            csr,
            node_targets=node_targets,
            edge_node_indices=edge_nodes,
            edge_parent_targets=parent_targets,
            edge_slot_targets=slot_targets,
            clamped_mask=clamped_mask,
        )
    assert loss_final < loss_initial * 0.25

    probs = mpnn.predict_node_proba(x, csr, clamped_mask=clamped_mask)[:4]
    assert list(np.argmax(probs, axis=1)) == [0, 1, 2, 3]

    h_enc = mpnn.encode(x, csr, clamped_mask=clamped_mask)
    edge_logits = mpnn.score_edges(h_enc, node_indices=edge_nodes)
    tree = maximum_spanning_arborescence(edge_logits, root=0, slot_exclusivity=True)
    assert list(tree.parents) == [-1, 0, 1, 1]
    assert list(tree.slots) == [-1, 0, 0, 2]

    save_path = mpnn.save(tmp_path / "mpnn_test.npz")
    reloaded = RelationalMPNN.load(save_path)
    assert np.allclose(
        reloaded.predict_node_proba(x, csr, clamped_mask=clamped_mask),
        mpnn.predict_node_proba(x, csr, clamped_mask=clamped_mask),
    )
