"""Tests for serogen.loci and serogen.genes classification, segmentation, and graph operations."""

from pathlib import Path

import numpy as np
import pytest

from serogen.core.interval import Intervals
from serogen.core.seq.seq import Sequences
from serogen.genes import GeneRole, GeneRoleClassifier, Genes
from serogen.loci import (
    ArchetypeCategory,
    ArchetypeClassifier,
    ArchetypeTensors,
    GeneCluster,
    GeneClusters,
    GeneNode,
    Loci,
    Locus,
    LocusArchetype,
    LocusDatabase,
    LocusFinder,
    LocusMatcher,
    LocusPredictor,
    LocusSegmenter,
    Match,
    Matches,
    MatchType,
    SerotypeDatabase,
    SerotypeRecord,
    get_archetype_tensors,
    mask_to_roles,
    roles_to_mask,
)


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


def test_gene_role_classifier_cluster_pooling_transform_and_int64_labels(tmp_path: Path) -> None:
    """Verify int64/32-bit labels, sample_weight, transform, init_from, and cluster probability pooling."""
    seqs = Sequences.from_bytes(
        [
            b"MKKLLVLLPLLAASAAAWZAWZAWZAWZAWZAWZA",
            b"MKKLLVLLPLLAASAAAWZAWZAWZAWZAWZAWZAKK",
            b"MRVLVTGGAGFIGSHVLRLLEAGHEVVVLDNLCNSKR",
            b"MRVLVTGGAGFIGSHVLRLLEAGHEVVVLDNLCNSKRVV",
        ]
    )
    # 32-bit TAG_EDGE style labels (> int8 max and > signed int32 max)
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
    base_clf.fit(seqs, y_large, sample_weight=sample_w)
    assert base_clf.classes_.dtype == np.int64
    assert list(base_clf.classes_) == [tag_a, tag_b]
    assert base_clf.score(seqs, y_large) == 1.0

    # Feature extraction transform(X) matches dense layer input
    feats = base_clf.transform(seqs)
    assert feats.shape == (4, base_clf.feature_dim)
    manual_logits = feats @ base_clf.dense_w_ + base_clf.dense_b_
    assert np.allclose(manual_logits, base_clf.decision_function(seqs), atol=1e-5)

    # Warm-start with frozen backbone (backbone_lr_scale=0.0)
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

    # Cluster-level pooling (log_proba, mean, max) and pooled sequence broadcast
    cids = np.array([101, 101, 202, 0], dtype=np.int64)
    for pool_mode in ("log_proba", "mean", "max"):
        uniq_c, c_probs = base_clf.predict_cluster_proba(seqs, cids, pool=pool_mode, sample_weight=sample_w)
        assert list(uniq_c) == [101, 202]
        assert c_probs.shape == (2, 2)
        assert np.allclose(c_probs.sum(axis=1), 1.0, atol=1e-5)

    uniq_c, c_preds, c_conf = base_clf.predict_clusters(seqs, cids, min_confidence=0.5)
    assert list(uniq_c) == [101, 202]
    assert list(c_preds) == [tag_a, tag_b]
    assert np.all(c_conf >= 0.5)

    pooled_probs = base_clf.predict_proba_pooled(seqs, cids)
    assert pooled_probs.shape == (4, 2)
    assert np.allclose(pooled_probs[0], pooled_probs[1])

    # Save and load preserves int64 classes
    p = tmp_path / "large_classes.npz"
    base_clf.save(p)
    reloaded = GeneRoleClassifier.load(p)
    assert reloaded.classes_.dtype == np.int64
    assert list(reloaded.predict(seqs)) == list(y_large)


def test_locus_and_loci_soa_and_graph_operations() -> None:
    """Verify unified Locus and Loci (and GeneCluster/GeneClusters aliases) SoA and GraphBatch methods."""
    assert GeneCluster is Locus
    assert GeneClusters is Loci
    assert LocusFinder is LocusPredictor

    loci = Loci.from_genes_list(
        [
            ["wza", "wzb", "wzc", "wzx", "wzy", "wbaP"],
            ["wzm", "wzt", "rmlA", "rmlB"],
            ["wzx", "wzy", "wzz", "wbaP", "tnpA", "rmlC"],
        ],
        strands_list=[
            [1, 1, 1, 1, 1, 1],
            [-1, -1, -1, -1],
            [1, 1, 1, 1, 1, 1],
        ],
    )
    assert len(loci) == 3
    assert len(loci.genes) == 16

    loc0 = loci[0]
    assert isinstance(loc0, Locus)
    assert loc0.num_genes == 6
    assert len(loc0.genes) == 6

    # Mobile element healing on locus 2 (which contains tnpA)
    loc2 = loci[2]
    assert loc2.has_mobile_elements is True
    healed = loc2.heal_mobile_elements(remove_nodes=True)
    assert healed.has_mobile_elements is False
    assert healed.num_genes == 5
    assert len(healed.genes) == 5

    # Slicing and concatenation
    sub = loci[[2, 0]]
    assert len(sub) == 2
    assert sub[0] == loc2
    assert sub[1] == loc0
    assert len(sub.genes) == 12

    concat_loci = Loci.concat([loci[:1], loci[1:]])
    assert len(concat_loci) == 3
    assert concat_loci[1] == loci[1]

    # GraphBatch sparse methods
    pan_adj, pan_nodes, _ = loci.as_pan_graph()
    assert pan_adj.shape == (len(pan_nodes), len(pan_nodes))


def test_locus_segmenter_tandem_and_gap_bridging() -> None:
    """Verify LocusSegmenter disambiguates tandem K+O loci, bridges internal gaps, and respects contigs."""
    roles_contig0 = [
        GeneRole.UNKNOWN,
        GeneRole.UNKNOWN,
        GeneRole.UNKNOWN,
        # CPS_WZY locus (with 1 internal UNKNOWN gene at index 6)
        GeneRole.TRANSLOCON,
        GeneRole.KINASE,
        GeneRole.FLIPPASE,
        GeneRole.UNKNOWN,
        GeneRole.POLYMERASE,
        GeneRole.INITIATOR,
        GeneRole.GLYCOSYLTRANSFERASE,
        GeneRole.PYRUVYLTRANSFERASE,
        # Tandem LPS_ABC locus separated by 1,200 bp
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

    # CPS_WZY locus bridged the internal UNKNOWN gene and embeds its constituent Genes directly
    assert loci[0].num_genes == 8
    assert len(loci[0].genes) == 8
    assert loci[0].archetype == LocusArchetype.CPS_WZY
    assert loci[1].archetype == LocusArchetype.LPS_ABC
    assert loci[2].archetype == LocusArchetype.LPS_WZY
    assert loci[2].strand == -1
    assert len(loci[:1].genes) == 8


def test_sgh10_tandem_k1_o1() -> None:
    """Verify exact K1 Capsule + O1 O-Antigen tandem segmentation on SGH10."""
    from pathlib import Path

    from serogen.core.genome import GenomeAssembly

    sgh10_path = Path(__file__).parent / "data" / "SGH10_chromosome.fna.zst"
    if sgh10_path.exists():
        asm = GenomeAssembly.from_file(sgh10_path)
        predictor = LocusPredictor(whole_genome=True)
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
    """Verify LocusPredictor (LocusFinder) role probability projection and empty/Genes handling."""
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
    # Forward locus (+1 strand): wza -> wzb -> wzc -> wbaP
    loc_fwd = Locus.from_genes(
        ["wza", "wzb", "wzc", "wbaP"],
        strands=[1, 1, 1, 1],
    )
    # Reverse-complemented locus (-1 strand) with an internal tnpA insertion:
    # In ascending contig coordinates: wbaP <- tnpA <- wzc <- wzb <- wza
    loc_rev_mge = Locus.from_genes(
        ["wbaP", "tnpA", "wzc", "wzb", "wza"],
        strands=[-1, -1, -1, -1, -1],
    )
    assert loc_rev_mge.strand == -1

    # Commutativity of heal_mobile_elements() and canonicalize_strand()
    c_then_h = loc_rev_mge.canonicalize_strand().heal_mobile_elements()
    h_then_c = loc_rev_mge.heal_mobile_elements().canonicalize_strand()
    assert c_then_h == h_then_c
    assert c_then_h.strand == 1
    assert list(c_then_h.roles) == list(loc_fwd.roles)
    # Idempotency of canonicalize_strand()
    assert c_then_h.canonicalize_strand() == c_then_h

    # Attach synthetic translated proteins so Clusterer groups orthologous genes
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
    # 4 unique pan-nodes (wza, wzb, wzc, wbaP), each appearing twice (once in loc_a, once in loc_b)
    assert len(pan_nodes) == 4
    assert list(node_counts) == [2, 2, 2, 2]
    # 3 directed transitions (wza->wzb, wzb->wzc, wzc->wbaP), each observed twice
    assert int(adj.sum()) == 6
    assert list(adj.data) == [2, 2, 2]
    for pn in pan_nodes:
        assert GeneNode(int(pn)).is_clustered is True


def test_locus_matcher_variants_and_novelty() -> None:
    """Verify MatchType bitflags, Matches SoA methods, and LocusMatcher novelty detection."""
    # Reference database with 2 distinct loci (KL1-like and KL2-like)
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

    # 1. Exact match
    m_exact = ref_kl1.match(ref_kl1)
    assert isinstance(m_exact, Match)
    assert m_exact.match_type == MatchType.EXACT
    assert m_exact.is_exact is True
    assert m_exact.is_novel is False
    assert np.isclose(m_exact.score, 1.0)
    assert m_exact.shared_genes == 6
    assert m_exact.missing_genes == 0
    assert m_exact.extra_genes == 0

    # 2. Strand variant (-1 strand copy of ref_kl1)
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
    assert matcher.is_novel(q_strand) is False

    # 3. Mobile Genetic Element (MGE) variant (tnpA inserted between wbaP and wzx)
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
    assert matcher.is_novel(q_mge) is False

    # 4. Frameshift variant (wzc has a frameshift)
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

    # 5. Combined STRAND_VARIANT | MGE_VARIANT | FRAMESHIFT_VARIANT
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
    assert matcher.is_novel(q_combo) is False

    # 6. Synteny variant (wzx and wzy swapped) -> novel!
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
    assert matcher.is_novel(q_synteny) is True

    # 7. Gene variant (missing wbaP and carrying a novel GT cluster 50) -> novel!
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
    assert matcher.is_novel(q_gene_var) is True

    # 8. Batch query matching, best_per_query(), slicing, and concat
    q_batch = Loci.from_list([ref_kl1, q_combo, q_gene_var])
    all_matches = db.match(q_batch)
    assert isinstance(all_matches, Matches)
    assert len(all_matches) == 6
    top_per_q = all_matches.best_per_query()
    assert len(top_per_q) == 3
    assert list(top_per_q.target_indices) == [0, 0, 0]
    novel_flags = matcher.is_novel(q_batch)
    assert isinstance(novel_flags, np.ndarray)
    assert list(novel_flags) == [False, False, True]

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

    # Query 1: unclustered reverse-complemented locus with identical protein sequences
    rev_base = Locus.from_genes(["wbaP", "wzc", "wzb", "wza"], strands=[-1, -1, -1, -1])
    q_rev_genes = Genes(intervals=rev_base.genes.intervals, translations=Sequences.from_bytes(prot_seqs[::-1]))
    q_rev_loc = Locus.from_genes(q_rev_genes, roles=rev_base.roles, archetype=LocusArchetype.CPS_WZY)

    m1 = matcher.match(q_rev_loc).best()
    assert m1 is not None
    assert m1.match_type == MatchType.STRAND_VARIANT
    assert m1.shared_genes == 4
    assert matcher.is_novel(q_rev_loc) is False

    # Query 2: unclustered locus where the 4th protein is completely novel
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
    assert matcher.is_novel(q_nov_loc) is True

    # Detailed node-by-node GraphAligner alignment on Query 1
    alignment = matcher.align(q_rev_loc, 0)
    assert len(alignment.unmatched_a) == 0
    assert len(alignment.unmatched_b) == 0


def test_locus_segmenter_contig_split_stitching() -> None:
    """Verify LocusSegmenter rescues and stitches loci split across contigs (including strand inversions)."""
    import pytest

    # Intact reference CPS_WZY locus: wza -> wzb -> wzc -> wbaP -> wzx -> wzy -> ugd
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

    # Split across Contig 0 (3' boundary: wza, wzb, wzc -> 2/5 = 0.40 CPS_WZY completeness alone)
    # and Contig 1 (reverse-complemented -1 strand at 3' end: ugd <- wzy <- wzx <- wbaP)
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

    # Without contig-split stitching, Contig 0's capsule export module (wza/wzb/wzc) is lost
    # and Contig 1's polymerization half is misclassified as a truncated 4-gene LPS_WZY
    seg_no_stitch = LocusSegmenter(min_completeness=0.80, stitch_contig_splits=False)
    unstitched = seg_no_stitch.segment(genes, all_roles, clusters=all_cids)
    assert len(unstitched) == 1
    assert unstitched[0].archetype == LocusArchetype.LPS_WZY
    assert len(unstitched[0].nodes) == 4

    # With contig-split stitching enabled (default), both halves are stitched into the 7-gene CPS_WZY
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
    assert matcher.is_novel(loc) is False


def test_sgh10_simulated_fragmented_assemblies() -> None:
    """Simulate fragmented assemblies of SGH10 with cuts and strand inversions across K1 and O1."""
    from pathlib import Path

    import pytest

    from serogen.core.genome import GenomeAssembly

    sgh10_path = Path(__file__).parent / "data" / "SGH10_chromosome.fna.zst"
    if not sgh10_path.exists():
        pytest.skip("SGH10_chromosome.fna.zst not found")

    rc_tab = bytes.maketrans(b"ACGTacgtNn", b"TGCAtgcaNn")

    def _revcomp(b: bytes) -> bytes:
        return b.translate(rc_tab)[::-1]

    asm = GenomeAssembly.from_file(sgh10_path)
    # 100 kb window spanning SGH10 K1 (relative 26,476..51,461) and O1 (relative 54,456..62,380)
    region = asm[0][1_680_000:1_780_000]

    predictor = LocusPredictor(whole_genome=True)
    ref_loci = predictor.predict(region)
    assert len(ref_loci) == 2
    assert list(ref_loci.archetypes) == [int(LocusArchetype.CPS_WZY), int(LocusArchetype.LPS_ABC)]
    assert list(ref_loci.is_fragmented) == [False, False]

    cps_only = predictor.predict(region, categories=ArchetypeCategory.CAPSULE, min_completeness=0.95, only_intact=True)
    assert len(cps_only) == 1
    assert cps_only[0].archetype == LocusArchetype.CPS_WZY

    matcher = LocusMatcher(ref_loci)

    # 1. Test cuts across early K1 (30 kb), mid K1 (45 kb), late K1 (50 kb), and mid O1 (56 kb)
    #    with both forward and reverse-complemented contig orientations
    for cut_bp, expected_frag_target in [(30_000, 0), (45_000, 0), (50_000, 0), (56_000, 1)]:
        for rc_second in (False, True):
            c1 = region[:cut_bp]
            c2 = _revcomp(region[cut_bp:]) if rc_second else region[cut_bp:]
            frag_seqs = Sequences.from_bytes([c1, c2])
            pred = predictor.predict(frag_seqs)

            assert len(pred) == 2
            top_matches = matcher.match(pred, top_k=1)
            assert len(top_matches) == 2
            assert set(int(t) for t in top_matches.target_indices) == {0, 1}

            for m in top_matches:
                q_loc = pred[m.query_index]
                if m.target_index == expected_frag_target:
                    assert q_loc.is_fragmented is True
                    assert q_loc.num_contigs == 2
                    assert bool(m.match_type & MatchType.FRAGMENTED_VARIANT)
                    assert m.jaccard == pytest.approx(1.0)
                    assert m.score >= 0.99
                    assert m.is_novel is False
                else:
                    assert q_loc.is_fragmented is False
                    assert q_loc.num_contigs == 1
                    assert m.jaccard == pytest.approx(1.0)
                    assert m.is_novel is False

    # 2. Test 3-way split of K1 across 3 contigs (cuts at 32 kb and 45 kb) with middle contig inverted
    frags_3way = Sequences.from_bytes(
        [
            region[:32_000],
            _revcomp(region[32_000:45_000]),
            region[45_000:],
        ]
    )
    pred_3way = predictor.predict(frags_3way)
    assert len(pred_3way) == 2
    matches_3way = matcher.match(pred_3way, top_k=1)
    k1_match = [m for m in matches_3way if m.target_index == 0][0]
    k1_stitched = pred_3way[k1_match.query_index]
    assert k1_stitched.archetype == LocusArchetype.CPS_WZY
    assert k1_stitched.is_fragmented is True
    assert k1_stitched.num_contigs == 3
    assert bool(k1_match.match_type & MatchType.FRAGMENTED_VARIANT)
    assert k1_match.jaccard >= 0.90

    # 3. Test simultaneous fragmentation of BOTH K1 (at 40 kb) and O1 (at 58 kb) across 3 shuffled contigs
    frags_both = Sequences.from_bytes(
        [
            region[58_000:],
            region[:40_000],
            _revcomp(region[40_000:58_000]),
        ]
    )
    pred_both = predictor.predict(frags_both)
    assert len(pred_both) == 2
    assert all(loc.is_fragmented for loc in pred_both)
    matches_both = matcher.match(pred_both, top_k=1)
    assert set(int(t) for t in matches_both.target_indices) == {0, 1}
    assert all(m.jaccard >= 0.85 for m in matches_both)


def test_loci_filter_allelic_identity_and_novel_locus_tier() -> None:
    """Verify Loci.filter(categories=...), allelic AA identity tie-breaking, and MatchType.NOVEL_LOCUS."""
    import pytest

    # 1. Category and archetype filtering + Loci.best()
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

    # 2. Allelic amino-acid identity tie-breaking on identical-cluster serotypes (e.g. 6C vs 6D)
    prot_6d = [
        b"MKKLLVLLPLLAASAAAWZAWZAWZAWZAWZAWZA" * 3,
        b"MKLNIIVCTGGNRSIGKSVLALLQERGYQVRSMP" * 3,
        b"MNTQVLSVVLAALLLLVAGAWZCWZCWZCWZCWZC" * 3,
        b"MKRIALLFLLPLLVLAALWBAPWBAPWBAPWBAP" * 3,
    ]
    # 6C differs from 6D by ~6% AA substitutions in the 4th gene (still >= 80% cluster threshold)
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
    # Put 6D at index 0 and 6C at index 1 so index tie-breaking alone would wrongly pick 6D (index 0) for a 6C query!
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

    # 3. Two-tier novelty bitflags: GENE_VARIANT (jaccard >= 0.50) vs NOVEL_LOCUS (jaccard < 0.50)
    q_intra_serogroup = Locus.from_genes(
        ["wza", "wzb", "wzc", "wbaP"],
        strands=[1, 1, 1, 1],
        clusters=[1, 2, 3, 99],
        archetype=LocusArchetype.CPS_WZY,
    )
    q_distant_serogroup = Locus.from_genes(
        ["wza", "wzb", "wzc", "wbaP"],
        strands=[1, 1, 1, 1],
        clusters=[1, 97, 98, 99],
        archetype=LocusArchetype.CPS_WZY,
    )
    ref_4 = Locus.from_genes(
        ["wza", "wzb", "wzc", "wbaP"],
        strands=[1, 1, 1, 1],
        clusters=[1, 2, 3, 4],
        archetype=LocusArchetype.CPS_WZY,
    )
    m_intra = ref_4.match(q_intra_serogroup)
    assert m_intra.jaccard == pytest.approx(3 / 5)
    assert bool(m_intra.match_type & MatchType.GENE_VARIANT)
    assert m_intra.is_novel is True
    assert m_intra.is_novel_locus is False

    m_distant = ref_4.match(q_distant_serogroup)
    assert m_distant.jaccard == pytest.approx(1 / 7)
    assert bool(m_distant.match_type & MatchType.GENE_VARIANT)
    assert bool(m_distant.match_type & MatchType.NOVEL_LOCUS)
    assert m_distant.is_novel is True
    assert m_distant.is_novel_locus is True


def test_serotype_database_and_persistence(tmp_path: Path) -> None:
    """Verify Loci.save/load, LocusMatcher.save/load, and SerotypeDatabase.build/save/load/without."""
    import pytest

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
    prot_19a = [
        prot_6d[0],
        b"MRKILVTGAGGSIGSELCRQILKFQPKKLVLLDI" * 3,
        b"MNNKVIILGAAGGVGQSLARLLKNRGHEVIGIDN" * 3,
        b"MSLLRKKIIVLLAVLLVFSFALFLPLLGLLFSLF" * 3,
    ]
    base_4 = Locus.from_genes(["wza", "wzb", "wzc", "wbaP"], strands=[1, 1, 1, 1])

    loc_6d = Locus.from_genes(
        Genes(intervals=base_4.genes.intervals, translations=Sequences.from_bytes(prot_6d)),
        roles=base_4.roles,
        archetype=LocusArchetype.CPS_WZY,
        name="6D",
    )
    loc_6c = Locus.from_genes(
        Genes(intervals=base_4.genes.intervals, translations=Sequences.from_bytes(prot_6c)),
        roles=base_4.roles,
        archetype=LocusArchetype.CPS_WZY,
        name="6C",
    )
    loc_19a = Locus.from_genes(
        Genes(intervals=base_4.genes.intervals, translations=Sequences.from_bytes(prot_19a)),
        roles=base_4.roles,
        archetype=LocusArchetype.CPS_WZY,
        name="19A",
    )

    # 1. Loci naming, string indexing, .without(), and .save() / .load()
    loci = Loci.from_list([loc_6d, loc_6c, loc_19a]).cluster_genes()
    assert list(loci.names) == ["6D", "6C", "19A"]
    assert loci["6C"].name == "6C"
    assert loci["6C"] == loc_6c.with_name("6C") or loci["6C"].name == "6C"

    loci_no_6c = loci.without("6C")
    assert list(loci_no_6c.names) == ["6D", "19A"]

    loci_path = tmp_path / "loci.npz"
    loci.save(loci_path)
    loci_loaded = Loci.load(loci_path)
    assert len(loci_loaded) == 3
    assert list(loci_loaded.names) == ["6D", "6C", "19A"]
    assert loci_loaded["19A"] == loci["19A"]
    np.testing.assert_array_equal(loci_loaded[0].genes.translations.seqs, loci[0].genes.translations.seqs)

    # 2. LocusMatcher.save() / .load() zero-compute persistence and target_name
    matcher = LocusMatcher(loci)
    matcher_path = tmp_path / "matcher.npz"
    matcher.save(matcher_path)
    matcher_loaded = LocusMatcher.load(matcher_path)

    m_orig = matcher.match(loc_6c)
    m_load = matcher_loaded.match(loc_6c)
    assert m_load[0].target_name == "6C"
    assert m_load[0].target_index == 1
    assert m_load[0].score == pytest.approx(m_orig[0].score)
    assert m_load[0].mean_aa_identity == pytest.approx(m_orig[0].mean_aa_identity)
    assert list(m_load.target_names) == list(m_orig.target_names)

    # 3. SerotypeDatabase (and alias LocusDatabase) build, save, load, indexing, and LOO .without()
    assert LocusDatabase is SerotypeDatabase
    db = SerotypeDatabase.build(
        [loc_6d, loc_6c, loc_19a],
        serogroups=["6", "6", "19"],
        accessions=["HM171374", "EF538714", "CR931673"],
        strains=["BM420", "CH2099", "1777/39"],
        cluster_identity=0.80,
    )
    assert len(db) == 3
    assert list(db.serotypes) == ["6D", "6C", "19A"]
    assert "6C" in db

    rec_6c = db["6C"]
    assert isinstance(rec_6c, SerotypeRecord)
    assert rec_6c.name == "6C"
    assert rec_6c.serogroup == "6"
    assert rec_6c.accession == "HM171374" if False else rec_6c.accession == "EF538714"
    assert rec_6c.strain == "CH2099"

    db_path = tmp_path / "spneumo_db.npz"
    db.save(db_path)
    db_loaded = SerotypeDatabase.load(db_path)
    assert len(db_loaded) == 3
    assert list(db_loaded.names) == ["6D", "6C", "19A"]
    assert list(db_loaded.serogroups) == ["6", "6", "19"]
    assert list(db_loaded.accessions) == ["HM171374", "EF538714", "CR931673"]
    assert len(db_loaded.pan_graph) == 3
    np.testing.assert_array_equal(db_loaded.pan_nodes, db.pan_nodes)
    np.testing.assert_array_equal(db_loaded.pan_counts, db.pan_counts)

    # Match against loaded DB
    db_matches = db_loaded.match(loc_6c)
    assert db_matches[0].target_name == "6C"
    assert db_matches[0].mean_aa_identity == pytest.approx(1.0)
    assert db_matches[1].target_name == "6D"
    assert db_matches[1].mean_aa_identity < 1.0

    # Leave-one-out via db.without("6C")
    loo_db = db_loaded.without("6C")
    assert len(loo_db) == 2
    assert list(loo_db.names) == ["6D", "19A"]
    loo_matches = loo_db.match(loc_6c)
    assert loo_matches[0].target_name == "6D"
    assert loo_matches[0].jaccard == pytest.approx(1.0)

    # Leave-one-out via db.without("19A") -> distant serogroup (NOVEL_LOCUS)
    loo_19a = db_loaded.without("19A")
    m_19a = loo_19a.match(loc_19a)[0]
    assert m_19a.is_novel is True
    assert m_19a.is_novel_locus is True


def test_wzg_capsule_anchor_operon_snapping_teichoic_acid_and_predictor_filters() -> None:
    """Verify Wzg/CpsA CAPSULE_ANCHOR classification, outward operon boundary snapping, TEICHOIC_ACID, and predictor filters."""
    import pytest

    # 1. Gram-positive LytR-CpsA-Psr (Wzg / CpsA / CapA) in GeneRole.CAPSULE_ANCHOR and pretrained classifier
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
    assert float(wzg_probs[0, anchor_col]) > 0.90

    # 2. Outward operon boundary-snapping pass in LocusSegmenter (< 150 bp gap, same strand, 0.35 <= P(active) < 0.50)
    # Layout of 10 genes:
    #   idx 0: UNKNOWN (distant flanking housekeeping)
    #   idx 1: CAPSULE_ANCHOR with P(active)=0.42, gap=400 bp (> 150 bp, should NOT snap)
    #   idx 2: CAPSULE_ANCHOR (wzg) with P(active)=0.42, gap=45 bp to idx 3, strand=+1 (SHOULD snap!)
    #   idx 3..7: Core CPS_WZY operon (wzh, wzd, wze, wchA, wzx, wzy -> Wait: 6 genes idx 3..8) with P(active)=0.95
    #   idx 9: OXIDOREDUCTASE (rmlD) with P(active)=0.40, gap=30 bp from idx 8, strand=+1 (SHOULD snap!)
    #   idx 10: GLYCOSYLTRANSFERASE with P(active)=0.44, gap=25 bp from idx 9, but strand=-1 (opposite strand, should NOT snap!)
    n_roles = len(GeneRole)
    probs = np.zeros((11, n_roles), dtype=np.float32)
    probs[0, GeneRole.UNKNOWN] = 0.98
    probs[0, GeneRole.OXIDOREDUCTASE] = 0.02

    # idx 1: 400 bp upstream of idx 2 -> too far to snap
    probs[1, GeneRole.UNKNOWN] = 0.58
    probs[1, GeneRole.CAPSULE_ANCHOR] = 0.42

    # idx 2: wzg 45 bp upstream of idx 3 on +1 strand with P(active) = 0.42 -> snaps into 5' boundary!
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

    # idx 9: rmlD 30 bp downstream of idx 8 on +1 strand with P(active) = 0.40 -> snaps into 3' boundary!
    probs[9, GeneRole.UNKNOWN] = 0.60
    probs[9, GeneRole.OXIDOREDUCTASE] = 0.40

    # idx 10: opposite strand (-1) despite 25 bp gap -> must NOT snap!
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
    # Snapped locus spans indices 2..9 inclusive (8 genes: wzg + 6 core + rmlD)
    assert snapped_loc.num_genes == 8
    assert snapped_loc.start == 3400
    assert snapped_loc.end == 11730

    seg_no_snap = LocusSegmenter(max_operon_snap_bp=0)
    unsnapped_loci = seg_no_snap.segment(genes, probs)
    assert len(unsnapped_loci) == 1
    assert unsnapped_loci[0].num_genes == 6
    assert unsnapped_loci[0].start == 4445
    assert unsnapped_loci[0].end == 10700

    # 3. TEICHOIC_ACID archetype (category == ArchetypeCategory.OTHER) vs LPS_ABC
    assert LocusArchetype.TEICHOIC_ACID.category == ArchetypeCategory.OTHER
    assert LocusArchetype.from_name("teichoic_acid") == LocusArchetype.TEICHOIC_ACID
    assert LocusArchetype.from_name("wta") == LocusArchetype.TEICHOIC_ACID

    wta_roles = [
        GeneRole.ABC_PERMEASE,  # tarG / tagG
        GeneRole.ABC_ATPASE,  # tarH / tagH
        GeneRole.PHOSPHOTRANSFERASE,  # licD (phosphocholine / polyol-phosphate transferase)
        GeneRole.ALANINE_TRANSFERASE,  # dltA
        GeneRole.NUCLEOTIDYLTRANSFERASE,  # tarI
        GeneRole.OXIDOREDUCTASE,  # tarJ
        GeneRole.GLYCOSYLTRANSFERASE,  # tarL / tagE
    ]
    arch_clf = ArchetypeClassifier()
    wta_arch, wta_comp = arch_clf.classify_mask(roles_to_mask(wta_roles))
    assert wta_arch == LocusArchetype.TEICHOIC_ACID
    assert wta_comp == pytest.approx(1.0)

    wta_starts = np.arange(len(wta_roles), dtype=np.int32) * 1100
    wta_ends = wta_starts + 1000
    wta_genes = Genes(
        intervals=Intervals(starts=wta_starts, ends=wta_ends, strands=np.ones(len(wta_roles), dtype=np.int8))
    )
    wta_seg = seg_snap.segment(wta_genes, np.array(wta_roles, dtype=np.int32))
    assert len(wta_seg) == 1
    assert wta_seg[0].archetype == LocusArchetype.TEICHOIC_ACID
    assert wta_seg[0].category == ArchetypeCategory.OTHER

    # 4. LocusPredictor.predict() and Loci.filter() with categories and min_completeness
    combined_loci = Loci.concat([snapped_loci, wta_seg])
    assert len(combined_loci) == 2
    caps_only = combined_loci.filter(categories={ArchetypeCategory.CAPSULE}, min_completeness=0.90)
    assert len(caps_only) == 1
    assert caps_only[0].archetype == LocusArchetype.CPS_WZY

    other_only = combined_loci.filter(categories=ArchetypeCategory.OTHER)
    assert len(other_only) == 1
    assert other_only[0].archetype == LocusArchetype.TEICHOIC_ACID


def test_ectopic_synthase_partial_coverage_module_gating_and_novelty_tiers(tmp_path: Path) -> None:
    """Verify 1-2 gene CPS_SYNTHASE, intactness filtering, partial-gene unidirectional coverage, module gating, and novelty tiers."""
    import pytest

    n_roles = len(GeneRole)

    # 1. Ectopic 1-gene (Serotype 37 tts) and 2-gene (Serotype 3 ugd + cps3S) CPS_SYNTHASE segmentation
    # Contig 0: housekeeping + single-gene tts (TRANSMEMBRANE_SYNTHASE, P=0.94) + housekeeping
    probs_1gene = np.zeros((3, n_roles), dtype=np.float32)
    probs_1gene[0, GeneRole.UNKNOWN] = 0.99
    probs_1gene[1, GeneRole.UNKNOWN] = 0.06
    probs_1gene[1, GeneRole.TRANSMEMBRANE_SYNTHASE] = 0.94
    probs_1gene[2, GeneRole.UNKNOWN] = 0.99
    starts_1gene = np.array([100, 1500, 3200], dtype=np.int32)
    ends_1gene = np.array([1100, 2900, 4200], dtype=np.int32)
    genes_1gene = Genes(
        intervals=Intervals(starts=starts_1gene, ends=ends_1gene, strands=np.array([1, 1, 1], dtype=np.int8))
    )

    arch_clf = ArchetypeClassifier()
    assert arch_clf.classify_mask(roles_to_mask([GeneRole.TRANSMEMBRANE_SYNTHASE])) == (
        LocusArchetype.CPS_SYNTHASE,
        pytest.approx(1.0),
    )

    seg = LocusSegmenter()
    loci_1gene = seg.segment(genes_1gene, probs_1gene)
    assert len(loci_1gene) == 1
    tts_loc = loci_1gene[0]
    assert tts_loc.archetype == LocusArchetype.CPS_SYNTHASE
    assert tts_loc.category == ArchetypeCategory.CAPSULE
    assert tts_loc.completeness > 0.98
    assert tts_loc.num_genes == 1
    assert tts_loc.is_intact is True
    assert tts_loc.num_frameshifts == 0
    assert tts_loc.frameshift_fraction == pytest.approx(0.0)

    # 2-gene synthase (Serotype 3 ugd + cps3S: OXIDOREDUCTASE + TRANSMEMBRANE_SYNTHASE)
    probs_2gene = np.zeros((4, n_roles), dtype=np.float32)
    probs_2gene[0, GeneRole.UNKNOWN] = 0.99
    probs_2gene[1, GeneRole.UNKNOWN] = 0.05
    probs_2gene[1, GeneRole.OXIDOREDUCTASE] = 0.95
    probs_2gene[2, GeneRole.UNKNOWN] = 0.04
    probs_2gene[2, GeneRole.TRANSMEMBRANE_SYNTHASE] = 0.96
    probs_2gene[3, GeneRole.UNKNOWN] = 0.99
    starts_2gene = np.array([100, 1500, 2600, 4200], dtype=np.int32)
    ends_2gene = np.array([1100, 2550, 3900, 5200], dtype=np.int32)
    genes_2gene = Genes(
        intervals=Intervals(starts=starts_2gene, ends=ends_2gene, strands=np.array([1, 1, 1, 1], dtype=np.int8))
    )
    loci_2gene = seg.segment(genes_2gene, probs_2gene)
    assert len(loci_2gene) == 1
    assert loci_2gene[0].archetype == LocusArchetype.CPS_SYNTHASE
    assert loci_2gene[0].completeness > 0.98
    assert loci_2gene[0].num_genes == 2

    # Serotype 37 scenario: Cryptic/defective canonical CPS_WZY (with frameshifted genes) + intact ectopic tts
    cryptic_genes = Genes(
        intervals=Intervals(
            starts=np.arange(6, dtype=np.int32) * 1100,
            ends=np.arange(6, dtype=np.int32) * 1100 + 1000,
            strands=np.ones(6, dtype=np.int8),
        ),
    )
    cryptic_wzy = Locus.from_genes(
        cryptic_genes,
        roles=np.array(
            [
                GeneRole.CAPSULE_ANCHOR,
                GeneRole.PHOSPHATASE,
                GeneRole.COPOLYMERASE,
                GeneRole.INITIATOR,
                GeneRole.FLIPPASE,
                GeneRole.POLYMERASE,
            ],
            dtype=np.int32,
        ),
        frameshifts=np.array([False, True, False, True, False, False], dtype=np.bool_),
        archetype=LocusArchetype.CPS_WZY,
        completeness=1.0,
        score=38.0,
        name="cryptic_33F",
    )
    intact_tts = tts_loc.with_name("ectopic_37_tts")
    serotype37_loci = Loci.from_list([cryptic_wzy, intact_tts])

    assert cryptic_wzy.num_frameshifts == 2
    assert cryptic_wzy.frameshift_fraction == pytest.approx(2 / 6)
    assert cryptic_wzy.is_intact is False
    np.testing.assert_array_equal(serotype37_loci.num_frameshifts, np.array([2, 0], dtype=np.int32))
    np.testing.assert_array_equal(serotype37_loci.is_intact, np.array([False, True], dtype=np.bool_))

    # Standard .best() picks cryptic_wzy by score (38.0 > ~2.8), whereas .best(prefer_intact=True) picks intact_tts!
    best_by_score = serotype37_loci.best(prefer_intact=False)
    best_intact = serotype37_loci.best(prefer_intact=True)
    assert best_by_score is not None and best_by_score.name == "cryptic_33F"
    assert best_intact is not None and best_intact.name == "ectopic_37_tts"
    intact_filtered = serotype37_loci.filter(max_frameshift_fraction=0.10)
    assert len(intact_filtered) == 1
    assert intact_filtered[0].name == "ectopic_37_tts"

    # 2. Complementary module gating in LocusSegmenter._stitch_boundary_islands
    # Contig 0 has a 100%-complete CPS_WZY locus near its right edge; Contig 1 has a standalone TEICHOIC_ACID locus near its left edge.
    # Because CPS_WZY is already 100% complete and TEICHOIC_ACID does not increase CPS_WZY completeness, they must NOT be stitched!
    cps_roles = [
        GeneRole.CAPSULE_ANCHOR,
        GeneRole.PHOSPHATASE,
        GeneRole.COPOLYMERASE,
        GeneRole.KINASE,
        GeneRole.INITIATOR,
        GeneRole.FLIPPASE,
        GeneRole.POLYMERASE,
    ]
    wta_roles = [
        GeneRole.ABC_PERMEASE,
        GeneRole.ABC_ATPASE,
        GeneRole.PHOSPHOTRANSFERASE,
        GeneRole.ALANINE_TRANSFERASE,
        GeneRole.NUCLEOTIDYLTRANSFERASE,
    ]
    # Contig 0 has 2 upstream UNKNOWN genes followed by a 100%-complete CPS_WZY locus at its 3' edge;
    # Contig 1 has a standalone TEICHOIC_ACID locus at its 5' edge followed by 2 downstream UNKNOWN genes.
    all_roles_2contigs = np.array(
        [GeneRole.UNKNOWN, GeneRole.UNKNOWN] + cps_roles + wta_roles + [GeneRole.UNKNOWN, GeneRole.UNKNOWN],
        dtype=np.int32,
    )
    starts_c0 = np.arange(len(cps_roles) + 2, dtype=np.int32) * 1050 + 5000
    ends_c0 = starts_c0 + 1000
    starts_c1 = np.arange(len(wta_roles) + 2, dtype=np.int32) * 1050 + 100
    ends_c1 = starts_c1 + 1000
    seq_ids_2c = np.array([0] * (len(cps_roles) + 2) + [1] * (len(wta_roles) + 2), dtype=np.uint32)
    genes_2c = Genes(
        intervals=Intervals(
            starts=np.concatenate([starts_c0, starts_c1]),
            ends=np.concatenate([ends_c0, ends_c1]),
            strands=np.ones(len(all_roles_2contigs), dtype=np.int8),
        ),
        contig_indices=seq_ids_2c,
    )
    unstitched = seg.segment(genes_2c, all_roles_2contigs)
    assert len(unstitched) == 2
    assert {loc.archetype for loc in unstitched} == {LocusArchetype.CPS_WZY, LocusArchetype.TEICHOIC_ACID}
    assert all(not loc.is_fragmented for loc in unstitched)

    # Conversely, when CPS_WZY itself is split across Contig 0 (2 UNKNOWN + first 4 CPS genes at 3' edge)
    # and Contig 1 (remaining 3 CPS genes at 5' edge + 2 UNKNOWN),
    # comp_cross == 1.0 > max(comp_l, comp_r), so complementary module gating DOES stitch them!
    split_cps_roles = np.array(
        [GeneRole.UNKNOWN, GeneRole.UNKNOWN] + cps_roles + [GeneRole.UNKNOWN, GeneRole.UNKNOWN],
        dtype=np.int32,
    )
    split_cps_seq_ids = np.array([0] * 6 + [1] * 5, dtype=np.uint32)
    split_starts = np.concatenate(
        [
            np.arange(6, dtype=np.int32) * 1050 + 5000,
            np.arange(5, dtype=np.int32) * 1050 + 100,
        ]
    )
    split_ends = split_starts + 1000
    split_cps_genes = Genes(
        intervals=Intervals(
            starts=split_starts,
            ends=split_ends,
            strands=np.ones(len(split_cps_roles), dtype=np.int8),
        ),
        contig_indices=split_cps_seq_ids,
    )
    stitched_cps = seg.segment(split_cps_genes, split_cps_roles)
    assert len(stitched_cps) == 1
    assert stitched_cps[0].archetype == LocusArchetype.CPS_WZY
    assert stitched_cps[0].is_fragmented is True
    assert stitched_cps[0].completeness > 0.98

    # 3. Unidirectional coverage for contig-edge partial genes + contig-break split-gene collapsing in LocusMatcher
    prot_full = [
        b"MKKLLVLLPLLAASAAAWZAWZAWZAWZAWZAWZA" * 3,  # 105 aa
        b"MKLNIIVCTGGNRSIGKSVLALLQERGYQVRSMP" * 3,  # 102 aa
        b"MNTQVLSVVLAALLLLVAGAWZCWZCWZCWZCWZC" * 3,  # 105 aa
        (
            b"MKRIALLFLLPLLVLAALWBAPWBAPWBAPWBAP"
            b"GGGGSYVLVTGAGGSIGSELCRQILKFQPKKLVLL"
            b"DDDDNVIILGAAGGVGQSLARLLKNRGHEVIGIDN"
            b"TTTTKIIVLLAVLLVFSFALFLPLLGLLFSLFKKK"
        ),  # 139 aa distinct N-term and C-term halves
    ]
    ref_roles = np.array(
        [GeneRole.CAPSULE_ANCHOR, GeneRole.PHOSPHATASE, GeneRole.COPOLYMERASE, GeneRole.INITIATOR],
        dtype=np.int32,
    )
    ref_starts = np.array([100, 1200, 2300, 3400], dtype=np.int32)
    ref_ends = np.array([1100, 2200, 3300, 4500], dtype=np.int32)
    ref_loc = Locus.from_genes(
        Genes(
            intervals=Intervals(starts=ref_starts, ends=ref_ends, strands=np.ones(4, dtype=np.int8)),
            translations=Sequences.from_bytes(prot_full),
        ),
        roles=ref_roles,
        archetype=LocusArchetype.CPS_WZY,
        name="Ref4",
    )
    db_ref = Loci.from_list([ref_loc]).cluster_genes()
    matcher_ref = LocusMatcher(db_ref)

    # Split the 4th gene (139 aa) across a contig break into an 85 aa N-terminal fragment on Contig 0
    # (bidirectional coverage = 85/139 = 0.611 < 0.70, unidirectional coverage = 85/85 = 1.0 >= 0.70!)
    # and a 54 aa C-terminal fragment on Contig 1 with an artificial frameshift flag at the contig break.
    prot_split = [
        prot_full[0],
        prot_full[1],
        prot_full[2],
        prot_full[3][:85],
        prot_full[3][85:],
    ]
    q_starts = np.array([100, 1200, 2300, 3400, 0], dtype=np.int32)
    q_ends = np.array([1100, 2200, 3300, 3655, 162], dtype=np.int32)
    q_seq_ids = np.array([0, 0, 0, 0, 1], dtype=np.uint32)
    q_partial = np.array([False, False, False, True, True], dtype=np.bool_)
    q_frameshift = np.array([False, False, False, False, True], dtype=np.bool_)
    q_roles = np.array(
        [
            GeneRole.CAPSULE_ANCHOR,
            GeneRole.PHOSPHATASE,
            GeneRole.COPOLYMERASE,
            GeneRole.INITIATOR,
            GeneRole.INITIATOR,
        ],
        dtype=np.int32,
    )
    q_split_loc = Locus.from_genes(
        Genes(
            intervals=Intervals(
                starts=q_starts,
                ends=q_ends,
                strands=np.ones(5, dtype=np.int8),
            ),
            contig_indices=q_seq_ids,
            translations=Sequences.from_bytes(prot_split),
            is_partial=q_partial,
        ),
        roles=q_roles,
        frameshifts=q_frameshift,
        archetype=LocusArchetype.CPS_WZY,
    )

    m_split = matcher_ref.match(q_split_loc)[0]
    assert m_split.shared_genes == 4
    assert m_split.missing_genes == 0
    assert m_split.extra_genes == 0
    assert m_split.jaccard == pytest.approx(1.0)
    assert m_split.mean_aa_identity == pytest.approx(1.0)
    assert bool(m_split.match_type & MatchType.FRAGMENTED_VARIANT)
    assert not bool(m_split.match_type & MatchType.GENE_VARIANT)
    assert not bool(m_split.match_type & MatchType.FRAMESHIFT_VARIANT)

    # 4. Coverage-weighted AA identity tie-breaking and configurable novel_jaccard_threshold
    # Target 0 has a truncated 4th gene (85% length of full-length allele, 100% identity over aligned span, non-partial)
    # Target 1 has the full-length 4th gene (100% length, 100% identity)
    prot_trunc_target = [
        prot_full[0],
        prot_full[1],
        prot_full[2],
        prot_full[3][:115],  # 115 / 139 = 82.7% bidirectional coverage
    ]
    loc_trunc = Locus.from_genes(
        Genes(
            intervals=Intervals(starts=ref_starts, ends=ref_ends, strands=np.ones(4, dtype=np.int8)),
            translations=Sequences.from_bytes(prot_trunc_target),
        ),
        roles=ref_roles,
        archetype=LocusArchetype.CPS_WZY,
        name="TruncTarget",
    )
    db_cov = Loci.from_list([loc_trunc, ref_loc]).cluster_genes()
    matcher_cov = LocusMatcher(db_cov, novel_jaccard_threshold=0.65)
    m_cov = matcher_cov.match(ref_loc)
    assert m_cov[0].target_name == "Ref4"
    assert m_cov[0].mean_aa_identity == pytest.approx(1.0)
    assert m_cov[1].target_name == "TruncTarget"
    assert m_cov[1].mean_aa_identity < 1.0

    # Test Match.is_intra_serogroup_variant and Matches.is_intra_serogroup_variant + custom novel_jaccard_threshold persistence
    matcher_cov_path = tmp_path / "matcher_cov.npz"
    matcher_cov.save(matcher_cov_path)
    matcher_cov_loaded = LocusMatcher.load(matcher_cov_path)
    assert matcher_cov_loaded.novel_jaccard_threshold == pytest.approx(0.65)

    # A query with Jaccard = 3/5 = 0.60 is an intra-serogroup variant under default threshold (0.50),
    # but becomes a NOVEL_LOCUS under custom threshold (0.65)!
    q_jaccard_060 = Locus.from_genes(
        ["wza", "wzb", "wzc", "wbaP"],
        strands=[1, 1, 1, 1],
        clusters=[int(c) for c in db_cov[1].cluster_ids[:3]] + [999],
        archetype=LocusArchetype.CPS_WZY,
    )
    m_default_thresh = LocusMatcher(db_cov, novel_jaccard_threshold=0.50).match(q_jaccard_060)
    assert m_default_thresh[0].jaccard == pytest.approx(0.60)
    assert m_default_thresh[0].is_intra_serogroup_variant is True
    assert m_default_thresh[0].is_novel_locus is False
    assert bool(m_default_thresh.is_intra_serogroup_variant[0]) is True
    assert bool(m_default_thresh.is_novel_locus[0]) is False

    m_strict_thresh = matcher_cov_loaded.match(q_jaccard_060)
    assert m_strict_thresh[0].jaccard == pytest.approx(0.60)
    assert m_strict_thresh[0].is_intra_serogroup_variant is False
    assert m_strict_thresh[0].is_novel_locus is True
    assert bool(m_strict_thresh.is_intra_serogroup_variant[0]) is False
    assert bool(m_strict_thresh.is_novel_locus[0]) is True


def test_mcl_bitscore_clustering_and_auto_novel_threshold_inference() -> None:
    """Verify PairwiseAlignment bitscores, MCL functional orthogroup clustering, and data-driven novel_jaccard_threshold inference."""
    import pytest

    from serogen.core.seq.clustering import Clusterer
    from serogen.core.seq.pairwise import PairwiseAligner

    # 1. Divergent functional homologs (two glycosyltransferases with ~45% AA identity + one distinct Wzy polymerase)
    gt_a = (
        b"MKVLLVTGAGGSIGSELCRQILKFQPKKLVLLDINELVKEFKKLGVEIVPIIGDVRDKDL"
        b"VEEVLKKHPDIDVIIHFAGLKAVGESVQKPLDYYRVNLTGTIQLLEIMKAHGVKNLVFSS"
    )
    # gt_b has conservative/semi-conservative substitutions across ~55% of positions (~45% exact AA identity to gt_a)
    gt_b = (
        b"MRILVITGAGGSIGSALCRQVLRYQPERLVLLDVNELIREFQRLGVDIVPVIGDVRDREL"
        b"VEAVLRRYPDVDVIIHFAGLKAVGESVAKPLDYYRVNVTGTVRLLEVMRAHGVRNLVFSS"
    )
    # gt_query is a third divergent GT (~46% identity to gt_a, ~82% identity to gt_b)
    gt_query = (
        b"MRILVITGAGGSIGSALCRQVLRYQPERLVLLDVNELIREFQRLGVDIVPVIGDVRDREL"
        b"VEAVLRRYPDVDVIIHFAGLKAVGESVAKPLDYYRVNVTGTVRLLEVMRAHGVRNLAAAA"
    )
    wzy_poly = (
        b"MNNKVIILGAAGGVGQSLARLLKNRGHEVIGIDNKPKSKIAELLKDRGIKVYYTGDVSD"
        b"KEILKKIIDEYPDIDVLLVTSGTTGTPKGVMLSHDNILSNSAGKWLKDLGIDPNDVVLSI"
    )

    aligner = PairwiseAligner()
    aln_gt = aligner(Sequences.from_bytes([gt_a]), Sequences.from_bytes([gt_b]))
    assert 40.0 <= float(aln_gt.pidents[0]) < 75.0
    assert float(aln_gt.bitscores[0]) > 100.0
    assert aln_gt[0].bitscore == pytest.approx(float(aln_gt.bitscores[0]))

    # Standard CD-HIT at 80% identity splits gt_a and gt_b into 2 separate clusters (total 3 clusters with wzy_poly)
    seqs_3 = Sequences.from_bytes([gt_a, gt_b, wzy_poly])
    roles_3 = np.array(
        [GeneRole.GLYCOSYLTRANSFERASE, GeneRole.GLYCOSYLTRANSFERASE, GeneRole.POLYMERASE],
        dtype=np.int32,
    )
    clusters_cdhit = Clusterer(mode="cdhit", min_identity=80.0).cluster(seqs_3, roles=roles_3)
    assert clusters_cdhit.n_clusters == 3

    # MCL with bitscores + role stratification groups gt_a and gt_b into 1 functional orthogroup (total 2 clusters)
    clusters_mcl = Clusterer(mode="mcl", min_coverage=0.70, inflation=2.0).cluster(seqs_3, roles=roles_3)
    assert clusters_mcl.n_clusters == 2
    assert clusters_mcl.cluster_ids[0] == clusters_mcl.cluster_ids[1]
    assert clusters_mcl.cluster_ids[0] != clusters_mcl.cluster_ids[2]

    # Build a SerotypeDatabase in MCL mode and verify that a divergent query GT joins the MCL orthogroup
    starts_2 = np.array([100, 1200], dtype=np.int32)
    ends_2 = np.array([1000, 2100], dtype=np.int32)
    roles_2 = np.array([GeneRole.GLYCOSYLTRANSFERASE, GeneRole.POLYMERASE], dtype=np.int32)

    loc_a = Locus.from_genes(
        Genes(
            intervals=Intervals(starts=starts_2, ends=ends_2, strands=np.ones(2, dtype=np.int8)),
            translations=Sequences.from_bytes([gt_a, wzy_poly]),
        ),
        roles=roles_2,
        archetype=LocusArchetype.CPS_WZY,
        name="10A",
    )
    loc_b = Locus.from_genes(
        Genes(
            intervals=Intervals(starts=starts_2, ends=ends_2, strands=np.ones(2, dtype=np.int8)),
            translations=Sequences.from_bytes([gt_b, wzy_poly]),
        ),
        roles=roles_2,
        archetype=LocusArchetype.CPS_WZY,
        name="10B",
    )
    loc_q = Locus.from_genes(
        Genes(
            intervals=Intervals(starts=starts_2, ends=ends_2, strands=np.ones(2, dtype=np.int8)),
            translations=Sequences.from_bytes([gt_query, wzy_poly]),
        ),
        roles=roles_2,
        archetype=LocusArchetype.CPS_WZY,
        name="Query10",
    )

    db_mcl = SerotypeDatabase.build([loc_a, loc_b], cluster_mode="mcl", recluster=True)
    assert db_mcl.num_clusters == 2
    assert list(db_mcl.loci[0].cluster_ids) == list(db_mcl.loci[1].cluster_ids)

    matches_mcl = db_mcl.match(loc_q)
    assert matches_mcl[0].target_name == "10B"
    assert matches_mcl[0].jaccard == pytest.approx(1.0)
    assert matches_mcl[1].target_name == "10A"
    assert matches_mcl[1].jaccard == pytest.approx(1.0)
    assert matches_mcl[0].mean_aa_identity > matches_mcl[1].mean_aa_identity

    # 2. Data-driven inference of novel_jaccard_threshold (supervised Youden-J & unsupervised 1D Otsu)
    # Construct 5 reference loci across 3 serogroups:
    # - Serogroup 6: 6A ([1,2,3,4,5,6]) and 6B ([1,2,3,4,5,7]) -> intra-serogroup Jaccard = 5/7 = 0.714
    # - Serogroup 19: 19A ([1,2,10,11,12,13]) and 19F ([1,2,10,11,12,14]) -> intra-serogroup Jaccard = 5/7 = 0.714
    # - Serogroup 23: 23F ([1,2,20,21,22,23]) -> inter-serogroup Jaccard to 6/19 = 2/10 = 0.200
    roles_6 = [
        GeneRole.CAPSULE_ANCHOR,
        GeneRole.PHOSPHATASE,
        GeneRole.COPOLYMERASE,
        GeneRole.INITIATOR,
        GeneRole.FLIPPASE,
        GeneRole.POLYMERASE,
    ]
    sg_loci = Loci.from_list(
        [
            Locus.from_genes(roles_6, clusters=[1, 2, 3, 4, 5, 6], archetype=LocusArchetype.CPS_WZY, name="6A"),
            Locus.from_genes(roles_6, clusters=[1, 2, 3, 4, 5, 7], archetype=LocusArchetype.CPS_WZY, name="6B"),
            Locus.from_genes(roles_6, clusters=[1, 2, 10, 11, 12, 13], archetype=LocusArchetype.CPS_WZY, name="19A"),
            Locus.from_genes(roles_6, clusters=[1, 2, 10, 11, 12, 14], archetype=LocusArchetype.CPS_WZY, name="19F"),
            Locus.from_genes(roles_6, clusters=[1, 2, 20, 21, 22, 23], archetype=LocusArchetype.CPS_WZY, name="23F"),
        ]
    )
    # Supervised inference via parsed serogroup prefixes ("6A"->"6", "19A"->"19", etc.)
    matcher_auto_sg = LocusMatcher(sg_loci, novel_jaccard_threshold="auto")
    # Midpoint between inter-serogroup max (2/10 = 0.20) and intra-serogroup max (5/7 = 0.714) is ~0.457
    assert 0.35 <= matcher_auto_sg.novel_jaccard_threshold <= 0.60

    db_auto = SerotypeDatabase.build(
        sg_loci,
        serogroups=["6", "6", "19", "19", "23"],
        novel_jaccard_threshold="auto",
    )
    assert db_auto.novel_jaccard_threshold == pytest.approx(matcher_auto_sg.novel_jaccard_threshold)
    assert db_auto.infer_novel_threshold() == pytest.approx(matcher_auto_sg.novel_jaccard_threshold)

    # Unsupervised 1D Otsu inference when names do not encode serogroups and serogroups are omitted
    unnamed_loci = sg_loci.with_names(["Loc_I", "Loc_II", "Loc_III", "Loc_IV", "Loc_V"])
    matcher_otsu = LocusMatcher(unnamed_loci, novel_jaccard_threshold="auto")
    assert 0.35 <= matcher_otsu.novel_jaccard_threshold <= 0.60


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
    from serogen.core.seq.clustering import Clusterer

    # Construct two twilight-zone homologs with a shared conserved 20-aa catalytic motif (~32 bits)
    motif = b"DEDDVSVIVTGGAGFIGWYL"
    flank_a = b"MKKILVTG" * 6 + motif + b"ALRLLKAGHEVVVLD" * 6
    flank_b = b"MNRVFIAA" * 6 + motif + b"SIQYYQNNTRIIIFG" * 6
    unrelated = b"MWKLLVLLPLLAASAAAWZAWZAWZAWZAWZAWZA" * 4

    seqs = Sequences.from_bytes([flank_a, flank_b, unrelated])
    roles = np.array(
        [int(GeneRole.GLYCOSYLTRANSFERASE), int(GeneRole.GLYCOSYLTRANSFERASE), int(GeneRole.CAPSULE_ANCHOR)],
        dtype=np.int32,
    )

    # Without active-site embeddings, min_bitscore=45.0 keeps the ~32-bit twilight-zone pair separate
    c_no_emb = Clusterer(mode="mcl", min_identity=25.0, min_bitscore=45.0).cluster(seqs, roles=roles)
    assert c_no_emb.cluster_ids[0] != c_no_emb.cluster_ids[1]

    # With shared active-site motif embeddings (cosine similarity > 0.95), twilight-zone rescue merges 0 & 1
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
    from serogen.core.graph import (
        k_best_arborescences,
        maximum_spanning_arborescence,
        minimum_spanning_arborescence,
    )

    # 1. 4-node graph where greedy incoming edges form a directed cycle (1 -> 2 -> 3 -> 1)
    w_2d = np.full((4, 4), -1e9, dtype=np.float64)
    w_2d[0, 1] = 5.0
    w_2d[0, 2] = 4.0
    w_2d[0, 3] = 2.0
    # Greedy cycle of weight 10 + 10 + 10 = 30
    w_2d[1, 2] = 10.0
    w_2d[2, 3] = 10.0
    w_2d[3, 1] = 10.0
    # Optimal arborescence enters cycle at node 1 (w[0,1]=5) and takes 1->2 (10) and 2->3 (10) -> total 25.0
    res = maximum_spanning_arborescence(w_2d, root=0)
    parents, score = res
    assert score == pytest.approx(25.0)
    assert list(parents) == [-1, 0, 1, 2]

    # Minimum spanning arborescence on costs = -w_2d
    min_res = minimum_spanning_arborescence(-w_2d, root=0)
    assert min_res.score == pytest.approx(-25.0)
    assert list(min_res.parents) == [-1, 0, 1, 2]

    # 2. 3D (N=3, N=3, S=2) tensor where children 1 and 2 both prefer parent 0 slot 0,
    # but slot_exclusivity=True forces one child onto slot 1
    w_3d = np.full((3, 3, 2), -1e9, dtype=np.float64)
    w_3d[0, 1, 0] = 10.0  # child 1 prefers (parent=0, slot=0)
    w_3d[0, 1, 1] = 6.0
    w_3d[0, 2, 0] = 9.0  # child 2 also prefers (parent=0, slot=0)
    w_3d[0, 2, 1] = 8.0  # child 2 on slot=1 gives 10.0 + 8.0 = 18.0
    w_3d[1, 2, 0] = 5.0
    w_3d[2, 1, 0] = 4.0

    # Without slot exclusivity, both pick (0, slot=0) -> 19.0
    res_unconstrained = maximum_spanning_arborescence(w_3d, root=0, slot_exclusivity=False)
    assert res_unconstrained.score == pytest.approx(19.0)
    assert list(res_unconstrained.slots) == [-1, 0, 0]

    # With slot exclusivity (default), optimal is 0->1 (slot 0, 10.0) + 0->2 (slot 1, 8.0) = 18.0
    res_constrained = maximum_spanning_arborescence(w_3d, root=0, slot_exclusivity=True)
    assert res_constrained.score == pytest.approx(18.0)
    assert list(res_constrained.parents) == [-1, 0, 0]
    assert list(res_constrained.slots) == [-1, 0, 1]

    # Top-3 constrained arborescences
    top3 = k_best_arborescences(w_3d, root=0, k=3, slot_exclusivity=True)
    assert len(top3) == 3
    assert [round(t.score, 4) for t in top3] == [18.0, 15.0, 15.0]


def test_relational_mpnn_forward_backward_clamped_and_sinkhorn(tmp_path: Path) -> None:
    """Verify RelationalMPNN CSR message passing, clamped anchors, tree edge scoring, training, and Sinkhorn."""
    from serogen.core.graph import (
        RelationalMPNN,
        build_multihop_csr,
        maximum_spanning_arborescence,
        sinkhorn_bipartite_matching,
    )

    # 1. Sinkhorn bipartite matching produces doubly stochastic matrix
    log_alpha = np.array([[2.0, -1.0, 0.0], [-1.0, 3.0, 0.5], [0.0, 0.0, 2.5]], dtype=np.float32)
    p_mat = sinkhorn_bipartite_matching(log_alpha, n_iters=25, tau=0.5)
    assert p_mat.shape == (3, 3)
    assert np.allclose(p_mat.sum(axis=0), 1.0, atol=1e-2)
    assert np.allclose(p_mat.sum(axis=1), 1.0, atol=1e-2)
    assert list(np.argmax(p_mat, axis=1)) == [0, 1, 2]

    # 2. Multi-hop CSR with virtual global pool node
    node_offsets = np.array([0, 4], dtype=np.int32)
    edge_sources = np.array([0, 1, 2], dtype=np.int32)
    edge_targets = np.array([1, 2, 3], dtype=np.int32)
    csr = build_multihop_csr(node_offsets, edge_sources, edge_targets, include_2hop=True, add_virtual_pool=True)
    assert csr.n_nodes == 5  # 4 real nodes + 1 virtual global pool node
    assert csr.n_relations == 8

    # 3. Train RelationalMPNN on joint node classification + directed tree parent/slot prediction
    rng = np.random.default_rng(123)
    x = rng.normal(0.0, 1.0, size=(4, 16)).astype(np.float32)
    node_targets = np.array([0, 1, 2, 3], dtype=np.int32)
    clamped_mask = np.array([1, 0, 0, 0], dtype=np.uint8)
    edge_nodes = np.array([0, 1, 2, 3], dtype=np.int32)
    parent_targets = np.array([-1, 0, 1, 1], dtype=np.int32)  # Tree: 0 -> 1, 1 -> 2 (slot 0), 1 -> 3 (slot 2)
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

    # Verify node predictions and decoded arborescence match ground-truth tree
    probs = mpnn.predict_node_proba(x, csr, clamped_mask=clamped_mask)[:4]
    assert list(np.argmax(probs, axis=1)) == [0, 1, 2, 3]

    h_enc = mpnn.encode(x, csr, clamped_mask=clamped_mask)
    edge_logits = mpnn.score_edges(h_enc, node_indices=edge_nodes)
    tree = maximum_spanning_arborescence(edge_logits, root=0, slot_exclusivity=True)
    assert list(tree.parents) == [-1, 0, 1, 1]
    assert list(tree.slots) == [-1, 0, 0, 2]

    # Save and reload
    save_path = mpnn.save(tmp_path / "mpnn_test.npz")
    reloaded = RelationalMPNN.load(save_path)
    assert np.allclose(
        reloaded.predict_node_proba(x, csr, clamped_mask=clamped_mask),
        mpnn.predict_node_proba(x, csr, clamped_mask=clamped_mask),
    )
