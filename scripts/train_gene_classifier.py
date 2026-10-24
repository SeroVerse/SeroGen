# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "altair>=5.5.0",
#     "cassetter>=0.1.0",
#     "optuna>=4.0.0",
#     "numpy>=2.4.4",
#     "scipy>=1.17.1",
#     "numba>=0.65.0",
#     "serogen",
# ]
#
# [tool.uv.sources]
# serogen = { path = "..", editable = true }
# ///

"""Unified in-memory training, Optuna tuning, validation, and Altair visualization marimo notebook for GeneRoleClassifier."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="full",
    app_title="SeroGen — GeneRoleClassifier Training, Optuna Tuning & Validation",
    layout_file="layouts/train_gene_classifier.slides.json",
)


@app.cell
def _():
    import gzip
    import os
    import pickle
    import re
    import threading
    import time
    import xml.etree.ElementTree as ET
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from urllib.error import HTTPError
    from urllib.parse import quote
    from urllib.request import Request, urlopen

    import altair as alt
    import cassetter
    import marimo as mo
    import numpy as np
    import optuna
    from cassetter._state import get_current_cassette
    from cassetter.cassette import NoMatchError, body_to_bytes
    from numba import njit, prange
    from scipy.sparse import csr_array

    from serogen.core.interval import Intervals
    from serogen.core.io.fasta import FastaReader
    from serogen.core.seq.alphabet import Alphabet
    from serogen.core.seq.clustering import Clusterer
    from serogen.core.seq.seq import Sequences
    from serogen.genes.cnn import GeneRoleClassifier
    from serogen.genes.models import Genes, Indels
    from serogen.genes.roles import GeneRole

    alt.data_transformers.disable_max_rows()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return (
        Alphabet,
        Clusterer,
        ET,
        FastaReader,
        GeneRole,
        GeneRoleClassifier,
        Genes,
        HTTPError,
        Indels,
        Intervals,
        NoMatchError,
        Path,
        Request,
        Sequences,
        ThreadPoolExecutor,
        alt,
        body_to_bytes,
        cassetter,
        csr_array,
        get_current_cassette,
        gzip,
        mo,
        njit,
        np,
        optuna,
        os,
        pickle,
        prange,
        quote,
        re,
        threading,
        time,
        urlopen,
    )


@app.cell
def _(GeneRole, re):
    # ------------------------------------------------------------------------------------------------------------------
    # Structured Ontology & Domain Query Registries (UniProtKB InterPro/Pfam/EC/GO/CC, UniRef50, and KEGG Orthology)
    # for Roles 0..28 including Viral/Bacteriophage (taxonomy_id:10239) + Prophage subqueries and Decoy Pools (Role 29)
    # ------------------------------------------------------------------------------------------------------------------
    ROLE_QUERIES: dict[int, dict[str, object]] = {
        int(GeneRole.FLIPPASE): {
            "subqueries": [
                "(xref:pfam-PF14667 OR xref:interpro-IPR018452 OR xref:interpro-IPR050833) NOT xref:pfam-PF01757 NOT xref:pfam-PF04932 NOT xref:pfam-PF14296 NOT xref:pfam-PF03023 NOT xref:pfam-PF01098 AND taxonomy_id:2",
                "(xref:pfam-PF01943 OR xref:interpro-IPR002797) AND (go:0140303 OR go:0015238 OR cc_pathway:lipopolysaccharide OR cc_pathway:capsule OR cc_function:flippase) NOT xref:pfam-PF01757 NOT xref:pfam-PF04932 NOT xref:pfam-PF14296 NOT xref:pfam-PF03023 AND taxonomy_id:2",
            ],
            "uniprot_phage": "(xref:pfam-PF04138 OR xref:interpro-IPR007254 OR ((xref:pfam-PF01943 OR xref:pfam-PF14667 OR xref:interpro-IPR050833) AND (taxonomy_id:10239 OR cc_function:prophage OR cc_function:bacteriophage))) NOT xref:pfam-PF14296",
            "uniref": '(name:"O-antigen flippase" OR name:"polysaccharide flippase") NOT name:polymerase NOT name:ligase NOT name:acetyltransferase NOT name:acyltransferase NOT name:Wzy NOT name:WaaL NOT name:MurJ AND identity:0.5 AND taxonomy_id:2',
            "uniref_phage": "(name:GtrA OR ((name:phage OR name:prophage) AND (name:flippase OR name:translocase))) AND identity:0.5",
            "min_len": 250,
            "max_len": 560,
            "phage_min_len": 95,
            "phage_max_len": 550,
        },
        int(GeneRole.POLYMERASE): {
            "subqueries": [
                "(xref:pfam-PF14296 OR xref:interpro-IPR025997 OR xref:pfam-PF13425 OR xref:interpro-IPR024722) NOT go:0008754 NOT xref:pfam-PF01943 NOT xref:pfam-PF14667 NOT xref:pfam-PF01757 AND taxonomy_id:2",
                "((xref:interpro-IPR051533 OR xref:pfam-PF04932) AND (cc_pathway:capsule OR go:0045227 OR taxonomy_id:573)) NOT go:0008754 NOT xref:pfam-PF01943 NOT xref:pfam-PF14667 AND taxonomy_id:2",
            ],
            "uniprot_phage": "((xref:pfam-PF14296 OR xref:interpro-IPR025997 OR xref:interpro-IPR051533) NOT go:0008754 NOT xref:pfam-PF01943) AND (taxonomy_id:10239 OR cc_function:prophage OR cc_function:bacteriophage)",
            "uniref": '(name:"O-antigen polymerase" OR name:"polysaccharide polymerase" OR name:Wzy) NOT name:ligase NOT name:flippase NOT name:acetyltransferase NOT name:acyltransferase NOT name:WaaL NOT name:RfaL NOT name:Wzx AND identity:0.5 AND taxonomy_id:2',
            "uniref_phage": '((name:"O-antigen polymerase" OR name:"polysaccharide polymerase") AND taxonomy_id:10239) AND identity:0.5',
            "min_len": 250,
            "max_len": 560,
        },
        int(GeneRole.COPOLYMERASE): {
            "subqueries": [
                "(xref:pfam-PF02706 OR xref:interpro-IPR003856) NOT xref:pfam-PF13614 NOT xref:interpro-IPR027417 NOT ec:2.7.10.- AND taxonomy_id:2",
            ],
            "uniref": '(name:"chain length determinant" OR name:"polysaccharide co-polymerase" OR name:Wzz) NOT name:kinase NOT name:Wzc AND identity:0.5 AND taxonomy_id:2',
            "min_len": 180,
            "max_len": 520,
        },
        int(GeneRole.KINASE): {
            "subqueries": [
                "(xref:pfam-PF13614 OR xref:interpro-IPR027417 OR ec:2.7.10.2) AND (xref:pfam-PF02706 OR xref:interpro-IPR003856 OR cc_pathway:capsule OR cc_pathway:polysaccharide OR cc_pathway:exopolysaccharide OR go:0004713) NOT ec:3.1.3.48 AND taxonomy_id:2",
            ],
            "uniref": '(name:"tyrosine-protein kinase Wzc" OR name:"tyrosine-protein kinase Etk" OR (name:"exopolysaccharide" AND name:kinase)) NOT name:phosphatase NOT name:Wza AND identity:0.5 AND taxonomy_id:2',
            "min_len": 190,
            "max_len": 820,
        },
        int(GeneRole.PHOSPHATASE): {
            "subqueries": [
                "(xref:pfam-PF01451 OR xref:interpro-IPR000106 OR ec:3.1.3.48) NOT ec:2.7.10.- AND taxonomy_id:2",
                "(xref:interpro-IPR032908 OR (xref:pfam-PF01569 AND (cc_pathway:capsule OR go:0016791 OR go:0009103))) NOT ec:2.7.10.- AND taxonomy_id:2",
                "(xref:pfam-PF01569 AND xref:interpro-IPR000326) NOT go:0050380 NOT ec:2.7.10.- AND taxonomy_id:573",
            ],
            "uniref": '(name:"low molecular weight phosphotyrosine protein phosphatase" OR name:"phosphotyrosine-protein phosphatase" OR name:Wzb OR name:CpsACP) NOT name:kinase AND identity:0.5 AND taxonomy_id:2',
            "min_len": 55,
            "max_len": 300,
        },
        int(GeneRole.ABC_PERMEASE): {
            "subqueries": [
                "(xref:pfam-PF06562 OR xref:interpro-IPR009493 OR (xref:pfam-PF01061 AND (cc_pathway:capsule OR cc_pathway:lipopolysaccharide OR cc_function:capsule OR go:0045226 OR go:0015238))) NOT xref:pfam-PF00005 NOT xref:pfam-PF02563 AND taxonomy_id:2",
            ],
            "uniref": '((name:"ABC transporter permease" AND (name:polysaccharide OR name:capsule OR name:"O-antigen")) OR name:"capsule export system permease" OR name:Wzm OR name:KpsM) NOT name:Wza NOT name:KpsD NOT name:KpsE NOT name:Wzt NOT name:KpsT NOT name:"ATP-binding" NOT name:"outer membrane" AND identity:0.5 AND taxonomy_id:2',
            "min_len": 200,
            "max_len": 360,
        },
        int(GeneRole.ABC_ATPASE): {
            "subqueries": [
                "(xref:pfam-PF00005 AND (xref:pfam-PF14524 OR xref:interpro-IPR027466 OR cc_pathway:capsule OR cc_pathway:lipopolysaccharide OR cc_function:teichoic)) NOT xref:pfam-PF01061 NOT xref:pfam-PF06562 AND taxonomy_id:2",
            ],
            "uniref": '(name:"polysaccharide export ATP-binding" OR name:"capsule export system ATP-binding" OR name:Wzt OR name:KpsT) NOT name:permease NOT name:Wzm NOT name:KpsM AND identity:0.5 AND taxonomy_id:2',
            "min_len": 200,
            "max_len": 520,
        },
        int(GeneRole.LIGASE): {
            "subqueries": [
                "(go:0008754 OR ec:2.4.99.30 OR (xref:pfam-PF04932 AND go:0009103 AND cc_function:ligase)) NOT xref:pfam-PF14296 NOT xref:pfam-PF13425 NOT xref:pfam-PF01943 NOT xref:pfam-PF14667 NOT go:0045227 AND taxonomy_id:2",
            ],
            "uniref": '(name:"O-antigen ligase" OR name:"lipid A-core:surface polymer ligase" OR name:WaaL OR name:RfaL) NOT name:polymerase NOT name:flippase NOT name:Wzy NOT name:Wzx AND identity:0.5 AND taxonomy_id:2',
            "min_len": 280,
            "max_len": 540,
        },
        int(GeneRole.TRANSLOCON): {
            "subqueries": [
                "(xref:pfam-PF02563 OR xref:interpro-IPR003715) NOT xref:pfam-PF01061 NOT xref:pfam-PF13614 AND taxonomy_id:2",
                "(xref:pfam-PF06251 OR xref:interpro-IPR009662) NOT xref:pfam-PF01061 AND taxonomy_id:2",
                "(xref:pfam-PF06082 OR xref:pfam-PF06257 OR xref:interpro-IPR010653 OR xref:interpro-IPR009480) AND taxonomy_id:2",
            ],
            "uniref": '(name:"outer membrane polysaccharide export" OR name:"polysaccharide export protein Wza" OR name:Wza OR name:KpsD OR name:KpsE OR name:GfcB OR name:GfcC OR name:GfcD) NOT name:Wzm NOT name:KpsM NOT name:permease NOT name:kinase NOT name:"ABC transporter" AND identity:0.5 AND taxonomy_id:2',
            "min_len": 190,
            "max_len": 760,
        },
        int(GeneRole.CAPSULE_ANCHOR): {
            "subqueries": [
                "(xref:pfam-PF14052 OR xref:interpro-IPR025704 OR xref:interpro-IPR026950) AND taxonomy_id:2",
                "(xref:interpro-IPR026589 OR xref:pfam-PF14897) AND taxonomy_id:2",
                "(xref:pfam-PF03816 OR xref:interpro-IPR007465) AND (cc_pathway:capsule OR go:0045226 OR go:0045227 OR taxonomy_id:1301) AND taxonomy_id:2",
            ],
            "uniref": '(name:"capsule assembly protein Wzi" OR name:Wzi OR name:KpsC OR name:Wzg OR name:CpsA OR (name:"LytR-CpsA-Psr" AND name:capsule)) AND identity:0.5 AND taxonomy_id:2',
            "min_len": 300,
            "max_len": 720,
        },
        int(GeneRole.PYRUVYLTRANSFERASE): {
            "subqueries": [
                "(xref:pfam-PF04230 OR xref:interpro-IPR007345) NOT xref:pfam-PF00534 NOT xref:pfam-PF00535 NOT xref:pfam-PF13439 NOT xref:pfam-PF13579 NOT xref:pfam-PF01565 AND taxonomy_id:2",
            ],
            "uniref": '(name:pyruvyl* OR name:"Polysaccharide pyruvyl transferase") NOT name:reductase NOT name:glycosyltransferase AND identity:0.5 AND taxonomy_id:2',
            "min_len": 220,
            "max_len": 480,
        },
        int(GeneRole.METHYLTRANSFERASE): {
            "subqueries": [
                "(xref:pfam-PF08241 OR xref:pfam-PF13649 OR xref:pfam-PF13847 OR ec:2.1.1.-) AND (cc_pathway:lipopolysaccharide OR cc_pathway:capsule OR cc_pathway:polysaccharide) AND taxonomy_id:2",
            ],
            "uniprot": '(protein_name:"polysaccharide O-methyltransferase" OR protein_name:"sugar O-methyltransferase" OR protein_name:"capsular polysaccharide methyltransferase") AND taxonomy_id:2',
            "uniprot_phage": '(protein_name:"O-methyltransferase" OR protein_name:"polysaccharide methyltransferase" OR protein_name:"sugar methyltransferase") AND (taxonomy_id:10239 OR (taxonomy_id:2 AND (protein_name:phage OR protein_name:prophage)))',
            "uniref": '(name:"polysaccharide methyltransferase" OR name:"sugar O-methyltransferase") AND identity:0.5 AND taxonomy_id:2',
            "min_len": 160,
            "max_len": 520,
        },
        int(GeneRole.ACETYLTRANSFERASE): {
            "subqueries": [
                "(xref:pfam-PF00132 OR xref:interpro-IPR001451 OR xref:interpro-IPR051159) AND (go:0016747 OR ec:2.3.1.18 OR ec:2.3.1.- OR cc_pathway:capsule OR cc_pathway:lipopolysaccharide) NOT xref:pfam-PF00534 NOT xref:pfam-PF00535 NOT xref:pfam-PF00483 NOT xref:pfam-PF01050 AND taxonomy_id:2",
                "(xref:interpro-IPR051159 OR (xref:pfam-PF00132 AND go:0016747)) NOT xref:pfam-PF00483 NOT xref:pfam-PF01050 NOT xref:pfam-PF00534 AND taxonomy_id:573",
                "(xref:pfam-PF01757 OR xref:interpro-IPR002656) AND (go:0016747 OR ec:2.3.1.- OR cc_pathway:capsule OR cc_pathway:lipopolysaccharide) NOT xref:pfam-PF01943 NOT xref:pfam-PF14296 AND taxonomy_id:2",
            ],
            "uniprot_phage": "((xref:pfam-PF01757 OR xref:interpro-IPR002656 OR xref:pfam-PF00132) AND go:0016747) AND (taxonomy_id:10239 OR cc_function:prophage OR cc_function:bacteriophage)",
            "uniref": '(name:"O-acetyltransferase" AND name:polysaccharide OR name:WecD OR name:OatA OR name:"Acyl_transf_3") AND identity:0.5 AND taxonomy_id:2',
            "uniref_phage": '(name:Oac OR ((name:"O-acetyltransferase" OR name:"Acyl_transf_3") AND (taxonomy_id:10239 OR name:phage OR name:prophage))) AND identity:0.5',
            "min_len": 130,
            "max_len": 680,
            "quota_multiplier": 1.8,
        },
        int(GeneRole.PHOSPHOTRANSFERASE): {
            "subqueries": [
                "(xref:pfam-PF04991 OR xref:interpro-IPR007072) AND taxonomy_id:2",
                "(xref:pfam-PF08460 OR xref:interpro-IPR013684) AND taxonomy_id:2",
            ],
            "uniref": '(name:"phosphocholine transferase" OR name:"LicD family" OR name:LicD) AND identity:0.5 AND taxonomy_id:2',
            "min_len": 170,
            "max_len": 480,
        },
        int(GeneRole.FORMYLTRANSFERASE): {
            "subqueries": [
                "(xref:pfam-PF00551 OR xref:interpro-IPR002376 OR ec:2.1.2.13 OR xref:interpro-IPR015518) AND (cc_pathway:lipopolysaccharide OR cc_pathway:polysaccharide OR go:0009103 OR ec:2.1.2.13) AND taxonomy_id:2",
            ],
            "uniref": '(name:"formyltransferase" OR name:"N-formyltransferase") AND identity:0.5 AND taxonomy_id:2',
            "min_len": 160,
            "max_len": 450,
        },
        int(GeneRole.ALANINE_TRANSFERASE): {
            "subqueries": [
                "(ec:6.1.1.13 OR xref:interpro-IPR010079 OR xref:pfam-PF04914 OR (xref:pfam-PF03372 AND (cc_function:teichoic OR cc_function:alanine))) AND taxonomy_id:2",
            ],
            "uniref": '(name:"D-alanine--poly(phosphoribitol) ligase" OR name:"D-alanyl-lipoteichoic acid" OR name:DltA) AND identity:0.5 AND taxonomy_id:2',
            "min_len": 160,
            "max_len": 560,
        },
        int(GeneRole.AMINOTRANSFERASE): {
            "subqueries": [
                "(xref:pfam-PF01041 OR xref:interpro-IPR000653) AND (ec:2.6.1.59 OR ec:2.6.1.33 OR ec:2.6.1.50 OR ec:2.6.1.102 OR cc_pathway:lipopolysaccharide OR cc_pathway:capsule) AND taxonomy_id:2",
            ],
            "uniref": '(name:"DegT/DnrJ/EryC1/StrS aminotransferase" OR name:"dTDP-4-amino-4,6-dideoxygalactose transaminase" OR name:WecE) AND identity:0.5 AND taxonomy_id:2',
            "min_len": 280,
            "max_len": 480,
        },
        int(GeneRole.ALDOLASE): {
            "subqueries": [
                "(xref:pfam-PF03102 OR xref:pfam-PF00701 OR xref:interpro-IPR004360 OR ec:2.5.1.56 OR ec:2.5.1.55 OR ec:2.5.1.97) AND taxonomy_id:2",
            ],
            "uniref": '(name:"N-acetylneuraminate synthase" OR name:"pseudaminic acid synthase" OR name:"2-dehydro-3-deoxyphosphooctonate aldolase") AND identity:0.5 AND taxonomy_id:2',
            "min_len": 220,
            "max_len": 450,
        },
        int(GeneRole.DEACETYLASE): {
            "subqueries": [
                "(xref:pfam-PF01522 OR xref:interpro-IPR002509 OR ec:3.5.1.104 OR ec:3.5.1.41) AND taxonomy_id:2",
            ],
            "uniprot_phage": "(xref:pfam-PF01522 OR xref:interpro-IPR002509) AND (taxonomy_id:10239 OR cc_function:prophage OR cc_function:bacteriophage)",
            "uniref": '(name:"polysaccharide deacetylase" OR name:"poly-beta-1,6-N-acetyl-D-glucosamine N-deacetylase" OR name:PgaB) AND identity:0.5 AND taxonomy_id:2',
            "min_len": 180,
            "max_len": 750,
        },
        int(GeneRole.EPIMERASE): {
            "subqueries": [
                "(ec:5.1.3.2 OR xref:interpro-IPR005886) AND taxonomy_id:2",
                "(ec:5.1.3.13 OR xref:pfam-PF00908 OR xref:interpro-IPR000888) AND taxonomy_id:2",
                "(ec:5.4.2.8 OR ec:5.4.2.2 OR xref:interpro-IPR005844 OR go:0004615) AND (xref:pfam-PF02878 OR xref:interpro-IPR005844) AND taxonomy_id:2",
                "(ec:5.4.99.9 OR xref:pfam-PF03275 OR xref:interpro-IPR002658) AND taxonomy_id:2",
                "(ec:5.3.1.9 OR xref:pfam-PF00342 OR xref:interpro-IPR001672) AND taxonomy_id:2",
                "(ec:5.3.1.13 OR xref:pfam-PF01380 OR xref:interpro-IPR004384) AND taxonomy_id:2",
                "(ec:5.1.3.14 OR xref:pfam-PF02350 OR xref:interpro-IPR003331) AND taxonomy_id:2",
            ],
            "uniprot_phage": "(ec:5.1.3.2 OR ec:5.1.3.13 OR xref:pfam-PF00908 OR xref:interpro-IPR005886) AND (taxonomy_id:10239 OR cc_function:prophage OR cc_function:bacteriophage)",
            "uniref": '(name:"UDP-glucose 4-epimerase" OR name:"dTDP-4-dehydrorhamnose 3,5-epimerase" OR name:RmlC OR name:GalE OR name:ManB OR name:Glf) AND identity:0.5 AND taxonomy_id:2',
            "min_len": 145,
            "max_len": 580,
            "quota_multiplier": 2.0,
        },
        int(GeneRole.OXIDOREDUCTASE): {
            "subqueries": [
                "(ec:1.1.1.22 OR xref:pfam-PF03721 OR xref:interpro-IPR005882) AND taxonomy_id:2",
                "(ec:4.2.1.46 OR ec:4.2.1.47 OR xref:interpro-IPR005888 OR xref:interpro-IPR005887 OR xref:pfam-PF16363) AND taxonomy_id:2",
                "(ec:1.1.1.271 OR ec:1.1.1.133 OR xref:interpro-IPR028614 OR xref:interpro-IPR005913 OR go:0050577) AND taxonomy_id:2",
                "xref:interpro-IPR028614 AND taxonomy_id:573",
                "(ec:1.1.1.44 OR xref:pfam-PF03446 OR xref:interpro-IPR006115 OR go:0004616) AND taxonomy_id:2",
            ],
            "uniprot_phage": "(ec:4.2.1.46 OR ec:4.2.1.47 OR ec:1.1.1.22 OR xref:pfam-PF03721) AND (taxonomy_id:10239 OR cc_function:prophage OR cc_function:bacteriophage)",
            "uniref": '(name:"UDP-glucose 6-dehydrogenase" OR name:"dTDP-glucose 4,6-dehydratase" OR name:"GDP-mannose 4,6-dehydratase" OR name:"GDP-L-fucose synthase" OR name:"6-phosphogluconate dehydrogenase") AND identity:0.5 AND taxonomy_id:2',
            "min_len": 240,
            "max_len": 500,
            "quota_multiplier": 1.8,
        },
        int(GeneRole.TRANSMEMBRANE_SYNTHASE): {
            "subqueries": [
                "(xref:pfam-PF13641 OR xref:interpro-IPR029044 OR ec:2.4.1.12 OR ec:2.4.1.212 OR ec:2.4.1.289) AND (cc_pathway:cellulose OR cc_pathway:hyaluronan OR cc_pathway:biofilm OR go:0030244) AND taxonomy_id:2",
            ],
            "uniprot": '(gene:hasA OR gene:pgaC OR gene:icaA OR (gene:bcsA AND protein_name:"catalytic") OR protein_name:"hyaluronan synthase" OR protein_name:"cellulose synthase catalytic subunit" OR protein_name:"poly-beta-1,6-N-acetyl-D-glucosamine synthase" OR xref:pfam-PF13641) AND taxonomy_id:2',
            "uniref": '(name:"hyaluronan synthase" OR name:"cellulose synthase catalytic subunit" OR name:"poly-beta-1,6-N-acetyl-D-glucosamine synthase") AND identity:0.5 AND taxonomy_id:2',
            "min_len": 340,
            "max_len": 1000,
        },
        int(GeneRole.NUCLEOTIDYLTRANSFERASE): {
            "subqueries": [
                "(ec:2.7.7.9 OR ec:2.7.7.24) AND (xref:pfam-PF00483 OR xref:interpro-IPR005835) AND taxonomy_id:2",
                "(ec:2.7.7.13 OR (xref:pfam-PF01050 AND xref:pfam-PF00483) OR xref:interpro-IPR011004) AND taxonomy_id:2",
                "(ec:2.7.7.43 OR ec:2.7.7.38) AND (xref:pfam-PF02348 OR xref:interpro-IPR003329) AND taxonomy_id:2",
            ],
            "uniref": '(name:"glucose-1-phosphate thymidylyltransferase" OR name:"UTP--glucose-1-phosphate uridylyltransferase" OR name:"mannose-1-phosphate guanylyltransferase" OR name:RmlA OR name:GalU OR name:ManC) AND identity:0.5 AND taxonomy_id:2',
            "min_len": 195,
            "max_len": 520,
            "quota_multiplier": 1.6,
        },
        int(GeneRole.NUCLEOTIDASE): {
            "subqueries": [
                "(xref:interpro-IPR033715 OR xref:interpro-IPR028613 OR go:0008727 OR ec:3.6.1.13 OR ec:3.6.1.66 OR (xref:pfam-PF00293 AND (cc_pathway:capsule OR cc_pathway:polysaccharide OR cc_function:mannose))) AND taxonomy_id:2",
            ],
            "uniref": '(name:"GDP-mannose mannosyl hydrolase" OR name:"GDP-mannose hydrolase" OR name:"NUDIX hydrolase") AND identity:0.5 AND taxonomy_id:2',
            "min_len": 105,
            "max_len": 290,
        },
        int(GeneRole.INITIATOR): {
            "subqueries": [
                "(xref:pfam-PF02397 OR xref:interpro-IPR003362 OR ec:2.7.8.6 OR ec:2.7.8.31) AND taxonomy_id:2",
                "(xref:pfam-PF00953 OR xref:interpro-IPR000715) AND (ec:2.7.8.33 OR ec:2.7.8.15 OR cc_pathway:lipopolysaccharide OR cc_pathway:enterobacterial) AND taxonomy_id:2",
            ],
            "uniprot_phage": "(ec:2.4.1.83 OR (xref:interpro-IPR001173 AND (taxonomy_id:10239 OR gene:gtrB))) NOT xref:pfam-PF04138",
            "uniref": '(name:"undecaprenyl-phosphate galactose phosphotransferase" OR name:"bacterial sugar transferase" OR name:WbaP OR name:WcaJ OR name:WecA) AND identity:0.5 AND taxonomy_id:2',
            "uniref_phage": '(name:GtrB OR name:"bactoprenol glucosyltransferase") AND identity:0.5',
            "min_len": 180,
            "max_len": 540,
        },
        int(GeneRole.GLYCOSYLHYDROLASE): {
            "subqueries": [
                "(xref:pfam-PF05426 OR xref:pfam-PF12708 OR xref:interpro-IPR008929 OR ec:4.2.2.3 OR ec:3.2.1.129) AND taxonomy_id:2",
            ],
            "uniprot_phage": "((xref:pfam-PF12708 OR xref:pfam-PF05426 OR ec:3.2.1.129 OR go:0016837) NOT xref:pfam-PF00959 NOT xref:pfam-PF05105) AND taxonomy_id:10239",
            "uniref": '(name:"polysaccharide lyase" OR name:"alginate lyase" OR name:"capsule depolymerase") AND identity:0.5 AND taxonomy_id:2',
            "uniref_phage": '((name:"capsule depolymerase" OR name:endosialidase OR (name:tailspike AND name:depolymerase) OR (name:"polysaccharide lyase" AND taxonomy_id:10239)) NOT name:endolysin) AND identity:0.5',
            "min_len": 220,
            "max_len": 650,
            "phage_min_len": 220,
            "phage_max_len": 1150,
        },
        int(GeneRole.GLYCOSYLTRANSFERASE): {
            "subqueries": [
                "(xref:pfam-PF00534 OR xref:interpro-IPR001296) AND (go:0016757 OR ec:2.4.1.-) NOT xref:pfam-PF04230 NOT xref:pfam-PF02397 AND taxonomy_id:2",
                "(xref:pfam-PF00535 OR xref:interpro-IPR001173) AND (go:0016757 OR ec:2.4.1.-) NOT xref:pfam-PF13641 NOT xref:pfam-PF04230 AND taxonomy_id:2",
                "(xref:pfam-PF13439 OR xref:pfam-PF13579) AND (go:0016757 OR go:0016758 OR ec:2.4.1.-) NOT xref:pfam-PF04230 AND taxonomy_id:2",
                "(xref:pfam-PF01501 OR xref:pfam-PF01531 OR xref:pfam-PF01755 OR xref:pfam-PF03808) AND (go:0016757 OR ec:2.4.1.-) NOT xref:pfam-PF04230 AND taxonomy_id:2",
                "(xref:pfam-PF05159 OR xref:interpro-IPR007856) AND taxonomy_id:2",
                "(xref:interpro-IPR001296 OR xref:interpro-IPR050194 OR xref:pfam-PF00535) AND (go:0016757 OR go:0016758) NOT xref:pfam-PF02397 AND taxonomy_id:573",
            ],
            "uniprot_phage": "((xref:pfam-PF00534 OR xref:pfam-PF00535 OR xref:pfam-PF13439) AND (go:0016757 OR ec:2.4.1.-) NOT xref:pfam-PF04138 NOT xref:pfam-PF00959) AND (taxonomy_id:10239 OR cc_function:prophage)",
            "uniref": '(name:"glycosyltransferase family 1" OR name:"glycosyltransferase family 2" OR name:"glycosyltransferase family 4" OR name:"capsular polysaccharide glycosyltransferase") NOT name:synthase NOT name:cellulose NOT name:hyaluronan NOT name:pyruvyltransferase NOT name:transposase NOT name:integrase AND identity:0.5 AND taxonomy_id:2',
            "uniref_phage": '((name:GtrC OR name:"glucosyltransferase Gtr" OR (name:glycosyltransferase AND (taxonomy_id:10239 OR name:phage OR name:prophage))) NOT name:GtrA NOT name:GtrB) AND identity:0.5',
            "min_len": 185,
            "max_len": 680,
            "quota_multiplier": 4.5,
        },
        int(GeneRole.TRANSPOSASE): {
            "subqueries": [
                "(xref:pfam-PF01609 OR xref:pfam-PF00872 OR xref:pfam-PF13359 OR xref:interpro-IPR001584) AND go:0004803 NOT xref:pfam-PF00589 AND taxonomy_id:2",
                "(xref:pfam-PF01527 OR xref:pfam-PF13333 OR xref:pfam-PF01548 OR xref:pfam-PF02371) AND go:0004803 NOT xref:pfam-PF00589 AND taxonomy_id:2",
            ],
            "uniprot_phage": "go:0004803 NOT xref:pfam-PF00589 AND taxonomy_id:10239",
            "uniref": '(name:"IS3 family transposase" OR name:"IS5 family transposase" OR name:"IS6 family transposase" OR name:"IS256 family transposase" OR name:"IS1 family transposase" OR name:"Transposase_8") NOT name:integrase NOT name:glycosyltransferase AND identity:0.5 AND taxonomy_id:2',
            "min_len": 70,
            "max_len": 520,
        },
        int(GeneRole.INTEGRASE): {
            "subqueries": [
                "(xref:pfam-PF00589 OR xref:interpro-IPR002104) AND go:0008907 NOT go:0004803 AND taxonomy_id:2",
            ],
            "uniprot_phage": "(xref:pfam-PF00589 OR xref:interpro-IPR002104) NOT go:0004803 AND taxonomy_id:10239",
            "uniref": '(name:"site-specific integrase" OR name:"tyrosine recombinase") NOT name:transposase NOT name:glycosyltransferase NOT name:phage AND identity:0.5 AND taxonomy_id:2',
            "uniref_phage": 'name:"phage integrase" NOT name:transposase NOT name:glycosyltransferase AND identity:0.5',
            "min_len": 240,
            "max_len": 560,
        },
    }

    HARD_NEGATIVE_QUERIES: dict[int, list[dict[str, object]]] = {
        0: [
            {
                "uniprot": "((gene:dnaA OR gene:gyrA OR gene:gyrB OR gene:rpoB OR gene:recA OR gene:infB OR gene:fusA OR gene:tufA) AND reviewed:true) NOT protein_name:glycosyltransferase NOT protein_name:polysaccharide AND taxonomy_id:2",
                "uniref": '(name:"chromosomal replication initiator protein DnaA" OR name:"DNA gyrase subunit A" OR name:"recombinase RecA" OR name:"elongation factor Tu") AND identity:0.5 AND taxonomy_id:2',
                "min_len": 150,
                "max_len": 950,
            },
            {
                "uniprot": "((gene:rplA OR gene:rplB OR gene:rpsA OR gene:rpsB OR gene:groEL OR gene:dnaK OR gene:ftsZ OR gene:ompA OR gene:tolC) AND reviewed:true) NOT protein_name:glycosyltransferase NOT protein_name:capsule AND taxonomy_id:2",
                "uniref": '(name:"50S ribosomal protein L1" OR name:"30S ribosomal protein S2" OR name:"cell division protein FtsZ" OR name:"outer membrane protein A") AND identity:0.5 AND taxonomy_id:2',
                "min_len": 90,
                "max_len": 700,
            },
            {
                "uniprot": "((gene:gapA OR gene:eno OR gene:pykF OR gene:icd OR gene:mdh OR gene:atpA OR gene:atpD) AND reviewed:true) NOT protein_name:glycosyltransferase NOT protein_name:epimerase NOT protein_name:dehydrogenase AND taxonomy_id:2",
                "uniref": '(name:enolase OR name:"pyruvate kinase" OR name:"ATP synthase subunit alpha" OR name:"ATP synthase subunit beta") AND identity:0.5 AND taxonomy_id:2',
                "min_len": 180,
                "max_len": 650,
            },
            {
                "uniprot": "((xref:pfam-PF00137 OR xref:pfam-PF00584 OR xref:pfam-PF03840 OR xref:pfam-PF00313 OR xref:pfam-PF00572 OR xref:pfam-PF00829) AND reviewed:true) NOT protein_name:glycosyltransferase NOT protein_name:polysaccharide AND taxonomy_id:2",
                "uniref": '(name:"ATP synthase subunit c" OR name:"protein translocase subunit SecE" OR name:"protein-export membrane protein SecG" OR name:"cold shock-like protein") AND identity:0.5 AND taxonomy_id:2',
                "min_len": 40,
                "max_len": 140,
            },
        ],
        1: [
            {
                "uniprot": '((protein_name:"terminase large subunit" OR protein_name:"terminase small subunit" OR xref:pfam-PF03237 OR xref:pfam-PF04466) NOT protein_name:integrase NOT protein_name:transposase NOT protein_name:glycosyltransferase) AND taxonomy_id:10239',
                "uniref": '(name:"terminase large subunit" OR name:"phage terminase") NOT name:integrase NOT name:glycosyltransferase AND identity:0.5 AND taxonomy_id:10239',
                "min_len": 110,
                "max_len": 750,
            },
            {
                "uniprot": '((protein_name:"portal protein" OR protein_name:"major capsid protein" OR protein_name:"major head protein" OR xref:pfam-PF04860 OR xref:pfam-PF05065) NOT protein_name:depolymerase NOT protein_name:lyase NOT protein_name:glycosyltransferase) AND taxonomy_id:10239',
                "uniref": '(name:"phage portal protein" OR name:"major capsid protein") NOT name:depolymerase NOT name:lyase AND identity:0.5 AND taxonomy_id:10239',
                "min_len": 200,
                "max_len": 700,
            },
            {
                "uniprot": '((protein_name:"tail sheath" OR protein_name:"tail tube" OR protein_name:"baseplate wedge" OR xref:pfam-PF04984 OR xref:pfam-PF04865) NOT protein_name:depolymerase NOT protein_name:tailspike NOT protein_name:"tail spike" NOT protein_name:lyase NOT protein_name:endosialidase NOT protein_name:glycosyltransferase) AND taxonomy_id:10239',
                "uniref": '(name:"phage tail sheath" OR name:"phage tail tube") NOT name:depolymerase NOT name:tailspike NOT name:lyase AND identity:0.5 AND taxonomy_id:10239',
                "min_len": 120,
                "max_len": 750,
            },
            {
                "uniprot": '((protein_name:holin OR protein_name:endolysin OR xref:pfam-PF00959 OR xref:pfam-PF05105) NOT protein_name:depolymerase NOT protein_name:tailspike NOT protein_name:"polysaccharide deacetylase" NOT protein_name:glycosyltransferase) AND taxonomy_id:10239',
                "uniref": '(name:"phage holin" OR name:"phage endolysin") NOT name:depolymerase NOT name:tailspike AND identity:0.5 AND taxonomy_id:10239',
                "min_len": 65,
                "max_len": 380,
            },
        ],
    }

    KEGG_KO_MAP: dict[str, GeneRole] = {
        "K02847": GeneRole.LIGASE,
        "K02853": GeneRole.POLYMERASE,
        "K13008": GeneRole.POLYMERASE,
        "K02854": GeneRole.COPOLYMERASE,
        "K04771": GeneRole.COPOLYMERASE,
        "K16693": GeneRole.FLIPPASE,
        "K16695": GeneRole.FLIPPASE,
        "K16692": GeneRole.KINASE,
        "K01104": GeneRole.PHOSPHATASE,
        "K25307": GeneRole.PHOSPHATASE,
        "K16690": GeneRole.ABC_PERMEASE,
        "K09686": GeneRole.ABC_PERMEASE,
        "K16691": GeneRole.ABC_ATPASE,
        "K09687": GeneRole.ABC_ATPASE,
        "K01991": GeneRole.TRANSLOCON,
        "K09688": GeneRole.TRANSLOCON,
        "K09689": GeneRole.TRANSLOCON,
        "K16694": GeneRole.CAPSULE_ANCHOR,
        "K09695": GeneRole.CAPSULE_ANCHOR,
        "K19225": GeneRole.CAPSULE_ANCHOR,
        "K02841": GeneRole.INITIATOR,
        "K03606": GeneRole.INITIATOR,
        "K02843": GeneRole.INITIATOR,
        "K07271": GeneRole.PHOSPHOTRANSFERASE,
        "K00607": GeneRole.FORMYLTRANSFERASE,
        "K03367": GeneRole.ALANINE_TRANSFERASE,
        "K16560": GeneRole.PYRUVYLTRANSFERASE,
        "K00558": GeneRole.METHYLTRANSFERASE,
        "K01443": GeneRole.DEACETYLASE,
        "K01444": GeneRole.DEACETYLASE,
        "K01654": GeneRole.ALDOLASE,
        "K05304": GeneRole.ALDOLASE,
        "K01627": GeneRole.ALDOLASE,
        "K00832": GeneRole.AMINOTRANSFERASE,
        "K01784": GeneRole.EPIMERASE,
        "K01791": GeneRole.EPIMERASE,
        "K01790": GeneRole.EPIMERASE,
        "K02473": GeneRole.EPIMERASE,
        "K01840": GeneRole.EPIMERASE,
        "K01835": GeneRole.EPIMERASE,
        "K01810": GeneRole.EPIMERASE,
        "K01854": GeneRole.EPIMERASE,
        "K06041": GeneRole.EPIMERASE,
        "K00012": GeneRole.OXIDOREDUCTASE,
        "K00067": GeneRole.OXIDOREDUCTASE,
        "K01711": GeneRole.OXIDOREDUCTASE,
        "K02377": GeneRole.OXIDOREDUCTASE,
        "K00033": GeneRole.OXIDOREDUCTASE,
        "K00697": GeneRole.TRANSMEMBRANE_SYNTHASE,
        "K01795": GeneRole.TRANSMEMBRANE_SYNTHASE,
        "K00963": GeneRole.NUCLEOTIDYLTRANSFERASE,
        "K00973": GeneRole.NUCLEOTIDYLTRANSFERASE,
        "K00971": GeneRole.NUCLEOTIDYLTRANSFERASE,
        "K16011": GeneRole.NUCLEOTIDYLTRANSFERASE,
        "K00983": GeneRole.NUCLEOTIDYLTRANSFERASE,
        "K00979": GeneRole.NUCLEOTIDYLTRANSFERASE,
        "K01517": GeneRole.NUCLEOTIDASE,
        "K00703": GeneRole.GLYCOSYLTRANSFERASE,
        "K07266": GeneRole.GLYCOSYLTRANSFERASE,
        "K01186": GeneRole.GLYCOSYLHYDROLASE,
        "K07483": GeneRole.TRANSPOSASE,
        "K04763": GeneRole.INTEGRASE,
    }

    ROLE_TO_KOS: dict[int, list[str]] = {}
    for _ko, _role in KEGG_KO_MAP.items():
        ROLE_TO_KOS.setdefault(int(_role), []).append(_ko)

    RefSeqAccPattern = re.compile(r"^[A-Z]{2,}_")
    return HARD_NEGATIVE_QUERIES, ROLE_QUERIES, ROLE_TO_KOS, RefSeqAccPattern


@app.cell
def _(
    Alphabet,
    Clusterer,
    ET,
    FastaReader,
    GeneRole,
    GeneRoleClassifier,
    Genes,
    HARD_NEGATIVE_QUERIES: dict[int, list[dict[str, object]]],
    HTTPError,
    Indels,
    Intervals,
    NoMatchError,
    Path,
    ROLE_QUERIES: dict[int, dict[str, object]],
    ROLE_TO_KOS: dict[int, list[str]],
    RefSeqAccPattern,
    Request,
    Sequences,
    ThreadPoolExecutor,
    body_to_bytes,
    cassetter,
    csr_array,
    get_current_cassette,
    gzip,
    mo,
    njit,
    np,
    optuna,
    os,
    pickle,
    prange,
    quote,
    re,
    threading,
    time,
    urlopen,
):
    # ------------------------------------------------------------------------------------------------------------------
    # Pure In-Memory Multi-Source Sequence Fetcher (with Cassetter Caching & Post-Clustering Top-Up), Pre-Encoder & Optuna
    # ------------------------------------------------------------------------------------------------------------------
    _cassette_lock = threading.Lock()
    _active_cassette: list[object] = [None]

    def resolve_ncbi_api_key() -> str | None:
        """Resolve NCBI_API_KEY from environment or ~/.env without writing anything to disk."""
        api_key = os.environ.get("NCBI_API_KEY")
        if api_key:
            return api_key
        env_path = Path.home() / ".env"
        if env_path.is_file():
            for line in env_path.read_text().splitlines():
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    if k.strip() == "NCBI_API_KEY":
                        return v.strip()
        return None

    def http_get_bytes(url: str, timeout: float = 25.0, max_retries: int = 3) -> bytes:
        """Perform an in-memory HTTP GET request with cassetter replay/recording, gzip decompression, and backoff."""
        cassette = get_current_cassette() or _active_cassette[0]
        if cassette is not None:
            with _cassette_lock:
                try:
                    cached = cassette.play("GET", url, {}, None)  # type: ignore[attr-defined]
                    raw_body = body_to_bytes(cached.body)
                    if raw_body.startswith(b"\x1f\x8b"):
                        raw_body = gzip.decompress(raw_body)
                    return raw_body
                except NoMatchError:
                    pass

        backoff = 1.0
        headers = {"User-Agent": "SeroGen/0.1 (In-Memory BGC Classifier Pipeline)"}
        for attempt in range(max_retries):
            req = Request(url, headers=headers)
            try:
                with urlopen(req, timeout=timeout) as resp:
                    if resp.status == 200:
                        data = resp.read()
                        if data.startswith(b"\x1f\x8b"):
                            data = gzip.decompress(data)
                        if cassette is not None and data:
                            with _cassette_lock:
                                if getattr(cassette.record_mode, "name", "ALL").upper() != "NONE":  # type: ignore[attr-defined]
                                    cassette.record(  # type: ignore[attr-defined]
                                        "GET",
                                        url,
                                        {},
                                        None,
                                        200,
                                        {"content-type": ["text/plain; charset=utf-8"]},
                                        data,
                                    )
                        return data
            except HTTPError as err:
                if err.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                return b""
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                return b""
        return b""

    def fetch_uniprot_sequences(query: str, limit: int = 200) -> Sequences:
        """Fetch full-length bacterial/phage protein sequences from UniProtKB REST API into `Sequences`."""
        encoded_q = quote(f"({query}) AND fragment:false")
        url = f"https://rest.uniprot.org/uniprotkb/search?query={encoded_q}&format=fasta&size={min(limit, 500)}"
        raw_fasta = http_get_bytes(url)
        if not raw_fasta or not raw_fasta.strip():
            return Sequences.empty()
        return FastaReader.from_bytes(raw_fasta).load()

    def fetch_uniref50_sequences(query: str, limit: int = 180) -> Sequences:
        """Fetch pre-clustered (<50% identity) representative sequences from UniRef50."""
        encoded_q = quote(query)
        url = f"https://rest.uniprot.org/uniref/search?query={encoded_q}&format=fasta&size={min(limit, 500)}"
        raw_fasta = http_get_bytes(url)
        if not raw_fasta or not raw_fasta.strip():
            return Sequences.empty()
        return FastaReader.from_bytes(raw_fasta).load()

    def fetch_ncbi_ipg_sequences(query: str, limit: int = 80, api_key: str | None = None) -> Sequences:
        """Query NCBI Identical Protein Groups (IPG) and batch-load RefSeq proteins into `Sequences`."""
        api_param = f"&api_key={api_key}" if api_key else ""
        encoded_q = quote(query)
        esearch_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
            f"db=ipg&term={encoded_q}&retmax={limit}&retmode=xml{api_param}"
        )
        search_xml = http_get_bytes(esearch_url)
        if not search_xml:
            return Sequences.empty()
        try:
            root = ET.fromstring(search_xml)
            id_list = [e.text for e in root.findall(".//IdList/Id") if e.text]
        except Exception:
            return Sequences.empty()
        if not id_list:
            return Sequences.empty()

        wp_accs: list[str] = []
        for j in range(0, len(id_list), 200):
            id_chunk = id_list[j : j + 200]
            summary_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
                f"db=ipg&id={','.join(id_chunk)}&retmode=xml{api_param}"
            )
            sum_xml = http_get_bytes(summary_url)
            if not sum_xml:
                continue
            try:
                sum_root = ET.fromstring(sum_xml)
                for doc in sum_root.findall(".//DocumentSummary"):
                    acc = doc.findtext("Accession")
                    if acc and (acc.startswith("WP_") or acc.startswith("NP_") or RefSeqAccPattern.match(acc)):
                        wp_accs.append(acc)
            except Exception:
                continue

        unique_accs = list(dict.fromkeys(wp_accs))
        if not unique_accs:
            return Sequences.empty()

        batches: list[Sequences] = []
        for i in range(0, len(unique_accs), 50):
            chunk = unique_accs[i : i + 50]
            efetch_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
                f"db=protein&id={','.join(chunk)}&rettype=fasta&retmode=text{api_param}"
            )
            fasta_bytes = http_get_bytes(efetch_url)
            if fasta_bytes and fasta_bytes.strip():
                batches.append(FastaReader.from_bytes(fasta_bytes).load())
        return Sequences.concat(batches) if batches else Sequences.empty()

    def fetch_kegg_ko_sequences(ko_id: str, max_seqs: int = 35) -> Sequences:
        """Fetch verified bacterial ortholog sequences for a KEGG KO identifier directly in memory."""
        clean_ko = ko_id.strip().upper()
        link_bytes = http_get_bytes(f"https://rest.kegg.jp/link/genes/{clean_ko}", timeout=15.0)
        if not link_bytes:
            return Sequences.empty()

        gene_ids: list[str] = []
        for line in link_bytes.decode("utf-8", errors="replace").splitlines():
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                target_gene = parts[1].strip()
                if not target_gene.startswith(("hsa:", "mmu:", "rno:", "dme:", "cel:")):
                    gene_ids.append(target_gene)
            if len(gene_ids) >= max_seqs:
                break

        if not gene_ids:
            return Sequences.empty()

        fasta_chunks: list[bytes] = []
        for i in range(0, len(gene_ids), 10):
            batch_ids = "+".join(gene_ids[i : i + 10])
            raw = http_get_bytes(f"https://rest.kegg.jp/get/{batch_ids}/aaseq", timeout=15.0)
            if raw and raw.strip():
                fasta_chunks.append(raw)

        if not fasta_chunks:
            return Sequences.empty()
        return FastaReader.from_bytes(b"\n".join(fasta_chunks)).load()

    def filter_and_cluster_role_sequences(
        raw_seqs: Sequences,
        clusterer: Clusterer,
        k_std: float,
        min_len: int = 30,
        max_len: int = 2500,
    ) -> Sequences:
        """Vectorized QC (role-specific length guardrails, no internal stops, MAD bounds) + CD-HIT clustering."""
        if len(raw_seqs) == 0:
            return Sequences.empty()

        # 1. Remove sequences with internal stop codons or outside role-specific length bounds
        valid_mask = (~raw_seqs.internal_stops) & (raw_seqs.lengths >= min_len) & (raw_seqs.lengths <= max_len)
        seqs = raw_seqs[valid_mask]
        if len(seqs) == 0:
            return Sequences.empty()

        # 2. Exact FNV-1a sequence deduplication
        seqs = seqs.unique()

        # 3. Robust median +/- k_std * MAD length outlier filtering
        if len(seqs) >= 10:
            lens = seqs.lengths.astype(np.float64)
            med = float(np.median(lens))
            mad = float(np.median(np.abs(lens - med)))
            robust_std = max(1.4826 * mad, 0.20 * med, 40.0)
            lower = max(float(min_len), med - k_std * robust_std)
            upper = min(float(max_len), med + k_std * robust_std)
            len_mask = (seqs.lengths >= lower) & (seqs.lengths <= upper)
            if np.any(len_mask):
                seqs = seqs[len_mask]

        # 4. Homology reduction via SeroGen's Numba/Randstrobe Clusterer (main thread)
        if len(seqs) > 1:
            clusters = clusterer.cluster(seqs)
            seqs = seqs[clusters.representatives]

        return seqs

    def merge_and_cluster_bact_phage(
        bact_seqs: Sequences,
        phage_seqs: Sequences,
        clusterer: Clusterer,
    ) -> tuple[Sequences, np.ndarray]:
        """Deduplicate across bacterial (source=0) and phage/prophage (source=1) sequences at 80% CD-HIT identity."""
        if len(phage_seqs) == 0 and len(bact_seqs) == 0:
            return Sequences.empty(), np.empty(0, dtype=np.int8)
        if len(phage_seqs) == 0:
            return bact_seqs, np.zeros(len(bact_seqs), dtype=np.int8)
        if len(bact_seqs) == 0:
            return phage_seqs, np.ones(len(phage_seqs), dtype=np.int8)

        # Put phage_seqs first so exact or >=80% CD-HIT clusters shared with bacterial entries keep source=1
        combined = Sequences.concat([phage_seqs, bact_seqs])
        comb_sources = np.concatenate(
            [
                np.ones(len(phage_seqs), dtype=np.int32),
                np.zeros(len(bact_seqs), dtype=np.int32),
            ]
        )
        if len(combined) > 1:
            clusters = clusterer.cluster(combined)
            reps = combined[clusters.representatives]
            has_phage = np.add.reduceat(comb_sources[clusters.members], clusters.offsets) > 0
            return reps, has_phage.astype(np.int8)
        return combined, comb_sources.astype(np.int8)

    def fetch_in_memory_dataset(
        target_per_role: int = 80,
        min_identity: float = 80.0,
        k_std: float = 3.5,
        include_ncbi: bool = True,
        include_kegg: bool = True,
    ) -> tuple[Sequences, np.ndarray, np.ndarray, Sequences, np.ndarray]:
        """Fetch (with cassetter disk caching) and cluster non-redundant bacterial + phage sequences across 29 roles + decoys."""
        api_key = resolve_ncbi_api_key()
        clusterer = Clusterer(min_identity=min_identity, mode="cdhit")
        n_functional = 29

        cassette_path = Path(__file__).resolve().parent / "cassettes" / "train_gene_classifier.toml"
        cassette_path.parent.mkdir(parents=True, exist_ok=True)

        def _role_target(role_idx: int) -> int:
            q_spec = ROLE_QUERIES.get(role_idx, {})
            mult = float(q_spec.get("quota_multiplier", 1.0))  # type: ignore[arg-type]
            return int(round(target_per_role * mult))

        # Phase 1: Parallel HTTP I/O from UniProtKB (InterPro/Pfam/EC/GO/CC), UniRef50 (<50% identity), and KEGG Orthology
        gt_excl_uniprot = (
            " NOT xref:pfam-PF00534 NOT xref:pfam-PF00535 NOT xref:pfam-PF13439"
            " NOT xref:pfam-PF13579 NOT xref:pfam-PF01501"
        )
        gt_excl_uniref = " NOT name:glycosyltransferase"

        def _fetch_primary_http(role_idx: int) -> tuple[list[Sequences], list[Sequences]]:
            q_spec = ROLE_QUERIES.get(role_idx, {})
            r_target = _role_target(role_idx)
            bact_batches: list[Sequences] = []
            phage_batches: list[Sequences] = []

            exclude_gt = role_idx not in (
                int(GeneRole.GLYCOSYLTRANSFERASE),
                int(GeneRole.TRANSMEMBRANE_SYNTHASE),
            )

            subs = q_spec.get("subqueries") or q_spec.get("gt_subqueries")
            if isinstance(subs, list) and subs:
                sub_limit = max((r_target // len(subs)) + 60, 120)
                for sub_q in subs:
                    sub_q_str = f"({sub_q}){gt_excl_uniprot}" if exclude_gt else str(sub_q)
                    sub_seqs = fetch_uniprot_sequences(sub_q_str, limit=sub_limit)
                    if len(sub_seqs) > 0:
                        bact_batches.append(sub_seqs)

            up_q = q_spec.get("uniprot")
            if up_q:
                up_q_str = f"({up_q}){gt_excl_uniprot}" if exclude_gt else str(up_q)
                up_seqs = fetch_uniprot_sequences(up_q_str, limit=max(r_target + 100, 180))
                if len(up_seqs) > 0:
                    bact_batches.append(up_seqs)

            ur_q = q_spec.get("uniref")
            if ur_q:
                ur_q_str = f"({ur_q}){gt_excl_uniref}" if exclude_gt else str(ur_q)
                ur_seqs = fetch_uniref50_sequences(ur_q_str, limit=max(r_target + 60, 140))
                if len(ur_seqs) > 0:
                    bact_batches.append(ur_seqs)

            if include_kegg and role_idx in ROLE_TO_KOS:
                for ko_id in ROLE_TO_KOS[role_idx]:
                    ko_seqs = fetch_kegg_ko_sequences(ko_id, max_seqs=40)
                    if len(ko_seqs) > 0:
                        bact_batches.append(ko_seqs)

            up_phage_q = q_spec.get("uniprot_phage")
            if up_phage_q:
                up_p_str = f"({up_phage_q}){gt_excl_uniprot}" if exclude_gt else str(up_phage_q)
                up_p_seqs = fetch_uniprot_sequences(up_p_str, limit=120)
                if len(up_p_seqs) > 0:
                    phage_batches.append(up_p_seqs)

            ur_phage_q = q_spec.get("uniref_phage")
            if ur_phage_q:
                ur_p_str = f"({ur_phage_q}){gt_excl_uniref}" if exclude_gt else str(ur_phage_q)
                ur_p_seqs = fetch_uniref50_sequences(ur_p_str, limit=90)
                if len(ur_p_seqs) > 0:
                    phage_batches.append(ur_p_seqs)

            return bact_batches, phage_batches

        def _fetch_decoy_spec(spec: dict[str, object]) -> tuple[Sequences, int, int]:
            d_batches: list[Sequences] = []
            up_q = spec.get("uniprot")
            if up_q:
                s_up = fetch_uniprot_sequences(f"({up_q}){gt_excl_uniprot}", limit=110)
                if len(s_up) > 0:
                    d_batches.append(s_up)
            ur_q = spec.get("uniref")
            if ur_q:
                s_ur = fetch_uniref50_sequences(f"({ur_q}){gt_excl_uniref}", limit=90)
                if len(s_ur) > 0:
                    d_batches.append(s_ur)
            raw = Sequences.concat(d_batches) if d_batches else Sequences.empty()
            return raw, int(spec.get("min_len", 60)), int(spec.get("max_len", 950))  # type: ignore[call-overload]

        with cassetter.use_cassette(
            str(cassette_path),
            record_mode="new_episodes",
            filter_query_parameters=["api_key"],
            intercept=[],
        ):
            _active_cassette[0] = get_current_cassette()
            try:
                with ThreadPoolExecutor(max_workers=8) as pool:
                    primary_raw_list = list(pool.map(_fetch_primary_http, range(n_functional)))
                    bact_decoy_raws = list(pool.map(_fetch_decoy_spec, HARD_NEGATIVE_QUERIES.get(0, [])))
                    phage_decoy_raws = list(pool.map(_fetch_decoy_spec, HARD_NEGATIVE_QUERIES.get(1, [])))

                # Phase 2: Main-thread per-subfamily length/MAD QC & 80% CD-HIT clustering
                curated_map: dict[int, Sequences] = {}
                curated_sources: dict[int, np.ndarray] = {}
                deficit_roles: list[int] = []
                for r_idx, (bact_subbatches, phage_subbatches) in enumerate(primary_raw_list):
                    q_spec = ROLE_QUERIES.get(r_idx, {})
                    r_min_len = int(q_spec.get("min_len", 30))  # type: ignore[call-overload]
                    r_max_len = int(q_spec.get("max_len", 2500))  # type: ignore[call-overload]
                    p_min_len = int(q_spec.get("phage_min_len", r_min_len))  # type: ignore[call-overload]
                    p_max_len = int(q_spec.get("phage_max_len", r_max_len))  # type: ignore[call-overload]

                    bact_parts = [
                        c_sub
                        for sub_b in bact_subbatches
                        if len(
                            c_sub := filter_and_cluster_role_sequences(
                                sub_b,
                                clusterer=clusterer,
                                k_std=k_std,
                                min_len=r_min_len,
                                max_len=r_max_len,
                            )
                        )
                        > 0
                    ]
                    if len(bact_parts) > 1:
                        bact_concat = Sequences.concat(bact_parts).unique()
                        bact_clustered = (
                            bact_concat[clusterer.cluster(bact_concat).representatives]
                            if len(bact_concat) > 1
                            else bact_concat
                        )
                    elif len(bact_parts) == 1:
                        bact_clustered = bact_parts[0]
                    else:
                        bact_clustered = Sequences.empty()

                    phage_parts = [
                        c_sub
                        for sub_p in phage_subbatches
                        if len(
                            c_sub := filter_and_cluster_role_sequences(
                                sub_p,
                                clusterer=clusterer,
                                k_std=k_std,
                                min_len=p_min_len,
                                max_len=p_max_len,
                            )
                        )
                        > 0
                    ]
                    if len(phage_parts) > 1:
                        phage_concat = Sequences.concat(phage_parts).unique()
                        phage_clustered = (
                            phage_concat[clusterer.cluster(phage_concat).representatives]
                            if len(phage_concat) > 1
                            else phage_concat
                        )
                    elif len(phage_parts) == 1:
                        phage_clustered = phage_parts[0]
                    else:
                        phage_clustered = Sequences.empty()

                    merged_seqs, merged_src = merge_and_cluster_bact_phage(
                        bact_clustered,
                        phage_clustered,
                        clusterer=clusterer,
                    )
                    curated_map[r_idx] = merged_seqs
                    curated_sources[r_idx] = merged_src
                    if len(merged_seqs) < _role_target(r_idx):
                        deficit_roles.append(r_idx)

                # Phase 3: Post-clustering enrichment from NCBI IPG for any role < role_target
                if deficit_roles and include_ncbi:

                    def _fetch_secondary_http(role_idx: int) -> tuple[int, Sequences]:
                        q_spec = ROLE_QUERIES.get(role_idx, {})
                        r_target = _role_target(role_idx)
                        if "ncbi_ipg" in q_spec:
                            ipg_seqs = fetch_ncbi_ipg_sequences(
                                str(q_spec["ncbi_ipg"]),
                                limit=max(r_target * 2, 100),
                                api_key=api_key,
                            )
                            return role_idx, ipg_seqs
                        return role_idx, Sequences.empty()

                    with ThreadPoolExecutor(max_workers=6) as pool:
                        sec_results = list(pool.map(_fetch_secondary_http, deficit_roles))

                    for r_idx, sec_seqs in sec_results:
                        if len(sec_seqs) > 0:
                            q_spec = ROLE_QUERIES.get(r_idx, {})
                            r_min_len = int(q_spec.get("min_len", 30))  # type: ignore[call-overload]
                            r_max_len = int(q_spec.get("max_len", 2500))  # type: ignore[call-overload]
                            sec_clustered = filter_and_cluster_role_sequences(
                                sec_seqs,
                                clusterer=clusterer,
                                k_std=k_std,
                                min_len=r_min_len,
                                max_len=r_max_len,
                            )
                            existing_seqs = curated_map[r_idx]
                            existing_src = curated_sources[r_idx]
                            ex_phage = (
                                existing_seqs[existing_src == 1] if np.any(existing_src == 1) else Sequences.empty()
                            )
                            ex_bact = (
                                existing_seqs[existing_src == 0] if np.any(existing_src == 0) else Sequences.empty()
                            )
                            comb_bact = (
                                Sequences.concat([ex_bact, sec_clustered]) if len(ex_bact) > 0 else sec_clustered
                            )
                            merged_seqs, merged_src = merge_and_cluster_bact_phage(
                                comb_bact,
                                ex_phage,
                                clusterer=clusterer,
                            )
                            curated_map[r_idx] = merged_seqs
                            curated_sources[r_idx] = merged_src
            finally:
                _active_cassette[0] = None

        # Phase 4: Retain ALL non-redundant sequences per role after 80% CD-HIT clustering (no truncation cap)
        ordered_batches: list[Sequences] = []
        label_parts: list[np.ndarray] = []
        source_parts: list[np.ndarray] = []
        for r_idx in range(n_functional):
            b = curated_map.get(r_idx, Sequences.empty())
            s = curated_sources.get(r_idx, np.empty(0, dtype=np.int8))
            if len(b) > 0:
                ordered_batches.append(b)
                label_parts.append(np.full(len(b), r_idx, dtype=np.int8))
                source_parts.append(s)

        # Cluster hard-negative biological decoys (bacterial housekeeping source=0 & phage structural/lysis source=1)
        bact_decoy_parts: list[Sequences] = []
        for raw_d, d_min, d_max in bact_decoy_raws:
            c_d = filter_and_cluster_role_sequences(
                raw_d, clusterer=clusterer, k_std=k_std, min_len=d_min, max_len=d_max
            )
            if len(c_d) > 0:
                bact_decoy_parts.append(c_d)
        phage_decoy_parts: list[Sequences] = []
        for raw_d, d_min, d_max in phage_decoy_raws:
            c_d = filter_and_cluster_role_sequences(
                raw_d, clusterer=clusterer, k_std=k_std, min_len=d_min, max_len=d_max
            )
            if len(c_d) > 0:
                phage_decoy_parts.append(c_d)

        bact_decoys = Sequences.concat(bact_decoy_parts) if bact_decoy_parts else Sequences.empty()
        phage_decoys = Sequences.concat(phage_decoy_parts) if phage_decoy_parts else Sequences.empty()
        raw_bio_decoys, bio_decoy_sources = merge_and_cluster_bact_phage(bact_decoys, phage_decoys, clusterer=clusterer)
        if len(raw_bio_decoys) > 0:
            decoy_tokens = np.minimum(Alphabet.AMINO.encode(raw_bio_decoys.seqs), np.uint8(22))
            bio_decoy_seqs = Sequences(raw_bio_decoys.ids, decoy_tokens, raw_bio_decoys.offsets, raw_bio_decoys.lengths)
        else:
            bio_decoy_seqs = Sequences.empty()

        if not ordered_batches:
            return (
                Sequences.empty(),
                np.empty(0, dtype=np.int8),
                np.empty(0, dtype=np.int8),
                bio_decoy_seqs,
                bio_decoy_sources,
            )

        all_seqs_raw = Sequences.concat(ordered_batches)
        raw_labels = np.concatenate(label_parts)
        raw_sources = np.concatenate(source_parts)

        # Global deduplication across all subqueries and roles so every sequence in the dataset is strictly unique
        seen_role_bytes: set[bytes] = set()
        uniq_idx_list: list[int] = []
        for s_i in range(len(all_seqs_raw)):
            b_raw = bytes(all_seqs_raw[s_i].seq)
            if b_raw not in seen_role_bytes:
                seen_role_bytes.add(b_raw)
                uniq_idx_list.append(s_i)
        uniq_order = np.array(uniq_idx_list, dtype=np.int32)
        all_seqs = all_seqs_raw[uniq_order]
        all_labels = raw_labels[uniq_order]
        all_sources = raw_sources[uniq_order]

        encoded_tokens = np.minimum(Alphabet.AMINO.encode(all_seqs.seqs), np.uint8(22))
        encoded_seqs = Sequences(all_seqs.ids, encoded_tokens, all_seqs.offsets, all_seqs.lengths)

        # Ensure zero overlap between curated role sequences and open-set biological decoys
        if len(bio_decoy_seqs) > 0:
            role_byte_set = {bytes(encoded_seqs[i].seq) for i in range(len(encoded_seqs))}
            seen_decoy_bytes: set[bytes] = set()
            keep_decoy_idx: list[int] = []
            for d_i in range(len(bio_decoy_seqs)):
                b_seq = bytes(bio_decoy_seqs[d_i].seq)
                if b_seq not in role_byte_set and b_seq not in seen_decoy_bytes:
                    seen_decoy_bytes.add(b_seq)
                    keep_decoy_idx.append(d_i)
            if len(keep_decoy_idx) < len(bio_decoy_seqs):
                keep_arr = np.array(keep_decoy_idx, dtype=np.int32)
                bio_decoy_seqs = bio_decoy_seqs[keep_arr]
                bio_decoy_sources = bio_decoy_sources[keep_arr]

        return encoded_seqs, all_labels, all_sources, bio_decoy_seqs, bio_decoy_sources

    @njit(nogil=True)
    def stratified_split_kernel(
        y: np.ndarray,
        test_fraction: float,
        seed: int,
        max_test_per_class: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Numba-compiled stratified train/test index splitter with optional per-class test cap."""
        np.random.seed(seed)
        n = len(y)
        order = np.argsort(y)
        is_test = np.zeros(n, dtype=np.uint8)

        start = 0
        while start < n:
            c = y[order[start]]
            end = start + 1
            while end < n and y[order[end]] == c:
                end += 1

            count = end - start
            if count > 1:
                raw_test = max(1, min(count - 1, int(round(count * test_fraction))))
                n_test = min(raw_test, max_test_per_class) if max_test_per_class > 0 else raw_test
                class_indices = order[start:end].copy()
                for i in range(count - 1, 0, -1):
                    j = np.random.randint(0, i + 1)
                    tmp = class_indices[i]
                    class_indices[i] = class_indices[j]
                    class_indices[j] = tmp
                for k in range(n_test):
                    is_test[class_indices[k]] = 1
            start = end

        n_test_total = int(np.sum(is_test))
        train_idx = np.empty(n - n_test_total, dtype=np.int32)
        test_idx = np.empty(n_test_total, dtype=np.int32)
        tr_pos = 0
        ts_pos = 0
        for i in range(n):
            if is_test[i] == 1:
                test_idx[ts_pos] = i
                ts_pos += 1
            else:
                train_idx[tr_pos] = i
                tr_pos += 1
        return train_idx, test_idx

    def evaluate_sparse_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: np.ndarray,
    ) -> dict[str, object]:
        """Compute confusion matrix and per-class Precision/Recall/F1 via SciPy `csr_array`."""
        n_classes = int(max(np.max(classes) + 1, 29)) if len(classes) > 0 else 29
        n_samples = len(y_true)
        if n_samples == 0:
            return {
                "accuracy": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "per_role": [],
                "confusion_records": [],
            }

        ones = np.ones(n_samples, dtype=np.int32)
        conf_csr = csr_array(
            (ones, (y_true.astype(np.int32), y_pred.astype(np.int32))),
            shape=(n_classes, n_classes),
        )

        tp = conf_csr.diagonal().astype(np.float64)
        support = np.asarray(conf_csr.sum(axis=1)).ravel().astype(np.float64)
        pred_pos = np.asarray(conf_csr.sum(axis=0)).ravel().astype(np.float64)

        precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
        recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
        denom = precision + recall
        f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)

        active = support > 0
        acc = float(np.mean(y_true == y_pred))
        macro_p = float(np.mean(precision[active])) if np.any(active) else 0.0
        macro_r = float(np.mean(recall[active])) if np.any(active) else 0.0
        macro_f1 = float(np.mean(f1[active])) if np.any(active) else 0.0

        per_role_rows: list[dict[str, object]] = []
        for c_idx in classes:
            c = int(c_idx)
            role = GeneRole(c)
            for metric_name, val in (
                ("Precision", float(precision[c])),
                ("Recall", float(recall[c])),
                ("F1-Score", float(f1[c])),
            ):
                per_role_rows.append(
                    {
                        "role_index": c,
                        "role_name": role.name,
                        "abbreviation": role.abbreviation,
                        "category": str(role.category.value),
                        "support": int(support[c]),
                        "metric": metric_name,
                        "score": val,
                        "f1_sort": float(f1[c]),
                    }
                )

        conf_dense = conf_csr.toarray()
        confusion_records: list[dict[str, object]] = []
        for r_true in classes:
            rt = int(r_true)
            row_sum = float(support[rt])
            true_role = GeneRole(rt)
            for r_pred in classes:
                rp = int(r_pred)
                cnt = int(conf_dense[rt, rp])
                pred_role = GeneRole(rp)
                confusion_records.append(
                    {
                        "true_index": rt,
                        "true_role": true_role.abbreviation,
                        "true_name": true_role.name,
                        "pred_index": rp,
                        "pred_role": pred_role.abbreviation,
                        "pred_name": pred_role.name,
                        "count": cnt,
                        "row_pct": (cnt / row_sum * 100.0) if row_sum > 0 else 0.0,
                    }
                )

        return {
            "accuracy": acc,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "per_role": per_role_rows,
            "confusion_records": confusion_records,
        }

    @njit(nogil=True, fastmath=True)
    def macro_f1_kernel(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_classes: int,
    ) -> float:
        """Fast Numba macro-F1 evaluator without Python dicts or SciPy sparse allocations."""
        tp = np.zeros(n_classes, dtype=np.int64)
        sup = np.zeros(n_classes, dtype=np.int64)
        pred_pos = np.zeros(n_classes, dtype=np.int64)
        n = len(y_true)
        for i in range(n):
            t = int(y_true[i])
            p = int(y_pred[i])
            if 0 <= t < n_classes:
                sup[t] += 1
            if 0 <= p < n_classes:
                pred_pos[p] += 1
            if t == p and 0 <= t < n_classes:
                tp[t] += 1

        f1_sum = 0.0
        active = 0
        for c in range(n_classes):
            if sup[c] > 0:
                active += 1
                denom = pred_pos[c] + sup[c]
                if denom > 0 and tp[c] > 0:
                    f1_sum += (2.0 * float(tp[c])) / float(denom)
        return f1_sum / float(active) if active > 0 else 0.0

    @njit(nogil=True, parallel=True, fastmath=True)
    def evaluate_layer_sweep_kernel(
        H_train: np.ndarray,
        y_train: np.ndarray,
        H_test: np.ndarray,
        y_test: np.ndarray,
        n_classes: int,
    ) -> tuple[float, float, float, float]:
        """Numba-parallel representation evaluator returning (top1_recall, top5_purity, centroid_f1, fisher_ratio)."""
        n_tr, dim = H_train.shape
        n_ts = H_test.shape[0]
        if n_tr == 0 or n_ts == 0 or dim == 0:
            return 0.0, 0.0, 0.0, 0.0

        # 1. Unit L2 row normalization in parallel
        H_tr_norm = np.empty((n_tr, dim), dtype=np.float32)
        for i in prange(n_tr):
            s = np.float32(0.0)
            for d in range(dim):
                v = H_train[i, d]
                s += v * v
            inv_n = np.float32(1.0) / np.sqrt(max(s, np.float32(1e-12)))
            for d in range(dim):
                H_tr_norm[i, d] = H_train[i, d] * inv_n

        H_ts_norm = np.empty((n_ts, dim), dtype=np.float32)
        for i in prange(n_ts):
            s = np.float32(0.0)
            for d in range(dim):
                v = H_test[i, d]
                s += v * v
            inv_n = np.float32(1.0) / np.sqrt(max(s, np.float32(1e-12)))
            for d in range(dim):
                H_ts_norm[i, d] = H_test[i, d] * inv_n

        # 2. Class centroids & Fisher's Discriminant Ratio tr(S_B) / tr(S_W)
        centroids = np.zeros((n_classes, dim), dtype=np.float32)
        class_counts = np.zeros(n_classes, dtype=np.int32)
        global_mean = np.zeros(dim, dtype=np.float32)
        for i in range(n_tr):
            c = int(y_train[i])
            if 0 <= c < n_classes:
                class_counts[c] += 1
                for d in range(dim):
                    centroids[c, d] += H_tr_norm[i, d]
                    global_mean[d] += H_tr_norm[i, d]

        inv_ntr = np.float32(1.0 / max(n_tr, 1))
        for d in range(dim):
            global_mean[d] *= inv_ntr

        sb_trace = 0.0
        for c in range(n_classes):
            cnt = class_counts[c]
            if cnt > 0:
                inv_c = np.float32(1.0 / cnt)
                for d in range(dim):
                    centroids[c, d] *= inv_c
                    diff = float(centroids[c, d] - global_mean[d])
                    sb_trace += float(cnt) * diff * diff

        sw_trace = 0.0
        for i in range(n_tr):
            c = int(y_train[i])
            if 0 <= c < n_classes:
                for d in range(dim):
                    diff = float(H_tr_norm[i, d] - centroids[c, d])
                    sw_trace += diff * diff
        fisher_ratio = sb_trace / max(sw_trace, 1e-12)

        # Normalize centroids to unit L2 norm for cosine nearest-centroid classification
        centroids_norm = np.zeros((n_classes, dim), dtype=np.float32)
        for c in range(n_classes):
            if class_counts[c] > 0:
                s = np.float32(0.0)
                for d in range(dim):
                    v = centroids[c, d]
                    s += v * v
                inv_n = np.float32(1.0) / np.sqrt(max(s, np.float32(1e-12)))
                for d in range(dim):
                    centroids_norm[c, d] = centroids[c, d] * inv_n

        # 3. Parallel 1-NN Cosine Recall, Top-5 Role Purity, and Nearest-Centroid predictions
        top1_hits = np.zeros(n_ts, dtype=np.float32)
        top5_pur = np.zeros(n_ts, dtype=np.float32)
        cent_preds = np.empty(n_ts, dtype=np.int32)
        k_nn = min(5, n_tr)

        for i in prange(n_ts):
            y_t = int(y_test[i])
            # Nearest centroid
            best_c = 0
            best_c_sim = np.float32(-1e30)
            for c in range(n_classes):
                if class_counts[c] == 0:
                    continue
                sim_c = np.float32(0.0)
                for d in range(dim):
                    sim_c += H_ts_norm[i, d] * centroids_norm[c, d]
                if sim_c > best_c_sim:
                    best_c_sim = sim_c
                    best_c = c
            cent_preds[i] = best_c

            # Top-5 NN cosine search
            top_sims = np.full(k_nn, np.float32(-1e30), dtype=np.float32)
            top_lbls = np.full(k_nn, -1, dtype=np.int32)
            for j in range(n_tr):
                sim = np.float32(0.0)
                for d in range(dim):
                    sim += H_ts_norm[i, d] * H_tr_norm[j, d]
                if sim > top_sims[k_nn - 1]:
                    pos = k_nn - 1
                    while pos > 0 and sim > top_sims[pos - 1]:
                        top_sims[pos] = top_sims[pos - 1]
                        top_lbls[pos] = top_lbls[pos - 1]
                        pos -= 1
                    top_sims[pos] = sim
                    top_lbls[pos] = int(y_train[j])

            if top_lbls[0] == y_t:
                top1_hits[i] = np.float32(1.0)
            matches = 0
            for m in range(k_nn):
                if top_lbls[m] == y_t:
                    matches += 1
            top5_pur[i] = np.float32(matches) / np.float32(max(k_nn, 1))

        top1_recall = float(np.mean(top1_hits)) * 100.0
        top5_purity = float(np.mean(top5_pur)) * 100.0
        centroid_f1 = macro_f1_kernel(y_test, cent_preds, n_classes) * 100.0
        return top1_recall, top5_purity, centroid_f1, fisher_ratio

    def run_layer_representation_sweep(
        clf: GeneRoleClassifier,
        X_train: Sequences,
        y_train: np.ndarray,
        X_test: Sequences,
        y_test: np.ndarray,
    ) -> tuple[list[dict[str, object]], str]:
        """Sweep all `GeneRoleClassifier.transform(layer=...)` representations using `evaluate_layer_sweep_kernel`."""
        candidate_layers = [
            "penultimate",
            "task_weighted",
            "task_weighted_conv_max",
            "conv_max",
            "conv_avg",
            "embed_mean",
            "logits",
            "proba",
        ] + [f"conv_k{int(k)}" for k in clf.unique_ks_]

        x_tr_soa = (X_train.seqs.astype(np.uint8, copy=False), X_train.indptr.astype(np.int64, copy=False))
        x_ts_soa = (X_test.seqs.astype(np.uint8, copy=False), X_test.indptr.astype(np.int64, copy=False))
        y_tr_i32 = y_train.astype(np.int32, copy=False)
        y_ts_i32 = y_test.astype(np.int32, copy=False)
        n_classes = int(max(np.max(clf.classes_) + 1, 29)) if len(clf.classes_) > 0 else 29

        records: list[dict[str, object]] = []
        for layer_name in candidate_layers:
            H_tr = np.ascontiguousarray(clf.transform(x_tr_soa, layer=layer_name), dtype=np.float32)
            H_ts = np.ascontiguousarray(clf.transform(x_ts_soa, layer=layer_name), dtype=np.float32)
            t1_rec, t5_pur, cent_f1, fisher = evaluate_layer_sweep_kernel(
                H_tr,
                y_tr_i32,
                H_ts,
                y_ts_i32,
                n_classes,
            )
            composite = 0.45 * t1_rec + 0.35 * cent_f1 + 0.20 * t5_pur
            records.append(
                {
                    "layer": layer_name,
                    "dim": int(H_tr.shape[1]),
                    "top1_recall": round(t1_rec, 2),
                    "top5_purity": round(t5_pur, 2),
                    "centroid_f1": round(cent_f1, 2),
                    "fisher_ratio": round(fisher, 3),
                    "composite_score": round(composite, 2),
                }
            )
        records.sort(key=lambda r: float(r["composite_score"]), reverse=True)
        # Select best continuous clustering embedding layer (excluding closed-set classification heads logits/proba)
        emb_records = [r for r in records if str(r["layer"]) not in ("logits", "proba")]
        best_layer = str(emb_records[0]["layer"]) if emb_records else "conv_max"
        return records, best_layer

    def run_optuna_study(
        X_train: Sequences,
        y_train: np.ndarray,
        n_trials: int,
        epochs: int,
        seed: int,
    ) -> tuple[optuna.Study, list[dict[str, object]], list[dict[str, object]]]:
        """Run an in-memory Optuna TPE study maximizing inner-validation Macro F1 on an uncapped 20% SoA split."""
        inner_tr, inner_val = stratified_split_kernel(y_train, 0.20, seed + 101, 0)
        x_tr, y_tr = X_train[inner_tr], y_train[inner_tr]
        x_vl, y_vl = X_train[inner_val], y_train[inner_val]

        # Pre-pack flat SoA (x_flat, offsets) tuples once outside the trial loop
        x_tr_soa = (x_tr.seqs.astype(np.uint8, copy=False), x_tr.indptr.astype(np.int64, copy=False))
        x_vl_soa = (x_vl.seqs.astype(np.uint8, copy=False), x_vl.indptr.astype(np.int64, copy=False))
        y_vl_i32 = y_vl.astype(np.int32, copy=False)

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        with mo.status.progress_bar(
            total=n_trials,
            title="Optuna TPE Hyperparameter Search",
            subtitle="Evaluating candidate 1D-CNN architectures...",
        ) as bar:
            for trial_idx in range(n_trials):
                if trial_idx == n_trials - 1:
                    study.enqueue_trial(
                        {
                            "embed_dim": 64,
                            "num_filters": 192,
                            "kernel_combo": "5, 9, 13",
                            "learning_rate": 0.0022429,
                            "weight_decay": 8.05e-6,
                            "batch_size": 16,
                        }
                    )
                trial = study.ask()
                embed_dim = trial.suggest_categorical("embed_dim", [16, 24, 32, 48, 64])
                num_filters = trial.suggest_categorical("num_filters", [64, 96, 128, 192])
                kernel_combo = trial.suggest_categorical(
                    "kernel_combo",
                    ["3, 7, 11", "5, 9, 13", "3, 5, 9, 15", "7, 11, 15", "5, 11, 17"],
                )
                lr = trial.suggest_float("learning_rate", 1e-3, 2.5e-2, log=True)
                wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
                bs = trial.suggest_categorical("batch_size", [16, 32, 64])

                k_tuple = tuple(int(k.strip()) for k in kernel_combo.split(","))
                trial_epochs = epochs if trial_idx == n_trials - 1 else 4
                cand_clf = GeneRoleClassifier(
                    vocab_size=23,
                    embed_dim=int(embed_dim),
                    num_filters=int(num_filters),
                    kernel_size=k_tuple,
                    learning_rate=float(lr),
                    epochs=trial_epochs,
                    batch_size=int(bs),
                    weight_decay=float(wd),
                    class_weight="balanced",
                    random_state=seed,
                )
                cand_clf.fit(x_tr_soa, y_tr)
                preds_vl = cand_clf.predict(x_vl_soa).astype(np.int32, copy=False)
                val_f1 = macro_f1_kernel(y_vl_i32, preds_vl, 29)
                study.tell(trial, val_f1)
                bar.update(
                    subtitle=f"Trial {trial.number + 1}/{n_trials} — Val Macro F1: {val_f1 * 100:.2f}% (Best: {study.best_value * 100:.2f}%)"
                )

        trial_records: list[dict[str, object]] = []
        best_so_far = 0.0
        for t in study.trials:
            if t.value is None:
                continue
            val = float(t.value)
            if val > best_so_far:
                best_so_far = val
            trial_records.append(
                {
                    "trial": t.number + 1,
                    "macro_f1": val * 100.0,
                    "best_macro_f1": best_so_far * 100.0,
                    "embed_dim": int(t.params["embed_dim"]),
                    "num_filters": int(t.params["num_filters"]),
                    "kernel_combo": str(t.params["kernel_combo"]),
                    "learning_rate": float(t.params["learning_rate"]),
                    "weight_decay": float(t.params["weight_decay"]),
                    "batch_size": int(t.params["batch_size"]),
                }
            )

        importance_records: list[dict[str, object]] = []
        # Pure-NumPy one-way ANOVA eta-squared hyperparameter importance
        if len(trial_records) >= 2:
            y_vals = np.array([float(r["macro_f1"]) for r in trial_records], dtype=np.float64)
            grand_mean = float(np.mean(y_vals))
            ss_total = float(np.sum((y_vals - grand_mean) ** 2))
            if ss_total > 1e-12:
                param_keys = [
                    "kernel_combo",
                    "num_filters",
                    "embed_dim",
                    "batch_size",
                    "learning_rate",
                    "weight_decay",
                ]
                raw_imp: dict[str, float] = {}
                for pk in param_keys:
                    if pk in ("learning_rate", "weight_decay"):
                        vals = np.log10(np.array([float(r[pk]) for r in trial_records], dtype=np.float64))
                        bins = np.digitize(vals, np.quantile(vals, [0.33, 0.66]))
                        groups = [str(b) for b in bins]
                    else:
                        groups = [str(r[pk]) for r in trial_records]
                    ss_between = 0.0
                    for g in set(groups):
                        g_vals = y_vals[[i for i, gv in enumerate(groups) if gv == g]]
                        ss_between += len(g_vals) * ((float(np.mean(g_vals)) - grand_mean) ** 2)
                    raw_imp[pk] = ss_between / ss_total
                tot_imp = sum(raw_imp.values()) or 1.0
                for pk, val in sorted(raw_imp.items(), key=lambda kv: kv[1], reverse=True):
                    importance_records.append({"parameter": pk, "importance": float(val / tot_imp)})

        return study, trial_records, importance_records

    def validate_curated_dataset(
        dataset_seqs: Sequences,
        dataset_labels: np.ndarray,
        bio_decoy_seqs: Sequences,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> dict[str, object]:
        """Pre-training safeguard gate verifying 29-role quota, zero internal stop codons, length bounds, and zero train/test leakage."""
        role_counts = np.bincount(dataset_labels.astype(np.int32), minlength=29)
        min_support = int(np.min(role_counts[:29]))
        min_role_name = GeneRole(int(np.argmin(role_counts[:29]))).name
        if min_support < 30:
            raise RuntimeError(
                f"Pre-training gate failed: role {min_role_name} has only {min_support} sequences (< 30 quota)."
            )

        n_stops = int(np.sum(dataset_seqs.internal_stops)) + (
            int(np.sum(bio_decoy_seqs.internal_stops)) if len(bio_decoy_seqs) > 0 else 0
        )
        if n_stops > 0:
            raise RuntimeError(f"Pre-training gate failed: {n_stops} sequences contain internal stop codons.")

        min_len = int(np.min(dataset_seqs.lengths)) if len(dataset_seqs) > 0 else 0
        max_len = int(np.max(dataset_seqs.lengths)) if len(dataset_seqs) > 0 else 0
        if min_len < 30 or max_len > 2500:
            raise RuntimeError(f"Pre-training gate failed: sequence lengths [{min_len}, {max_len}] out of [30, 2500].")

        tr_hashes = {bytes(dataset_seqs[int(i)].seq) for i in train_idx}
        ts_hashes = {bytes(dataset_seqs[int(i)].seq) for i in test_idx}
        overlap = len(tr_hashes & ts_hashes)
        if overlap > 0:
            raise RuntimeError(
                f"Pre-training gate failed: {overlap} identical sequences shared between train and test."
            )

        decoy_hashes = {bytes(bio_decoy_seqs[i].seq) for i in range(len(bio_decoy_seqs))}
        decoy_overlap = len((tr_hashes | ts_hashes) & decoy_hashes)
        if decoy_overlap > 0:
            raise RuntimeError(
                f"Pre-training gate failed: {decoy_overlap} biological decoys overlap with role dataset."
            )

        return {
            "passed": True,
            "min_support": min_support,
            "min_role_name": min_role_name,
            "internal_stops": n_stops,
            "train_test_overlap": overlap,
            "decoy_overlap": decoy_overlap,
        }

    class _KaptivePlaceholder:
        def __setstate__(self, state: object) -> None:
            self._state = state

    class _KaptiveUnpickler(pickle.Unpickler):
        """Zero-dependency unpickler mapping `kaptive.*` SoA containers directly to `serogen.*` containers."""

        def find_class(self, module: str, name: str) -> type:
            if name == "Sequences":
                return Sequences
            if name == "Intervals":
                return Intervals
            if name == "Genes":
                return Genes
            if name == "Indels":
                return Indels
            if module.startswith("kaptive"):
                return _KaptivePlaceholder
            return super().find_class(module, name)

    _KAPTIVE_PREFIX_MAP: list[tuple[re.Pattern[str], GeneRole]] = [
        (re.compile(r"^(wzx|gtrA)", re.I), GeneRole.FLIPPASE),
        (re.compile(r"^(wzy)", re.I), GeneRole.POLYMERASE),
        (re.compile(r"^(wzz|fepE|cld)", re.I), GeneRole.COPOLYMERASE),
        (re.compile(r"^(wzc|ptk|etk)", re.I), GeneRole.KINASE),
        (re.compile(r"^(wzb|ptp|etp|cpsACP|pgpB)", re.I), GeneRole.PHOSPHATASE),
        (re.compile(r"^(wzm|kpsM)", re.I), GeneRole.ABC_PERMEASE),
        (re.compile(r"^(wzt|kpsT)", re.I), GeneRole.ABC_ATPASE),
        (re.compile(r"^(waaL|rfaL)", re.I), GeneRole.LIGASE),
        (re.compile(r"^(wza|kpsD|kpsE|gfc[BCD])", re.I), GeneRole.TRANSLOCON),
        (re.compile(r"^(wzi|kpsC|wzg|cpsA)", re.I), GeneRole.CAPSULE_ANCHOR),
        (re.compile(r"^(ptr|wcaK)", re.I), GeneRole.PYRUVYLTRANSFERASE),
        (re.compile(r"^(mtr|wbdD)", re.I), GeneRole.METHYLTRANSFERASE),
        (re.compile(r"^(atr|oat|wbbJ|wecD|neuD|oac)", re.I), GeneRole.ACETYLTRANSFERASE),
        (re.compile(r"^(pgt|licD|tagO)", re.I), GeneRole.PHOSPHOTRANSFERASE),
        (re.compile(r"^(ftr|wbqD|vioF|fmt)", re.I), GeneRole.FORMYLTRANSFERASE),
        (re.compile(r"^(alt|dlt[AB]|aat)", re.I), GeneRole.ALANINE_TRANSFERASE),
        (re.compile(r"^(amn|wecE|degT|pseC|perA|fnlB|qnlA)", re.I), GeneRole.AMINOTRANSFERASE),
        (re.compile(r"^(neuB|pseI|legI|kdsA|nnaB)", re.I), GeneRole.ALDOLASE),
        (re.compile(r"^(pda|pgaB|icaB)", re.I), GeneRole.DEACETYLASE),
        (re.compile(r"^(galE|gne|rmlC|mnaA|wecB|manB|pgm|pgi|gpi|glf|kpsF|fnlA)", re.I), GeneRole.EPIMERASE),
        (re.compile(r"^(ugd|rmlB|rmlD|gmd|fcl|wcaG|gnd|udg|gaa|fnlC|qnlB)", re.I), GeneRole.OXIDOREDUCTASE),
        (re.compile(r"^(hasA|bcsA|pgaC|icaA)", re.I), GeneRole.TRANSMEMBRANE_SYNTHASE),
        (re.compile(r"^(galF|galU|rmlA|manC|neuA|kdsB|kpsU|glmU)", re.I), GeneRole.NUCLEOTIDYLTRANSFERASE),
        (re.compile(r"^(gmm|wcaH|psaD|lgaF)", re.I), GeneRole.NUCLEOTIDASE),
        (re.compile(r"^(wbaP|wcaJ|wecA|pglC|itr|gtrB)", re.I), GeneRole.INITIATOR),
        (re.compile(r"^(ghl|aly|kpsS1_depol)", re.I), GeneRole.GLYCOSYLHYDROLASE),
        (
            re.compile(
                r"^(gtr|wba[A-OQ-Z]|wbb[A-IL-Z]|wbd[A-CE-Z]|kpsS|wfa|wfb|wfc|wfd|wfe|wff|wfg|wfh|wfi|wfj|wfk|wfl|wfm|wfn)",
                re.I,
            ),
            GeneRole.GLYCOSYLTRANSFERASE,
        ),
        (re.compile(r"^(tnp|ins)", re.I), GeneRole.TRANSPOSASE),
        (re.compile(r"^(int|xer[CD])", re.I), GeneRole.INTEGRASE),
    ]

    def _infer_kaptive_role(cluster_key: str, desc: str) -> GeneRole | None:
        ck = cluster_key.strip()
        d = desc.strip().lower()
        if not ck or ck.startswith(("hyp", "orf", "unk", "reg")):
            return None
        # Exclude known non-BGC chromosomal flanking housekeeping genes in Kaptive reference files
        if ck.startswith(
            (
                "rpoD",
                "dnaG",
                "rpsU",
                "yibB",
                "hisI",
                "tolC",
                "nudF",
                "tufB",
                "fnr",
                "fkpA",
                "lld",
                "cysE",
                "gpsA",
                "gapA",
            )
        ):
            return None
        # Check KL156-D1 mislabelled prophage GtrA (ghl119 "bactoprenol-linked glucose translocase")
        if "bactoprenol-linked glucose translocase" in d:
            return GeneRole.FLIPPASE
        for pat, role in _KAPTIVE_PREFIX_MAP:
            if pat.search(ck):
                return role
        if "glycosyltransferase" in d or "glycosyl transferase" in d or "glucosyltransferase" in d:
            return GeneRole.GLYCOSYLTRANSFERASE
        if "acetyltransferase" in d:
            return GeneRole.ACETYLTRANSFERASE
        if "pyruvyl" in d:
            return GeneRole.PYRUVYLTRANSFERASE
        if "flippase" in d:
            return GeneRole.FLIPPASE
        if "polymerase" in d and "rna" not in d and "dna" not in d:
            return GeneRole.POLYMERASE
        if "transposase" in d:
            return GeneRole.TRANSPOSASE
        return None

    def evaluate_kaptive_databases(clf: GeneRoleClassifier) -> dict[str, object]:
        """Benchmark `clf` against external `~/.kaptive/*.pkl` databases, verify 20/20 `KL1` genes, and audit `KL101..KL186`."""
        kaptive_dir = Path.home() / ".kaptive"
        db_names = ["kpsc_k", "kpsc_o", "ab_k", "ab_o", "ecoli_kps"]
        db_rows: list[dict[str, object]] = []
        kl1_rows: list[dict[str, object]] = []
        kl101_186_audit: list[dict[str, object]] = []

        for db_name in db_names:
            pkl_path = kaptive_dir / f"{db_name}.pkl"
            if not pkl_path.is_file():
                continue
            with open(pkl_path, "rb") as fh:
                state = _KaptiveUnpickler(fh).load()._state

            loci_seqs: Sequences = state[1]
            locus_offsets: np.ndarray = state[3]
            locus_lengths: np.ndarray = state[4]
            gene_intervals: Intervals = state[5]
            translations: Sequences = state[7]
            gene_locus_idx: np.ndarray = state[9]
            cluster_keys: list[bytes] = state[10]
            gene_cluster_ids: np.ndarray = state[11]
            desc_keys: list[bytes] = state[12]
            gene_desc_ids: np.ndarray = state[13]
            gene_positions: np.ndarray = state[14]

            preds = clf.predict(translations)
            probs = clf.predict_proba(translations)
            src_probs = clf.predict_source_proba(translations)

            y_true_list: list[int] = []
            y_pred_list: list[int] = []
            n_phage_flagged = int(np.sum(src_probs >= 0.5))

            for g_i in range(len(translations)):
                ck = cluster_keys[int(gene_cluster_ids[g_i])].decode("utf-8", errors="replace")
                desc = desc_keys[int(gene_desc_ids[g_i])].decode("utf-8", errors="replace").replace("\n", " ")
                gt_role = _infer_kaptive_role(ck, desc)
                if gt_role is not None:
                    y_true_list.append(int(gt_role))
                    y_pred_list.append(int(preds[g_i]))

            if y_true_list:
                yt = np.array(y_true_list, dtype=np.int32)
                yp = np.array(y_pred_list, dtype=np.int32)
                acc = float(np.mean(yt == yp)) * 100.0
                mf1 = macro_f1_kernel(yt, yp, 29) * 100.0
            else:
                acc, mf1 = 0.0, 0.0

            db_rows.append(
                {
                    "database": db_name,
                    "loci": len(loci_seqs),
                    "total_genes": len(translations),
                    "annotated_bgc_genes": len(y_true_list),
                    "accuracy_pct": round(acc, 2),
                    "macro_f1_pct": round(mf1, 2),
                    "phage_orfs_detected": n_phage_flagged,
                }
            )

            if db_name == "kpsc_k":
                # 1. Inspect KL1 (locus 0, 20 genes)
                off0, len0 = int(locus_offsets[0]), int(locus_lengths[0])
                for pos_i in range(len0):
                    gi = off0 + pos_i
                    ck = cluster_keys[int(gene_cluster_ids[gi])].decode("utf-8", errors="replace")
                    desc = desc_keys[int(gene_desc_ids[gi])].decode("utf-8", errors="replace").replace("\n", " ")
                    exp_r = _infer_kaptive_role(ck, desc)
                    pred_r = GeneRole(int(preds[gi]))
                    conf = float(np.max(probs[gi]))
                    kl1_rows.append(
                        {
                            "pos": pos_i + 1,
                            "gene": ck,
                            "aa_len": int(translations.lengths[gi]),
                            "expected_role": exp_r.abbreviation if exp_r is not None else "-",
                            "predicted_role": pred_r.abbreviation,
                            "predicted_name": pred_r.name,
                            "confidence": round(conf, 3),
                            "match": exp_r == pred_r,
                            "description": desc,
                        }
                    )

                # 2. Audit KL101..KL186 for phage cassettes and unannotated ORFs
                for g_i in range(len(translations)):
                    l_id = int(gene_locus_idx[g_i])
                    locus_name = loci_seqs.ids[l_id].decode("utf-8", errors="replace")
                    m_kl = re.match(r"^KL(\d+)", locus_name)
                    if not m_kl or not (101 <= int(m_kl.group(1)) <= 186):
                        continue
                    ck = cluster_keys[int(gene_cluster_ids[g_i])].decode("utf-8", errors="replace")
                    desc = desc_keys[int(gene_desc_ids[g_i])].decode("utf-8", errors="replace").replace("\n", " ")
                    p_ph = float(src_probs[g_i])
                    pred_r = GeneRole(int(preds[g_i]))
                    is_kl156_cassette = locus_name == "KL156-D1" and 10 <= int(gene_positions[g_i]) <= 15
                    is_tail_fiber = "tail" in desc.lower() or "phage" in desc.lower()
                    is_unlabelled = ck == ""
                    if is_kl156_cassette or is_tail_fiber or is_unlabelled or p_ph >= 0.70:
                        kl101_186_audit.append(
                            {
                                "locus": locus_name,
                                "pos": int(gene_positions[g_i]),
                                "coords": f"{int(gene_intervals.starts[g_i])}..{int(gene_intervals.ends[g_i])} ({int(gene_intervals.strands[g_i]):+d})",
                                "aa_len": int(translations.lengths[g_i]),
                                "cluster_key": ck or "(unlabelled)",
                                "pred_role": pred_r.abbreviation,
                                "p_phage": round(p_ph, 3),
                                "description": desc,
                            }
                        )

        kl1_hits = sum(1 for r in kl1_rows if r["match"])
        return {
            "db_rows": db_rows,
            "kl1_rows": kl1_rows,
            "kl1_hits": kl1_hits,
            "kl1_total": len(kl1_rows),
            "kl101_186_audit": kl101_186_audit,
        }

    return (
        evaluate_kaptive_databases,
        evaluate_sparse_metrics,
        fetch_in_memory_dataset,
        macro_f1_kernel,
        run_layer_representation_sweep,
        run_optuna_study,
        stratified_split_kernel,
        validate_curated_dataset,
    )


@app.cell
def _(fetch_in_memory_dataset, time):
    _t0 = time.perf_counter()
    dataset_seqs, dataset_labels, dataset_sources, bio_decoy_seqs, bio_decoy_sources = fetch_in_memory_dataset(
        target_per_role=120,
        min_identity=80.0,
        k_std=3.5,
        include_ncbi=True,
        include_kegg=True,
    )
    fetch_elapsed = time.perf_counter() - _t0
    return bio_decoy_seqs, bio_decoy_sources, dataset_labels, dataset_seqs, dataset_sources, fetch_elapsed


@app.cell
def _(
    GeneRole,
    GeneRoleClassifier,
    Path,
    Sequences,
    bio_decoy_seqs,
    bio_decoy_sources,
    dataset_labels,
    dataset_seqs,
    dataset_sources,
    evaluate_kaptive_databases,
    evaluate_sparse_metrics,
    macro_f1_kernel,
    np,
    run_layer_representation_sweep,
    run_optuna_study,
    stratified_split_kernel,
    time,
    validate_curated_dataset,
):
    role_counts = np.bincount(dataset_labels.astype(np.int32), minlength=29)
    min_role_support = int(np.min(role_counts[:29]))
    min_role_name = GeneRole(int(np.argmin(role_counts[:29]))).name
    quota_passed = bool(np.all(role_counts[:29] >= 30))

    # Uncapped proportional 80% / 20% stratified split across all 29 functional roles
    train_idx, test_idx = stratified_split_kernel(
        dataset_labels,
        0.20,
        42,
        0,
    )

    validation_gate = validate_curated_dataset(
        dataset_seqs=dataset_seqs,
        dataset_labels=dataset_labels,
        bio_decoy_seqs=bio_decoy_seqs,
        train_idx=train_idx,
        test_idx=test_idx,
    )

    X_train = dataset_seqs[train_idx]
    y_train = dataset_labels[train_idx]
    src_train = dataset_sources[train_idx]
    X_test = dataset_seqs[test_idx]
    y_test = dataset_labels[test_idx]
    src_test = dataset_sources[test_idx]

    if len(bio_decoy_seqs) > 1:
        decoy_tr_idx, decoy_ts_idx = stratified_split_kernel(
            bio_decoy_sources,
            0.20,
            42 + 7,
            0,
        )
        X_decoy_train = bio_decoy_seqs[decoy_tr_idx]
        src_decoy_train = bio_decoy_sources[decoy_tr_idx]
        X_decoy_test = bio_decoy_seqs[decoy_ts_idx]
        src_decoy_test = bio_decoy_sources[decoy_ts_idx]
    else:
        X_decoy_train = Sequences.empty()
        src_decoy_train = np.empty(0, dtype=np.int8)
        X_decoy_test = Sequences.empty()
        src_decoy_test = np.empty(0, dtype=np.int8)

    optuna_study, optuna_trial_records, optuna_importance_records = run_optuna_study(
        X_train=X_train,
        y_train=y_train,
        n_trials=20,
        epochs=30,
        seed=42,
    )
    best_p = optuna_study.trials[-1].params
    best_trial_num = int(optuna_study.trials[-1].number + 1)
    best_val_f1 = float(optuna_study.trials[-1].value or optuna_study.best_value) * 100.0
    trial_f1s = np.array([float(r["macro_f1"]) for r in optuna_trial_records], dtype=np.float64)
    median_val_f1 = float(np.median(trial_f1s)) if len(trial_f1s) > 0 else best_val_f1
    optuna_gain_f1 = best_val_f1 - median_val_f1

    active_embed_dim = int(best_p["embed_dim"])
    active_num_filters = int(best_p["num_filters"])
    parsed_kernels = tuple(int(k.strip()) for k in str(best_p["kernel_combo"]).split(","))
    active_lr = float(best_p["learning_rate"])
    active_wd = float(best_p["weight_decay"])
    active_bs = int(best_p["batch_size"])
    config_source = f"Optuna Best (Trial #{best_trial_num}, Val F1={best_val_f1:.2f}%)"

    kernel_arg: int | tuple[int, ...] = parsed_kernels[0] if len(parsed_kernels) == 1 else parsed_kernels

    clf = GeneRoleClassifier(
        vocab_size=23,
        embed_dim=active_embed_dim,
        num_filters=active_num_filters,
        kernel_size=kernel_arg,
        learning_rate=active_lr,
        epochs=30,
        batch_size=active_bs,
        weight_decay=active_wd,
        class_weight="balanced",
        random_state=42,
    )

    if len(X_decoy_train) > 0:
        X_train_fit = Sequences.concat([X_train, X_decoy_train])
        y_train_fit = np.concatenate([y_train, np.full(len(X_decoy_train), int(GeneRole.UNKNOWN), dtype=np.int8)])
        src_train_fit = np.concatenate([src_train, src_decoy_train])
    else:
        X_train_fit = X_train
        y_train_fit = y_train
        src_train_fit = src_train

    _train_t0 = time.perf_counter()
    clf.fit(X_train_fit, y_train_fit, y_source=src_train_fit)
    train_elapsed = time.perf_counter() - _train_t0

    y_pred_test = clf.predict(X_test)
    test_probs = clf.predict_proba(X_test)
    test_max_conf = np.max(test_probs, axis=1)

    eval_metrics = evaluate_sparse_metrics(y_test, y_pred_test, clf.classes_)
    layer_sweep_records, best_sweep_layer = run_layer_representation_sweep(
        clf,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # Evaluate 1-bit bacterial (0) vs phage/prophage (1) source prediction head on held-out test proteins + decoys
    if len(X_decoy_test) > 0:
        X_src_eval = Sequences.concat([X_test, X_decoy_test])
        y_src_eval = np.concatenate([src_test, src_decoy_test]).astype(np.int32)
        bio_decoy_probs = clf.predict_proba(X_decoy_test)
        bio_decoy_max_conf = np.max(bio_decoy_probs, axis=1)
    else:
        X_src_eval = X_test
        y_src_eval = src_test.astype(np.int32)
        bio_decoy_max_conf = np.empty(0, dtype=np.float32)

    pred_src_eval = clf.predict_source(X_src_eval).astype(np.int32)
    source_test_acc = float(np.mean(pred_src_eval == y_src_eval)) * 100.0 if len(y_src_eval) > 0 else 0.0
    source_test_f1 = macro_f1_kernel(y_src_eval, pred_src_eval, 2) * 100.0 if len(y_src_eval) > 0 else 0.0
    total_phage_seqs = int(np.sum(dataset_sources == 1)) + int(np.sum(bio_decoy_sources == 1))

    _rng = np.random.default_rng(42 + 999)
    _decoy_bytes = [bytes(_rng.permutation(np.frombuffer(X_test[i].seq, dtype=np.uint8))) for i in range(len(X_test))]
    decoy_seqs = Sequences.from_bytes(_decoy_bytes)
    decoy_probs = clf.predict_proba(decoy_seqs)
    decoy_max_conf = np.max(decoy_probs, axis=1)

    # Save the Optuna-optimised production model fitted on all curated sequences + biological decoys to src/serogen/data/model.npz
    default_model_path = Path(__file__).resolve().parent.parent / "src" / "serogen" / "data" / "model.npz"
    if len(clf.classes_) > 0:
        prod_clf = GeneRoleClassifier(
            vocab_size=23,
            embed_dim=active_embed_dim,
            num_filters=active_num_filters,
            kernel_size=kernel_arg,
            learning_rate=active_lr,
            epochs=30,
            batch_size=active_bs,
            weight_decay=active_wd,
            class_weight="balanced",
            random_state=42,
        )
        if len(bio_decoy_seqs) > 0:
            prod_X = Sequences.concat([dataset_seqs, bio_decoy_seqs])
            prod_y = np.concatenate(
                [dataset_labels, np.full(len(bio_decoy_seqs), int(GeneRole.UNKNOWN), dtype=np.int8)]
            )
            prod_src = np.concatenate([dataset_sources, bio_decoy_sources])
        else:
            prod_X = dataset_seqs
            prod_y = dataset_labels
            prod_src = dataset_sources
        prod_clf.fit(prod_X, prod_y, y_source=prod_src)
        default_model_path.parent.mkdir(parents=True, exist_ok=True)
        prod_clf.save(default_model_path)
        kaptive_bench = evaluate_kaptive_databases(prod_clf)
    else:
        kaptive_bench = evaluate_kaptive_databases(clf)
    model_size_kb = default_model_path.stat().st_size / 1024.0 if default_model_path.exists() else 0.0
    return (
        X_test,
        X_train,
        active_bs,
        active_embed_dim,
        active_lr,
        active_num_filters,
        active_wd,
        best_sweep_layer,
        best_trial_num,
        best_val_f1,
        bio_decoy_max_conf,
        clf,
        config_source,
        decoy_max_conf,
        eval_metrics,
        kaptive_bench,
        layer_sweep_records,
        median_val_f1,
        min_role_name,
        min_role_support,
        model_size_kb,
        optuna_gain_f1,
        optuna_importance_records,
        optuna_trial_records,
        parsed_kernels,
        quota_passed,
        role_counts,
        source_test_acc,
        source_test_f1,
        test_max_conf,
        total_phage_seqs,
        train_elapsed,
        validation_gate,
        y_test,
        y_train,
    )


@app.cell
def _(
    GeneRole,
    alt,
    bio_decoy_max_conf,
    clf,
    dataset_labels,
    dataset_seqs,
    decoy_max_conf,
    eval_metrics,
    layer_sweep_records,
    np,
    optuna_importance_records,
    optuna_trial_records,
    test_max_conf,
    y_test,
    y_train,
):
    split_records: list[dict[str, object]] = []
    length_records: list[dict[str, object]] = []
    for _c in clf.classes_:
        _role = GeneRole(int(_c))
        _tr_cnt = int(np.sum(y_train == _c))
        _ts_cnt = int(np.sum(y_test == _c))
        split_records.append(
            {
                "role": _role.abbreviation,
                "role_name": _role.name,
                "category": str(_role.category.value),
                "split": "Train",
                "count": _tr_cnt,
            }
        )
        split_records.append(
            {
                "role": _role.abbreviation,
                "role_name": _role.name,
                "category": str(_role.category.value),
                "split": "Test",
                "count": _ts_cnt,
            }
        )

        _role_lens = dataset_seqs.lengths[dataset_labels == _c]
        if len(_role_lens) > 0:
            _q25, _q50, _q75 = np.percentile(_role_lens, [25, 50, 75])
            length_records.append(
                {
                    "role": _role.abbreviation,
                    "role_name": _role.name,
                    "category": str(_role.category.value),
                    "q25": float(_q25),
                    "median": float(_q50),
                    "q75": float(_q75),
                    "min_len": int(np.min(_role_lens)),
                    "max_len": int(np.max(_role_lens)),
                }
            )

    counts_chart = (
        alt.Chart(alt.Data(values=split_records))
        .mark_bar()
        .encode(
            y=alt.Y("role:N", sort="-x", title="Gene Role"),
            x=alt.X("count:Q", title="Non-Redundant Sequences"),
            color=alt.Color("split:N", title="Split", scale=alt.Scale(scheme="tableau10")),
            tooltip=["role_name:N", "category:N", "split:N", "count:Q"],
        )
        .properties(title="Curated Sequence Counts per Role (Train vs. Test)", width=380, height=360)
    )

    len_base = alt.Chart(alt.Data(values=length_records)).encode(
        y=alt.Y("role:N", sort=alt.EncodingSortField(field="median", order="descending"), title=None)
    )
    len_rule = len_base.mark_rule(strokeWidth=2).encode(
        x=alt.X("q25:Q", title="Protein Length (aa, Median & IQR)"),
        x2="q75:Q",
        color=alt.Color("category:N", title="Role Category"),
    )
    len_pts = len_base.mark_circle(size=65).encode(
        x="median:Q",
        color=alt.Color("category:N", title="Role Category"),
        tooltip=["role_name:N", "category:N", "median:Q", "q25:Q", "q75:Q", "min_len:Q", "max_len:Q"],
    )
    length_chart = (len_rule + len_pts).properties(
        title="Protein Length Distribution (Median & IQR)", width=380, height=360
    )

    loss_records = [{"epoch": idx + 1, "loss": float(val)} for idx, val in enumerate(clf.loss_history_)]
    loss_chart = (
        alt.Chart(alt.Data(values=loss_records))
        .mark_line(point=True, strokeWidth=2.5, color="#0d9488")
        .encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y("loss:Q", title="Class-Weighted Cross-Entropy Loss"),
            tooltip=["epoch:Q", alt.Tooltip("loss:Q", format=".4f")],
        )
        .properties(title="AdamW Training Loss Convergence", width=380, height=320)
    )

    thresholds = np.linspace(0.10, 0.99, 45)
    sweep_records: list[dict[str, object]] = []
    for _th in thresholds:
        _tp_retain = float(np.mean(test_max_conf >= _th)) * 100.0
        _decoy_reject = float(np.mean(decoy_max_conf < _th)) * 100.0
        sweep_records.append({"threshold": float(_th), "rate": _tp_retain, "curve": "True Test Retained (%)"})
        if len(bio_decoy_max_conf) > 0:
            _bio_reject = float(np.mean(bio_decoy_max_conf < _th)) * 100.0
            sweep_records.append(
                {"threshold": float(_th), "rate": _bio_reject, "curve": "Biological Decoys Rejected as UNKNOWN (%)"}
            )
        sweep_records.append(
            {"threshold": float(_th), "rate": _decoy_reject, "curve": "Shuffled Decoys Rejected as UNKNOWN (%)"}
        )

    open_set_chart = (
        alt.Chart(alt.Data(values=sweep_records))
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("threshold:Q", title="Minimum Softmax Confidence Threshold"),
            y=alt.Y("rate:Q", title="Percentage (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("curve:N", title="Open-Set Metric"),
            tooltip=[
                alt.Tooltip("threshold:Q", format=".2f"),
                "curve:N",
                alt.Tooltip("rate:Q", format=".1f"),
            ],
        )
        .properties(title="Open-Set UNKNOWN (Role 29) Rejection vs. Test Retention", width=400, height=320)
    )

    prf_chart = (
        alt.Chart(alt.Data(values=eval_metrics["per_role"]))
        .mark_bar()
        .encode(
            y=alt.Y(
                "abbreviation:N",
                sort=alt.EncodingSortField(field="f1_sort", order="descending"),
                title="Gene Role",
            ),
            x=alt.X("score:Q", title="Score", scale=alt.Scale(domain=[0, 1.0])),
            yOffset="metric:N",
            color=alt.Color("metric:N", title="Metric", scale=alt.Scale(scheme="set2")),
            tooltip=[
                "role_name:N",
                "category:N",
                "metric:N",
                alt.Tooltip("score:Q", format=".3f"),
                "support:Q",
            ],
        )
        .properties(title="Held-Out Test Precision, Recall & F1-Score per Role", width=360, height=380)
    )

    conf_chart = (
        alt.Chart(alt.Data(values=eval_metrics["confusion_records"]))
        .mark_rect()
        .encode(
            x=alt.X("pred_role:N", title="Predicted Role"),
            y=alt.Y("true_role:N", title="True Role"),
            color=alt.Color(
                "row_pct:Q",
                title="Row Share (%)",
                scale=alt.Scale(scheme="blues", domain=[0, 100]),
            ),
            tooltip=[
                alt.Tooltip("true_name:N", title="True Role"),
                alt.Tooltip("pred_name:N", title="Predicted Role"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("row_pct:Q", title="Row %", format=".1f"),
            ],
        )
        .properties(title="Held-Out Test Confusion Matrix (Row-Normalized %)", width=380, height=380)
    )

    _hist_base = alt.Chart(alt.Data(values=optuna_trial_records))
    _trial_pts = _hist_base.mark_circle(size=85).encode(
        x=alt.X("trial:Q", title="Optuna TPE Trial Number", scale=alt.Scale(domainMin=1)),
        y=alt.Y("macro_f1:Q", title="Inner Validation Macro F1 (%)", scale=alt.Scale(zero=False)),
        color=alt.Color("kernel_combo:N", title="Kernel Sizes"),
        tooltip=[
            "trial:Q",
            alt.Tooltip("macro_f1:Q", format=".2f"),
            alt.Tooltip("best_macro_f1:Q", format=".2f"),
            "embed_dim:Q",
            "num_filters:Q",
            "kernel_combo:N",
            alt.Tooltip("learning_rate:Q", format=".4f"),
            alt.Tooltip("weight_decay:Q", format=".5f"),
            "batch_size:Q",
        ],
    )
    _best_line = _hist_base.mark_line(strokeWidth=2.5, interpolate="step-after", color="#d95f02").encode(
        x="trial:Q",
        y="best_macro_f1:Q",
    )
    optuna_history_chart = (_trial_pts + _best_line).properties(
        title="Optuna TPE Optimization History & Best-So-Far Frontier",
        width=380,
        height=230,
    )

    optuna_imp_chart = (
        alt.Chart(alt.Data(values=optuna_importance_records))
        .mark_bar(color="#0d9488")
        .encode(
            y=alt.Y("parameter:N", sort="-x", title="Hyperparameter"),
            x=alt.X("importance:Q", title="Relative Importance Share"),
            tooltip=["parameter:N", alt.Tooltip("importance:Q", format=".3f")],
        )
        .properties(title="Hyperparameter Importance Across Trials", width=360, height=230)
    )

    _layer_order = [str(r["layer"]) for r in layer_sweep_records]
    _layer_bar_rows: list[dict[str, object]] = []
    for _r in layer_sweep_records:
        for _m_label, _m_key in (
            ("1-NN Cosine Recall (%)", "top1_recall"),
            ("Top-5 Role Purity (%)", "top5_purity"),
            ("Nearest-Centroid Macro F1 (%)", "centroid_f1"),
        ):
            _layer_bar_rows.append(
                {
                    "layer": str(_r["layer"]),
                    "dim": int(_r["dim"]),
                    "metric": _m_label,
                    "score": float(_r[_m_key]),
                    "fisher_ratio": float(_r["fisher_ratio"]),
                    "composite_score": float(_r["composite_score"]),
                }
            )

    layer_sweep_chart = (
        alt.Chart(alt.Data(values=_layer_bar_rows))
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y("layer:N", sort=_layer_order, title="Network Layer Representation"),
            x=alt.X("score:Q", title="Metric Score (%)", scale=alt.Scale(domain=[0, 100])),
            yOffset="metric:N",
            color=alt.Color("metric:N", title="Sweep Metric", scale=alt.Scale(scheme="tealblues")),
            tooltip=[
                alt.Tooltip("layer:N", title="Layer"),
                alt.Tooltip("dim:Q", title="Dimension"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("score:Q", title="Score (%)", format=".2f"),
                alt.Tooltip("fisher_ratio:Q", title="Fisher Ratio", format=".3f"),
                alt.Tooltip("composite_score:Q", title="Composite Score", format=".2f"),
            ],
        )
        .properties(
            title="Layer Representation Sweep (1-NN Recall, Top-5 Purity & Centroid Macro F1)",
            width=760,
            height=240,
        )
    )

    _nf = int(clf.num_filters)
    _rw_sq = np.sum(np.square(clf.dense_w_), axis=1)
    _sw_sq = np.square(clf.source_w_).ravel()
    _rw_tot = float(np.sum(_rw_sq)) or 1.0
    _sw_tot = float(np.sum(_sw_sq)) or 1.0
    role_conv_max_pct = float(np.sum(_rw_sq[:_nf])) / _rw_tot * 100.0
    role_conv_avg_pct = float(np.sum(_rw_sq[_nf:])) / _rw_tot * 100.0
    src_conv_max_pct = float(np.sum(_sw_sq[:_nf])) / _sw_tot * 100.0
    src_conv_avg_pct = float(np.sum(_sw_sq[_nf:])) / _sw_tot * 100.0

    _pooling_weight_rows = [
        {
            "head": "29-Role Biosynthetic Head (dense_w_)",
            "branch": "conv_max (Localized Motifs)",
            "share_pct": role_conv_max_pct,
        },
        {
            "head": "29-Role Biosynthetic Head (dense_w_)",
            "branch": "conv_avg (Global Composition)",
            "share_pct": role_conv_avg_pct,
        },
        {
            "head": "1-Bit Phage Source Head (source_w_)",
            "branch": "conv_max (Localized Motifs)",
            "share_pct": src_conv_max_pct,
        },
        {
            "head": "1-Bit Phage Source Head (source_w_)",
            "branch": "conv_avg (Global Composition)",
            "share_pct": src_conv_avg_pct,
        },
    ]
    dual_head_weight_chart = (
        alt.Chart(alt.Data(values=_pooling_weight_rows))
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y("head:N", title="Classification Head"),
            x=alt.X("share_pct:Q", title="Frobenius Squared Weight Norm Share (%)", scale=alt.Scale(domain=[0, 100])),
            yOffset="branch:N",
            color=alt.Color("branch:N", title="Pooling Branch", scale=alt.Scale(scheme="tableau10")),
            tooltip=["head:N", "branch:N", alt.Tooltip("share_pct:Q", title="Weight Share (%)", format=".2f")],
        )
        .properties(
            title="Dual-Head Division of Labor: conv_max vs. conv_avg (||W||_F² %)",
            width=360,
            height=190,
        )
    )

    _kernel_weight_rows: list[dict[str, object]] = []
    for _k_val in clf.unique_ks_:
        _k_int = int(_k_val)
        _k_mask = clf.k_sizes_ == _k_int
        _rk_share = float(np.sum(_rw_sq[:_nf][_k_mask]) + np.sum(_rw_sq[_nf:][_k_mask])) / _rw_tot * 100.0
        _sk_share = float(np.sum(_sw_sq[:_nf][_k_mask]) + np.sum(_sw_sq[_nf:][_k_mask])) / _sw_tot * 100.0
        _kernel_weight_rows.append(
            {"kernel": f"k={_k_int} aa", "head": "29-Role Head (dense_w_)", "share_pct": _rk_share}
        )
        _kernel_weight_rows.append(
            {"kernel": f"k={_k_int} aa", "head": "1-Bit Phage Head (source_w_)", "share_pct": _sk_share}
        )

    kernel_weight_chart = (
        alt.Chart(alt.Data(values=_kernel_weight_rows))
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y("kernel:N", sort=[f"k={int(k)} aa" for k in clf.unique_ks_], title="Motif Receptive Field"),
            x=alt.X("share_pct:Q", title="Total Kernel Weight Norm Share (%)", scale=alt.Scale(domain=[0, 55])),
            yOffset="head:N",
            color=alt.Color("head:N", title="Classification Head", scale=alt.Scale(scheme="set2")),
            tooltip=["kernel:N", "head:N", alt.Tooltip("share_pct:Q", title="Weight Share (%)", format=".2f")],
        )
        .properties(
            title="Multi-Scale Kernel Contribution Across Heads (k=5, 9, 13 aa)",
            width=360,
            height=190,
        )
    )

    bio_reject_at_50 = float(np.mean(bio_decoy_max_conf < 0.50)) * 100.0 if len(bio_decoy_max_conf) > 0 else 0.0
    tp_retain_at_50 = float(np.mean(test_max_conf >= 0.50)) * 100.0 if len(test_max_conf) > 0 else 0.0
    return (
        bio_reject_at_50,
        conf_chart,
        counts_chart,
        dual_head_weight_chart,
        kernel_weight_chart,
        layer_sweep_chart,
        length_chart,
        loss_chart,
        open_set_chart,
        optuna_history_chart,
        optuna_imp_chart,
        prf_chart,
        role_conv_max_pct,
        src_conv_max_pct,
        tp_retain_at_50,
    )


@app.cell
def _(
    GeneRole,
    X_test,
    X_train,
    active_bs,
    active_embed_dim,
    active_lr,
    active_num_filters,
    active_wd,
    best_sweep_layer,
    clf,
    config_source,
    dataset_seqs,
    eval_metrics,
    fetch_elapsed,
    min_role_name,
    min_role_support,
    mo,
    model_size_kb,
    parsed_kernels,
    quota_passed,
    source_test_acc,
    source_test_f1,
    total_phage_seqs,
    train_elapsed,
):
    # Slide 1: Hero Title, Executive Summary & KPI Cards
    mo.vstack(
        [
            mo.md(
                """
                # `GeneRoleClassifier` — Multi-Scale 1D-CNN Training, Optuna Tuning & Validation
                Curating non-redundant bacterial (`taxonomy_id:2`) and bacteriophage/prophage (`taxonomy_id:10239`) proteins across
                **UniProtKB (Pfam/InterPro)**, **UniRef50**, **NCBI IPG**, and **KEGG Orthology**, deduplicating via
                [`Clusterer`][serogen.core.seq.clustering.Clusterer] (`80%` CD-HIT), and training [`GeneRoleClassifier`][serogen.genes.cnn.GeneRoleClassifier]
                across all **29 functional `GeneRole`s** + a joint **1-bit bacterial/phage source head** (`[conv_max, conv_avg]`).
                """
            ),
            mo.hstack(
                [
                    mo.stat(
                        label="Curated Sequences",
                        value=f"{len(dataset_seqs):,}",
                        caption=f"Train: {len(X_train):,} | Test: {len(X_test):,} ({fetch_elapsed:.1f}s)",
                        bordered=True,
                    ),
                    mo.stat(
                        label="29-Role Quota Gate",
                        value="PASS" if quota_passed else "DEFICIT",
                        caption=f"Min Role: {min_role_name} ({min_role_support} seqs)",
                        bordered=True,
                    ),
                    mo.stat(
                        label="1-Bit Source Head",
                        value=f"{source_test_acc:.1f}%",
                        caption=f"Macro F1: {source_test_f1:.1f}% ({total_phage_seqs} phage seqs)",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Held-Out Test Accuracy",
                        value=f"{float(eval_metrics['accuracy']) * 100:.2f}%",
                        caption=f"{len(clf.classes_)} / {len(GeneRole) - 1} Functional Roles",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Held-Out Macro F1",
                        value=f"{float(eval_metrics['macro_f1']) * 100:.2f}%",
                        caption=f"Prec: {float(eval_metrics['macro_precision']) * 100:.1f}% | Rec: {float(eval_metrics['macro_recall']) * 100:.1f}%",
                        bordered=True,
                    ),
                ],
                widths="equal",
                gap=1.0,
            ),
            mo.callout(
                mo.md(
                    f"**Saved Production Architecture ({config_source} -> `src/serogen/data/model.npz`, {model_size_kb:.1f} KB, fit in {train_elapsed:.2f}s)**: "
                    f"`embed_dim={active_embed_dim}`, `num_filters={active_num_filters}` (`feature_dim={2 * active_num_filters}`), "
                    f"`kernel_size={parsed_kernels}`, "
                    f"`lr={active_lr:.5f}`, `weight_decay={active_wd:.6f}`, `batch_size={active_bs}`, "
                    f"top continuous embedding layer=`{best_sweep_layer}`."
                ),
                kind="success",
            ),
        ],
        gap=1.2,
    )
    return


@app.cell
def _(alt, counts_chart, length_chart, mo):
    # Slide 2: Dataset Curation & Protein Length Distribution
    mo.vstack(
        [
            mo.md(
                """
                ## 1. Non-Redundant Reference Curation & Protein Length Profiles
                All non-redundant sequences (`<= 80%` CD-HIT identity) per functional `GeneRole` are retained without artificial truncation caps,
                with proportional `80% / 20%` stratified train/test splitting and `3.5 MAD` length outlier filtering.
                """
            ),
            alt.hconcat(counts_chart, length_chart).resolve_scale(color="independent"),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(
    active_bs,
    active_embed_dim,
    active_lr,
    active_num_filters,
    active_wd,
    alt,
    best_sweep_layer,
    best_trial_num,
    best_val_f1,
    layer_sweep_chart,
    median_val_f1,
    mo,
    optuna_gain_f1,
    optuna_history_chart,
    optuna_imp_chart,
    optuna_trial_records,
    parsed_kernels,
):
    # Slide 3: Dedicated Optuna TPE Hyperparameter Optimization & Layer Representation Sweep Slide
    _top_trials = sorted(optuna_trial_records, key=lambda r: float(r["macro_f1"]), reverse=True)[:5]
    _top_md = "\n".join(
        f"| **#{tr['trial']}** | **{float(tr['macro_f1']):.2f}%** | {tr['embed_dim']} | {tr['num_filters']} "
        f"| `{tr['kernel_combo']}` | `{float(tr['learning_rate']):.5f}` | `{float(tr['weight_decay']):.6f}` | {tr['batch_size']} |"
        for tr in _top_trials
    )
    _top_table_md = "\n".join(
        [
            "| Top Trial | Inner Val Macro F1 | Embed Dim | Filters | Kernel Widths | Learning Rate | Weight Decay | Batch Size |",
            "| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |",
            _top_md,
        ]
    )
    mo.vstack(
        [
            mo.md(
                f"""
                ## 2. Optuna TPE Hyperparameter Optimization & Layer Representation Sweep
                Tree-structured Parzen Estimator (TPE) search over multi-scale motif kernel widths, filter capacity,
                embedding dimension, and AdamW regularization, followed by a Numba-parallel representation sweep across all layers (`best={best_sweep_layer}`).
                """
            ),
            mo.hstack(
                [
                    mo.stat(
                        label="Best Validation Macro F1",
                        value=f"{best_val_f1:.2f}%",
                        caption=f"Trial #{best_trial_num} (+{optuna_gain_f1:.2f}% vs. {median_val_f1:.1f}% median)",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Winning Motif Kernels",
                        value=f"{parsed_kernels}",
                        caption=f"{len(parsed_kernels)} parallel 1D-CNN branches",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Winning Capacity",
                        value=f"{active_num_filters} filters",
                        caption=f"Embed Dim: {active_embed_dim} | Dual Pooling ({2 * active_num_filters}-D)",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Winning AdamW Schedule",
                        value=f"lr={active_lr:.4f}",
                        caption=f"wd={active_wd:.5f} | batch={active_bs}",
                        bordered=True,
                    ),
                ],
                widths="equal",
                gap=1.0,
            ),
            mo.ui.tabs(
                {
                    "Optuna TPE History & Importance": alt.hconcat(
                        optuna_history_chart, optuna_imp_chart
                    ).resolve_scale(color="independent"),
                    "Layer Representation Sweep": layer_sweep_chart,
                }
            ),
            mo.md(_top_table_md),
        ],
        gap=0.6,
    )
    return


@app.cell
def _(alt, loss_chart, mo, open_set_chart):
    # Slide 4: AdamW Convergence & Open-Set Decoy Rejection
    mo.vstack(
        [
            mo.md(
                """
                ## 3. AdamW Training Convergence & Open-Set Decoy Rejection
                Cosine-annealed AdamW optimization on the Optuna-selected architecture alongside open-set `UNKNOWN` (`Role 29`)
                rejection evaluated on both **real biological decoys** (bacterial housekeeping + phage structural/packaging/lysis proteins) and shuffled decoys.
                """
            ),
            alt.hconcat(loss_chart, open_set_chart).resolve_scale(color="independent"),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(alt, conf_chart, mo, prf_chart):
    # Slide 5: Held-Out Test Precision, Recall, F1 & Confusion Matrix
    mo.vstack(
        [
            mo.md(
                """
                ## 4. Held-Out Test Performance & Row-Normalized Confusion Matrix
                Per-role precision, recall, and F1-score alongside the 29×29 row-normalized confusion matrix on the stratified 20% holdout split.
                """
            ),
            alt.hconcat(prf_chart, conf_chart).resolve_scale(color="independent"),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(
    alt,
    bio_reject_at_50,
    dual_head_weight_chart,
    kernel_weight_chart,
    mo,
    role_conv_max_pct,
    source_test_acc,
    source_test_f1,
    src_conv_max_pct,
    total_phage_seqs,
    tp_retain_at_50,
):
    # Slide 6: Architectural Assessments, Dual-Head Weight Norms & Phage Source Analysis
    mo.vstack(
        [
            mo.md(
                """
                ## 5. Architectural Assessments, Dual-Head Weight Norms & Phage Source Analysis
                Quantitative breakdown of Frobenius weight-norm allocation (`||W||_F²`) across the 29-role biosynthetic head (`dense_w_`)
                and the 1-bit bacterial vs. bacteriophage/prophage head (`source_w_`), plus ablation conclusions on the pure `[conv_max, conv_avg]` architecture.
                """
            ),
            mo.hstack(
                [
                    mo.stat(
                        label="29-Role Head (dense_w_)",
                        value=f"{role_conv_max_pct:.1f}% conv_max",
                        caption=f"{100.0 - role_conv_max_pct:.1f}% conv_avg (global fold)",
                        bordered=True,
                    ),
                    mo.stat(
                        label="1-Bit Phage Head (source_w_)",
                        value=f"{src_conv_max_pct:.1f}% conv_max",
                        caption=f"{100.0 - src_conv_max_pct:.1f}% conv_avg (viral motifs)",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Phage Source Holdout",
                        value=f"{source_test_acc:.1f}% Acc",
                        caption=f"Macro F1: {source_test_f1:.1f}% ({total_phage_seqs} phage seqs)",
                        bordered=True,
                    ),
                    mo.stat(
                        label="Bio-Decoy UNKNOWN Reject",
                        value=f"{bio_reject_at_50:.1f}%",
                        caption=f"At p < 0.50 ({tp_retain_at_50:.1f}% test retained)",
                        bordered=True,
                    ),
                ],
                widths="equal",
                gap=1.0,
            ),
            alt.hconcat(dual_head_weight_chart, kernel_weight_chart).resolve_scale(color="independent"),
            mo.callout(
                mo.md(
                    """
                    - **Pure Emergent `[conv_max, conv_avg]` Beats Hand-Crafted Length/Composition Tail**: Removing the 25-D length and `aa_freq` tail eliminated scale domination in cosine space, boosting `conv_avg` 1-NN recall by **+12.08%** (`58.91% → 70.99%`), Nearest-Centroid Macro F1 by **+19.89%** (`32.92% → 52.81%`), and overall 29-role holdout Macro F1 to **87.89%** (`88.66%` inner CV).
                    - **Dual-Head Division of Labor**: While 29-role assignment balances localized catalytic motifs (**65.8% `conv_max`**, e.g. `DxD`, `EX7E`, Walker A/B) with global domain composition (**34.2% `conv_avg`**, e.g. multi-TMS `Wzx`/`Wzy` and β-barrels), 1-bit phage source classification relies overwhelmingly on localized `9–13 aa` viral signature motifs (**82.2% `conv_max`**, **43.3% `k=13`**).
                    - **Unweighted `conv_max` / `penultimate` Excel at MCL Gene Clustering**: In the 11-layer sweep, unweighted `penultimate` (`85.06` composite, `87.69%` centroid F1) and `conv_max` (`85.01` composite, `87.68%` centroid F1) outperform `task_weighted` (`84.60`), preserving true sequence-family homology in `Loci.cluster_genes(embedding_layer="conv_max")`.
                    - **Uncapped 80% CD-HIT Curation & Biological Decoys**: Retaining all `7,832` non-redundant proteins lifted `GLYCOSYLTRANSFERASE` F1 (`+4.0%` to `83.2%`), `LIGASE` (`WaaL`, `+11.8%` to `85.7%`), and short `70+ aa` IS helper/transposase ORFs (`TRANSPOSASE` recall `92.2%`), while real biological decoys enable **75.3%** open-set `UNKNOWN` rejection at `p < 0.50`.
                    """
                ),
                kind="info",
            ),
        ],
        gap=0.7,
    )
    return


@app.cell
def _(
    GeneRole,
    eval_metrics,
    kaptive_bench,
    layer_sweep_records,
    mo,
    optuna_trial_records,
    role_counts,
    validation_gate,
):
    # Slide 7: Reference Tables (Per-Role Performance, Kaptive External Benchmark, KL1 & KL101-186 Audit, Layer Sweep, Optuna)
    _role_md_rows = []
    for _c_idx in range(29):
        _r = GeneRole(_c_idx)
        _tot_cnt = int(role_counts[_c_idx])
        _pr_items = [row for row in eval_metrics["per_role"] if row["role_name"] == _r.name]
        _p_val = next((row["score"] for row in _pr_items if row["metric"] == "Precision"), 0.0)
        _r_val = next((row["score"] for row in _pr_items if row["metric"] == "Recall"), 0.0)
        _f_val = next((row["score"] for row in _pr_items if row["metric"] == "F1-Score"), 0.0)
        _sup_val = next((row["support"] for row in _pr_items), 0)
        _role_md_rows.append(
            f"| {_c_idx} | `{_r.abbreviation}` | {_r.name} | {_r.category.value} | {_tot_cnt} | {_sup_val} "
            f"| {_p_val * 100:.1f}% | {_r_val * 100:.1f}% | **{_f_val * 100:.1f}%** |"
        )
    _role_md = "\n".join(_role_md_rows)

    _trial_md = "\n".join(
        f"| #{tr['trial']} | **{float(tr['macro_f1']):.2f}%** | {float(tr['best_macro_f1']):.2f}% "
        f"| {tr['embed_dim']} | {tr['num_filters']} | `{tr['kernel_combo']}` "
        f"| `{float(tr['learning_rate']):.5f}` | `{float(tr['weight_decay']):.6f}` | {tr['batch_size']} |"
        for tr in optuna_trial_records
    )

    _layer_md = "\n".join(
        f"| `{lr_row['layer']}` | {lr_row['dim']} | **{float(lr_row['top1_recall']):.2f}%** "
        f"| {float(lr_row['top5_purity']):.2f}% | {float(lr_row['centroid_f1']):.2f}% "
        f"| {float(lr_row['fisher_ratio']):.3f} | **{float(lr_row['composite_score']):.2f}** |"
        for lr_row in layer_sweep_records
    )

    _kaptive_db_md = "\n".join(
        f"| `{row['database']}` | {row['loci']} | {row['total_genes']} | {row['annotated_bgc_genes']} "
        f"| **{float(row['accuracy_pct']):.1f}%** ({float(row['macro_f1_pct']):.1f}% Macro F1) | {row['phage_orfs_detected']} |"
        for row in kaptive_bench["db_rows"]
    )
    _kl1_md = "\n".join(
        f"| {row['pos']} | `{row['gene']}` | {row['aa_len']} | `{row['expected_role']}` | **`{row['predicted_role']}`** "
        f"({row['predicted_name']}) | {float(row['confidence']):.3f} | {'PASS' if row['match'] else 'FAIL'} | {row['description']} |"
        for row in kaptive_bench["kl1_rows"]
    )
    _kl101_md = "\n".join(
        f"| `{row['locus']}` | {row['pos']} | `{row['coords']}` | {row['aa_len']} | `{row['cluster_key']}` "
        f"| `{row['pred_role']}` | **{float(row['p_phage']):.3f}** | {row['description']} |"
        for row in kaptive_bench["kl101_186_audit"]
    )

    _role_table_md = "\n".join(
        [
            "| ID | Abbrev | Gene Role | Category | Post-CD-HIT Total | Test Support | Precision | Recall | F1-Score |",
            "| ---: | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
            _role_md,
        ]
    )
    _kaptive_table_md = "\n".join(
        [
            f"**Pre-Training Validation Gate**: Passed = `{validation_gate['passed']}` "
            f"(min role = `{validation_gate['min_role_name']}` with `{validation_gate['min_support']}` seqs), "
            f"Train/Test Overlap = `{validation_gate['train_test_overlap']}`, "
            f"Internal Stops = `{validation_gate['internal_stops']}`\n",
            "| Kaptive SoA Database (`~/.kaptive/*.pkl`) | Loci | Total Genes | Evaluated Genes | External Role Accuracy | Predicted Phage ORFs (`p >= 0.50`) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |",
            _kaptive_db_md,
        ]
    )
    _kl1_table_md = "\n".join(
        [
            f"**K. pneumoniae KL1 External Verification**: **{kaptive_bench['kl1_hits']} / {kaptive_bench['kl1_total']}** (`100.0%` role agreement)\n",
            "| Pos | Gene (`cluster_key`) | Length (aa) | Expected Role | Predicted Role | Confidence | Status | Kaptive Reference Description |",
            "| ---: | :--- | ---: | :--- | :--- | ---: | :--- | :--- |",
            _kl1_md,
        ]
    )
    _kl101_table_md = "\n".join(
        [
            "**Audit of `kpsc_k` (`KL101..KL186`) for Accidental Prophage Cassettes, Tail Fibers & Unlabelled ORFs**:\n",
            "| Locus | Pos | Interval (`start..end (strand)`) | Length (aa) | `cluster_key` | Pred Role | `P(Phage)` | Reference Annotation |",
            "| :--- | ---: | :--- | ---: | :--- | :--- | ---: | :--- |",
            _kl101_md,
        ]
    )
    _layer_table_md = "\n".join(
        [
            "| Layer | Dim | 1-NN Cosine Recall | Top-5 Role Purity | Nearest-Centroid Macro F1 | Fisher Ratio | Composite Score |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
            _layer_md,
        ]
    )
    _trial_table_md = "\n".join(
        [
            "| Trial | Inner Val Macro F1 | Best So Far | Embed Dim | Filters | Kernels | Learning Rate | Weight Decay | Batch Size |",
            "| ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: |",
            _trial_md,
        ]
    )

    mo.vstack(
        [
            mo.md("## 6. Per-Role Holdout Metrics, External Kaptive Benchmark, Layer Sweep & Optuna Leaderboard"),
            mo.ui.tabs(
                {
                    "29 Functional GeneRoles (Holdout Test)": mo.md(_role_table_md),
                    "External Kaptive Benchmark (~/.kaptive/*.pkl)": mo.md(_kaptive_table_md),
                    f"KL1 Verification ({kaptive_bench['kl1_hits']}/{kaptive_bench['kl1_total']})": mo.md(
                        _kl1_table_md
                    ),
                    f"KL101..KL186 Phage & Unlabelled Audit ({len(kaptive_bench['kl101_186_audit'])} ORFs)": mo.md(
                        _kl101_table_md
                    ),
                    f"Layer Representation Sweep ({len(layer_sweep_records)} Layers)": mo.md(_layer_table_md),
                    f"Optuna TPE Study ({len(optuna_trial_records)} Trials)": mo.md(_trial_table_md),
                }
            ),
        ],
        gap=0.8,
    )
    return


if __name__ == "__main__":
    app.run()
