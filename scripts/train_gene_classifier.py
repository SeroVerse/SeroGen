# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo",
#     "altair>=5.5.0",
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

__generated_with = "0.11.0"
app = marimo.App(width="full", app_title="SeroGen — GeneRoleClassifier Training, Optuna Tuning & Validation")


@app.cell
def _():
    import gzip
    import os
    import re
    import time
    import xml.etree.ElementTree as ET
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path
    from urllib.error import HTTPError
    from urllib.parse import quote
    from urllib.request import Request, urlopen

    import altair as alt
    import marimo as mo
    import numpy as np
    import optuna
    from numba import njit
    from scipy.sparse import csr_array

    from serogen.core.io.fasta import FastaReader
    from serogen.core.seq.alphabet import Alphabet
    from serogen.core.seq.clustering import Clusterer
    from serogen.core.seq.seq import Sequences
    from serogen.genes import GeneRole, GeneRoleClassifier

    alt.data_transformers.disable_max_rows()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    return (
        Alphabet,
        Clusterer,
        ET,
        FastaReader,
        GeneRole,
        GeneRoleClassifier,
        HTTPError,
        Path,
        Request,
        Sequences,
        ThreadPoolExecutor,
        alt,
        as_completed,
        csr_array,
        gzip,
        mo,
        njit,
        np,
        optuna,
        os,
        quote,
        re,
        time,
        urlopen,
    )


@app.cell
def _(GeneRole, re):
    # ------------------------------------------------------------------------------------------------------------------
    # Multi-Source Query Registries (UniProtKB + Pfam/InterPro, UniRef50, NCBI IPG, and KEGG Orthology) for Roles 0..28
    # ------------------------------------------------------------------------------------------------------------------
    ROLE_QUERIES: dict[int, dict[str, object]] = {
        int(GeneRole.FLIPPASE): {
            "uniprot": '((gene:wzx OR protein_name:"O-antigen flippase" OR xref:pfam-PF01943 OR xref:pfam-PF14667) NOT gene:wzy NOT gene:waaL NOT gene:rfaL NOT gene:murJ NOT gene:ftsW NOT gene:msbA NOT protein_name:polymerase NOT protein_name:ligase NOT xref:pfam-PF04932 NOT xref:pfam-PF14296 NOT xref:pfam-PF13425 NOT xref:pfam-PF03023 NOT xref:pfam-PF01098) AND taxonomy_id:2',
            "uniref": '(name:"O-antigen flippase" OR name:"polysaccharide flippase") NOT name:polymerase NOT name:ligase NOT name:Wzy NOT name:WaaL NOT name:MurJ AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("O-antigen flippase"[Product] OR wzx[Gene Name]) NOT polymerase[Product] NOT ligase[Product] NOT wzy[Gene Name] NOT waaL[Gene Name] AND "Bacteria"[Organism]',
            "min_len": 280,
            "max_len": 550,
        },
        int(GeneRole.POLYMERASE): {
            "uniprot": '((gene:wzy OR protein_name:"O-antigen polymerase" OR protein_name:"capsular polysaccharide polymerase" OR xref:pfam-PF14296) NOT gene:waaL NOT gene:rfaL NOT gene:wzx NOT protein_name:ligase NOT protein_name:flippase NOT xref:pfam-PF04932 NOT xref:pfam-PF01943 NOT xref:pfam-PF14667 NOT xref:pfam-PF13440) AND taxonomy_id:2',
            "uniref": '(name:"O-antigen polymerase" OR name:"polysaccharide polymerase" OR name:Wzy) NOT name:ligase NOT name:flippase NOT name:WaaL NOT name:RfaL NOT name:Wzx AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("O-antigen polymerase"[Product] OR wzy[Gene Name]) NOT ligase[Product] NOT flippase[Product] NOT waaL[Gene Name] NOT wzx[Gene Name] AND "Bacteria"[Organism]',
            "min_len": 260,
            "max_len": 560,
        },
        int(GeneRole.COPOLYMERASE): {
            "uniprot": '(gene:wzz OR gene:wzzB OR gene:wzzE OR gene:fepE OR gene_exact:cld OR protein_name:"chain length determinant" OR protein_name:"polysaccharide copolymerase" OR xref:pfam-PF02706) NOT gene:wzc NOT protein_name:kinase AND taxonomy_id:2',
            "uniref": '(name:"chain length determinant" OR name:"polysaccharide co-polymerase" OR name:Wzz) NOT name:kinase NOT name:Wzc AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("chain length determinant"[Product] OR wzz[Gene Name] OR fepE[Gene Name]) AND "Bacteria"[Organism]',
            "min_len": 180,
            "max_len": 520,
        },
        int(GeneRole.KINASE): {
            "uniprot": '(gene:wzc OR gene:ptk OR gene:etk OR gene:cpsD OR gene:capB OR protein_name:"tyrosine-protein kinase Wzc" OR xref:pfam-PF13614) NOT gene:wza NOT gene:wzb NOT protein_name:phosphatase AND taxonomy_id:2',
            "uniref": '(name:"tyrosine-protein kinase Wzc" OR name:"tyrosine-protein kinase Etk" OR (name:"exopolysaccharide" AND name:kinase)) NOT name:phosphatase NOT name:Wza AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("tyrosine-protein kinase Wzc"[Product] OR wzc[Gene Name] OR ptk[Gene Name]) AND "Bacteria"[Organism]',
            "min_len": 200,
            "max_len": 820,
        },
        int(GeneRole.PHOSPHATASE): {
            "uniprot": '(gene:wzb OR gene:ptp OR gene:etp OR gene:cpsB OR protein_name:"phosphotyrosine protein phosphatase" OR xref:pfam-PF01451) NOT protein_name:kinase AND taxonomy_id:2',
            "uniref": '(name:"low molecular weight phosphotyrosine protein phosphatase" OR name:"phosphotyrosine-protein phosphatase" OR name:Wzb) NOT name:kinase AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("low molecular weight phosphotyrosine protein phosphatase"[Product] OR wzb[Gene Name]) AND "Bacteria"[Organism]',
            "min_len": 110,
            "max_len": 270,
        },
        int(GeneRole.ABC_PERMEASE): {
            "uniprot": '(gene:wzm OR gene:kpsM OR protein_name:"capsule export system permease" OR protein_name:"polysaccharide ABC transporter permease" OR xref:pfam-PF01061 OR xref:pfam-PF06562) NOT gene:wza NOT gene:kpsD NOT gene:kpsE NOT gene:wzt NOT gene:kpsT NOT protein_name:"ATP-binding" NOT protein_name:"outer membrane" AND taxonomy_id:2',
            "uniref": '((name:"ABC transporter permease" AND (name:polysaccharide OR name:capsule OR name:"O-antigen")) OR name:"capsule export system permease" OR name:Wzm OR name:KpsM) NOT name:Wza NOT name:KpsD NOT name:KpsE NOT name:Wzt NOT name:KpsT NOT name:"ATP-binding" NOT name:"outer membrane" AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("capsule export system permease"[Product] OR wzm[Gene Name] OR kpsM[Gene Name]) NOT wza[Gene Name] NOT kpsD[Gene Name] NOT wzt[Gene Name] AND "Bacteria"[Organism]',
            "min_len": 210,
            "max_len": 340,
        },
        int(GeneRole.ABC_ATPASE): {
            "uniprot": '(gene:wzt OR gene:kpsT OR protein_name:"capsule export system ATP-binding" OR protein_name:"teichoic acid export ATP-binding") NOT gene:wzm NOT gene:kpsM NOT protein_name:permease AND taxonomy_id:2',
            "uniref": '(name:"polysaccharide export ATP-binding" OR name:"capsule export system ATP-binding" OR name:Wzt OR name:KpsT) NOT name:permease NOT name:Wzm NOT name:KpsM AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("capsule export system ATP-binding"[Product] OR wzt[Gene Name] OR kpsT[Gene Name]) NOT wzm[Gene Name] NOT kpsM[Gene Name] AND "Bacteria"[Organism]',
            "min_len": 200,
            "max_len": 520,
        },
        int(GeneRole.LIGASE): {
            "uniprot": '((gene:waaL OR gene:rfaL OR protein_name:"O-antigen ligase" OR (xref:pfam-PF04932 AND protein_name:ligase)) NOT gene:wzy NOT gene:wzx NOT protein_name:polymerase NOT protein_name:flippase NOT xref:pfam-PF01943 NOT xref:pfam-PF14667 NOT xref:pfam-PF14296) AND taxonomy_id:2',
            "uniref": '(name:"O-antigen ligase" OR name:"lipid A-core:surface polymer ligase" OR name:WaaL OR name:RfaL) NOT name:polymerase NOT name:flippase NOT name:Wzy NOT name:Wzx AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("O-antigen ligase"[Product] OR waaL[Gene Name] OR rfaL[Gene Name]) NOT polymerase[Product] NOT flippase[Product] NOT wzy[Gene Name] NOT wzx[Gene Name] AND "Bacteria"[Organism]',
            "min_len": 280,
            "max_len": 540,
        },
        int(GeneRole.TRANSLOCON): {
            "uniprot": '(gene:wza OR gene:kpsD OR protein_name:"polysaccharide export protein Wza" OR protein_name:"outer membrane polysaccharide export" OR xref:pfam-PF02563) NOT gene:wzm NOT gene:kpsM NOT gene:kpsE NOT gene:wzc NOT protein_name:permease NOT protein_name:kinase NOT protein_name:"inner membrane" NOT protein_name:"ATP-binding" AND taxonomy_id:2',
            "uniref": '(name:"outer membrane polysaccharide export" OR name:"polysaccharide export protein Wza" OR name:Wza OR name:KpsD) NOT name:Wzm NOT name:KpsM NOT name:KpsE NOT name:permease NOT name:kinase NOT name:"ABC transporter" AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("outer membrane polysaccharide export"[Product] OR wza[Gene Name] OR kpsD[Gene Name]) NOT wzm[Gene Name] NOT kpsM[Gene Name] NOT kpsE[Gene Name] AND "Bacteria"[Organism]',
            "min_len": 290,
            "max_len": 500,
        },
        int(GeneRole.CAPSULE_ANCHOR): {
            "uniprot": '(gene:wzi OR gene:kpsC OR gene:wzg OR gene:cpsA OR gene:cps2A OR gene:capA OR protein_name:"capsule attachment protein" OR protein_name:"capsule assembly protein Wzi" OR protein_name:"capsular polysaccharide assembly protein Wzi" OR protein_name:"LytR-CpsA-Psr" OR xref:pfam-PF14052 OR (xref:pfam-PF03816 AND (gene:wzg OR gene:cps* OR gene:cap* OR protein_name:capsule))) AND taxonomy_id:2',
            "uniref": '(name:"capsule assembly protein Wzi" OR name:Wzi OR name:KpsC OR name:Wzg OR name:CpsA OR (name:"LytR-CpsA-Psr" AND name:capsule)) AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("capsule attachment protein"[Product] OR "capsule assembly protein Wzi"[Product] OR wzi[Gene Name] OR kpsC[Gene Name] OR wzg[Gene Name] OR cpsA[Gene Name] OR cps2A[Gene Name]) AND "Bacteria"[Organism]',
            "min_len": 320,
            "max_len": 680,
        },
        int(GeneRole.PYRUVYLTRANSFERASE): {
            "uniprot": "((gene:wcaK OR protein_name:pyruvyltransferase OR xref:pfam-PF04230 OR xref:interpro-IPR007345) NOT gene:murB NOT protein_name:reductase NOT protein_name:glycosyltransferase NOT xref:pfam-PF00534 NOT xref:pfam-PF00535 NOT xref:pfam-PF13439 NOT xref:pfam-PF13579 NOT xref:pfam-PF01565) AND taxonomy_id:2",
            "uniref": '(name:pyruvyl* OR name:"Polysaccharide pyruvyl transferase") NOT name:reductase NOT name:glycosyltransferase AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '(pyruvyltransferase[Product] OR "polysaccharide pyruvyl transferase"[Product] OR wcaK[Gene Name]) NOT glycosyltransferase[Product] NOT reductase[Product] AND "Bacteria"[Organism]',
            "min_len": 220,
            "max_len": 480,
        },
        int(GeneRole.METHYLTRANSFERASE): {
            "uniprot": '(protein_name:"polysaccharide O-methyltransferase" OR protein_name:"sugar O-methyltransferase" OR protein_name:"capsular polysaccharide methyltransferase") AND taxonomy_id:2',
            "uniref": '(name:"polysaccharide methyltransferase" OR name:"sugar O-methyltransferase") AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("polysaccharide O-methyltransferase"[Product] OR "sugar O-methyltransferase"[Product]) AND "Bacteria"[Organism]',
            "min_len": 160,
            "max_len": 520,
        },
        int(GeneRole.ACETYLTRANSFERASE): {
            "uniprot": '(gene:wecD OR gene:oatA OR gene:atr OR gene:wbbJ OR protein_name:"O-acetyltransferase" OR xref:pfam-PF01757) AND taxonomy_id:2',
            "uniref": '(name:"O-acetyltransferase" AND name:polysaccharide OR name:WecD OR name:OatA) AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("O-acetyltransferase"[Product] OR wecD[Gene Name] OR oatA[Gene Name]) AND "Bacteria"[Organism]',
            "min_len": 140,
            "max_len": 680,
        },
        int(GeneRole.PHOSPHOTRANSFERASE): {
            "uniprot": '(gene:licD OR gene:pgt OR gene:wcfF OR protein_name:"phosphocholine transferase" OR protein_name:"licD-family phosphotransferase" OR xref:pfam-PF04991) AND taxonomy_id:2',
            "uniref": '(name:"phosphocholine transferase" OR name:"LicD family" OR name:LicD) AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("phosphocholine transferase"[Product] OR licD[Gene Name] OR pgt[Gene Name]) AND "Bacteria"[Organism]',
            "min_len": 200,
            "max_len": 460,
        },
        int(GeneRole.FORMYLTRANSFERASE): {
            "uniprot": '(gene:wbqD OR gene:vioF OR gene:wbtB OR gene:fmt OR protein_name:"N-formyltransferase" OR protein_name:"dTDP-4-amino-4,6-dideoxyglucose formyltransferase" OR protein_name:"sugar N-formyltransferase") AND taxonomy_id:2',
            "uniref": '(name:"formyltransferase" OR name:"N-formyltransferase") AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("N-formyltransferase"[Product] OR "formyltransferase"[Product] OR wbqD[Gene Name]) AND "Bacteria"[Organism]',
            "min_len": 160,
            "max_len": 450,
        },
        int(GeneRole.ALANINE_TRANSFERASE): {
            "uniprot": '(gene:alt1 OR gene_exact:dltA OR gene:dltB OR gene:aat OR protein_name:"alanine--poly(phosphoribitol) ligase" OR protein_name:"D-alanine transferase" OR protein_name:"D-alanyl-lipoteichoic acid") AND taxonomy_id:2',
            "uniref": '(name:"D-alanine--poly(phosphoribitol) ligase" OR name:"D-alanyl-lipoteichoic acid" OR name:DltA) AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("alanine--poly(phosphoribitol) ligase"[Product] OR "D-alanine transferase"[Product] OR "D-alanyl-lipoteichoic acid biosynthesis"[Product] OR alt1[Gene Name]) AND "Bacteria"[Organism]',
            "min_len": 160,
            "max_len": 560,
        },
        int(GeneRole.AMINOTRANSFERASE): {
            "uniprot": '(gene:wecE OR gene:degT OR gene:pseC OR gene:perA OR gene:rhnb OR gene:rhnc OR protein_name:"dTDP-4-amino-4,6-dideoxygalactose transaminase" OR xref:pfam-PF01041) AND taxonomy_id:2',
            "uniref": '(name:"DegT/DnrJ/EryC1/StrS aminotransferase" OR name:"dTDP-4-amino-4,6-dideoxygalactose transaminase" OR name:WecE) AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("dTDP-4-amino-4,6-dideoxygalactose transaminase"[Product] OR wecE[Gene Name] OR pseC[Gene Name]) AND "Bacteria"[Organism]',
            "min_len": 280,
            "max_len": 480,
        },
        int(GeneRole.ALDOLASE): {
            "uniprot": '(gene:neuB OR gene:pseI OR gene:legI OR gene:kdsA OR protein_name:"N-acetylneuraminate synthase" OR protein_name:"pseudaminic acid synthase" OR xref:pfam-PF03102) AND taxonomy_id:2',
            "uniref": '(name:"N-acetylneuraminate synthase" OR name:"pseudaminic acid synthase" OR name:"2-dehydro-3-deoxyphosphooctonate aldolase") AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("N-acetylneuraminate synthase"[Product] OR "pseudaminic acid synthase"[Product] OR neuB[Gene Name] OR kdsA[Gene Name]) AND "Bacteria"[Organism]',
            "min_len": 220,
            "max_len": 450,
        },
        int(GeneRole.DEACETYLASE): {
            "uniprot": '(gene:pgaB OR gene:pdaA OR gene:icaB OR protein_name:"poly-beta-1,6-N-acetyl-D-glucosamine N-deacetylase" OR protein_name:"polysaccharide deacetylase" OR xref:pfam-PF01522) AND taxonomy_id:2',
            "uniref": '(name:"polysaccharide deacetylase" OR name:"poly-beta-1,6-N-acetyl-D-glucosamine N-deacetylase" OR name:PgaB) AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("poly-beta-1,6-N-acetyl-D-glucosamine N-deacetylase"[Product] OR "polysaccharide deacetylase"[Product] OR pgaB[Gene Name] OR pdaA[Gene Name]) AND "Bacteria"[Organism]',
            "min_len": 180,
            "max_len": 750,
        },
        int(GeneRole.EPIMERASE): {
            "uniprot": '(gene:gne OR gene:rmlC OR gene:galE OR gene:mnaA OR protein_name:"UDP-glucose 4-epimerase" OR xref:pfam-PF00908) AND taxonomy_id:2',
            "uniref": '(name:"UDP-glucose 4-epimerase" OR name:"dTDP-4-dehydrorhamnose 3,5-epimerase" OR name:RmlC OR name:GalE) AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("UDP-glucose 4-epimerase"[Product] OR galE[Gene Name] OR rmlC[Gene Name]) AND "Bacteria"[Organism]',
            "min_len": 150,
            "max_len": 420,
        },
        int(GeneRole.OXIDOREDUCTASE): {
            "uniprot": '(gene:rmlB OR gene:rmlD OR gene:ugd OR gene:gmd OR protein_name:"UDP-glucose 6-dehydrogenase" OR protein_name:"dTDP-glucose 4,6-dehydratase" OR xref:pfam-PF03721) AND taxonomy_id:2',
            "uniref": '(name:"UDP-glucose 6-dehydrogenase" OR name:"dTDP-glucose 4,6-dehydratase" OR name:"GDP-mannose 4,6-dehydratase") AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("UDP-glucose 6-dehydrogenase"[Product] OR "dTDP-glucose 4,6-dehydratase"[Product] OR "dTDP-4-dehydrorhamnose reductase"[Product] OR rmlB[Gene Name] OR ugd[Gene Name]) AND "Bacteria"[Organism]',
            "min_len": 240,
            "max_len": 500,
        },
        int(GeneRole.TRANSMEMBRANE_SYNTHASE): {
            "uniprot": '(gene:hasA OR gene:pgaC OR gene:icaA OR (gene:bcsA AND protein_name:"catalytic") OR protein_name:"hyaluronan synthase" OR protein_name:"cellulose synthase catalytic subunit" OR protein_name:"poly-beta-1,6-N-acetyl-D-glucosamine synthase" OR xref:pfam-PF13641) AND taxonomy_id:2',
            "uniref": '(name:"hyaluronan synthase" OR name:"cellulose synthase catalytic subunit" OR name:"poly-beta-1,6-N-acetyl-D-glucosamine synthase") AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("hyaluronan synthase"[Product] OR "cellulose synthase catalytic subunit"[Product] OR "cellulose synthase subunit A"[Product] OR hasA[Gene Name] OR pgaC[Gene Name] OR bcsA[Gene Name]) AND "Bacteria"[Organism]',
            "min_len": 340,
            "max_len": 1000,
        },
        int(GeneRole.NUCLEOTIDYLTRANSFERASE): {
            "uniprot": '(gene:rmlA OR gene:galU OR gene:glmM OR gene:manC OR protein_name:"glucose-1-phosphate thymidylyltransferase" OR xref:pfam-PF00483) AND taxonomy_id:2',
            "uniref": '(name:"glucose-1-phosphate thymidylyltransferase" OR name:"UTP--glucose-1-phosphate uridylyltransferase" OR name:RmlA) AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("glucose-1-phosphate thymidylyltransferase"[Product] OR rmlA[Gene Name] OR galU[Gene Name]) AND "Bacteria"[Organism]',
            "min_len": 210,
            "max_len": 520,
        },
        int(GeneRole.NUCLEOTIDASE): {
            "uniprot": '(gene:gmm OR gene:wcaH OR gene:psaD OR gene:lgaF OR protein_name:"GDP-mannose mannosyl hydrolase" OR protein_name:"sugar-nucleotide hydrolase" OR protein_name:"NUDIX hydrolase") AND taxonomy_id:2',
            "uniref": '(name:"GDP-mannose mannosyl hydrolase" OR name:"GDP-mannose hydrolase" OR name:"NUDIX hydrolase") AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("GDP-mannose mannosyl hydrolase"[Product] OR "GDP-mannose hydrolase"[Product] OR gmm[Gene Name] OR psaD[Gene Name]) AND "Bacteria"[Organism]',
            "min_len": 120,
            "max_len": 290,
        },
        int(GeneRole.INITIATOR): {
            "uniprot": '(gene:wbaP OR gene:wcaJ OR gene:wecA OR gene:pglC OR protein_name:"undecaprenyl-phosphate galactose phosphotransferase" OR xref:pfam-PF02397) AND taxonomy_id:2',
            "uniref": '(name:"undecaprenyl-phosphate galactose phosphotransferase" OR name:"bacterial sugar transferase" OR name:WbaP OR name:WcaJ) AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("undecaprenyl-phosphate galactose phosphotransferase"[Product] OR wbaP[Gene Name] OR wcaJ[Gene Name] OR wecA[Gene Name]) AND "Bacteria"[Organism]',
            "min_len": 180,
            "max_len": 540,
        },
        int(GeneRole.GLYCOSYLHYDROLASE): {
            "uniprot": '(protein_name:"alginate lyase" OR protein_name:"polysaccharide lyase" OR protein_name:"depolymerase" OR xref:pfam-PF05426) AND taxonomy_id:2',
            "uniref": '(name:"polysaccharide lyase" OR name:"alginate lyase" OR name:"capsule depolymerase") AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("polysaccharide lyase"[Product] OR "alginate lyase"[Product]) AND "Bacteria"[Organism]',
            "min_len": 220,
            "max_len": 650,
        },
        int(GeneRole.GLYCOSYLTRANSFERASE): {
            "uniprot": '(protein_name:"glycosyltransferase" AND (xref:pfam-PF00534 OR xref:pfam-PF00535 OR xref:pfam-PF13439 OR xref:pfam-PF13579 OR xref:pfam-PF01501)) NOT protein_name:synthase NOT protein_name:cellulose NOT protein_name:hyaluronan NOT protein_name:pyruvyltransferase NOT xref:pfam-PF04230 NOT gene:bcsA NOT gene:pgaC NOT gene:hasA NOT gene:icaA NOT gene:wecA NOT gene:wbaP NOT gene:wcaJ NOT protein_name:transposase NOT protein_name:integrase AND taxonomy_id:2',
            "uniref": '(name:"glycosyltransferase family 1" OR name:"glycosyltransferase family 2" OR name:"glycosyltransferase family 4" OR name:"capsular polysaccharide glycosyltransferase") NOT name:synthase NOT name:cellulose NOT name:hyaluronan NOT name:pyruvyltransferase NOT name:transposase NOT name:integrase AND identity:0.5 AND taxonomy_id:2',
            "gt_subqueries": [
                "(xref:pfam-PF00534 AND protein_name:glycosyltransferase) NOT protein_name:synthase NOT protein_name:pyruvyltransferase NOT xref:pfam-PF04230 NOT gene:wecA NOT gene:wbaP NOT gene:wcaJ AND taxonomy_id:2",
                "(xref:pfam-PF00535 AND protein_name:glycosyltransferase) NOT protein_name:synthase NOT protein_name:cellulose NOT protein_name:hyaluronan NOT protein_name:pyruvyltransferase NOT xref:pfam-PF04230 NOT gene:bcsA NOT gene:pgaC NOT gene:hasA NOT gene:icaA AND taxonomy_id:2",
                "((xref:pfam-PF13439 OR xref:pfam-PF13579) AND protein_name:glycosyltransferase) NOT protein_name:synthase NOT protein_name:pyruvyltransferase NOT xref:pfam-PF04230 AND taxonomy_id:2",
                "((xref:pfam-PF01501 OR xref:pfam-PF01531 OR xref:pfam-PF01755 OR xref:pfam-PF03808) AND protein_name:glycosyltransferase) NOT protein_name:synthase NOT protein_name:pyruvyltransferase NOT xref:pfam-PF04230 AND taxonomy_id:2",
            ],
            "ncbi_ipg": '(glycosyltransferase[Product] NOT synthase[Product] NOT cellulose[Product] NOT pyruvyltransferase[Product]) AND "Bacteria"[Organism]',
            "min_len": 210,
            "max_len": 460,
            "quota_multiplier": 1.6,
        },
        int(GeneRole.TRANSPOSASE): {
            "uniprot": '((xref:pfam-PF01609 OR xref:pfam-PF01527 OR xref:pfam-PF00872 OR xref:pfam-PF13359 OR protein_name:"IS3 family transposase" OR protein_name:"IS5 family transposase" OR protein_name:"IS6 family transposase" OR protein_name:"IS256 family transposase") AND protein_name:transposase) NOT protein_name:integrase NOT protein_name:recombinase NOT protein_name:glycosyltransferase AND taxonomy_id:2',
            "uniref": '(name:"IS3 family transposase" OR name:"IS5 family transposase" OR name:"IS6 family transposase" OR name:"IS256 family transposase") NOT name:integrase NOT name:glycosyltransferase AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("IS family transposase"[Product] OR "insertion sequence transposase"[Product] OR (transposase[Product] NOT integrase[Product])) AND "Bacteria"[Organism]',
            "min_len": 120,
            "max_len": 520,
        },
        int(GeneRole.INTEGRASE): {
            "uniprot": '(protein_name:integrase OR protein_name:"site-specific recombinase" OR xref:pfam-PF00589) NOT protein_name:transposase NOT protein_name:glycosyltransferase AND taxonomy_id:2',
            "uniref": '(name:"site-specific integrase" OR name:"tyrosine recombinase" OR name:"phage integrase") NOT name:transposase NOT name:glycosyltransferase AND identity:0.5 AND taxonomy_id:2',
            "ncbi_ipg": '("site-specific integrase"[Product] OR "phage integrase"[Product] OR "tyrosine recombinase XerC"[Product] OR "tyrosine recombinase XerD"[Product]) AND "Bacteria"[Organism]',
            "min_len": 240,
            "max_len": 560,
        },
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
        "K25307": GeneRole.PHOSPHATASE,
        "K16690": GeneRole.ABC_PERMEASE,
        "K16691": GeneRole.ABC_ATPASE,
        "K01991": GeneRole.TRANSLOCON,
        "K09695": GeneRole.CAPSULE_ANCHOR,
        "K19225": GeneRole.CAPSULE_ANCHOR,
        "K02841": GeneRole.INITIATOR,
        "K03606": GeneRole.INITIATOR,
        "K02843": GeneRole.INITIATOR,
        "K00607": GeneRole.FORMYLTRANSFERASE,
        "K16560": GeneRole.PYRUVYLTRANSFERASE,
        "K00558": GeneRole.METHYLTRANSFERASE,
        "K01443": GeneRole.DEACETYLASE,
        "K01444": GeneRole.DEACETYLASE,
        "K01654": GeneRole.ALDOLASE,
        "K05304": GeneRole.ALDOLASE,
        "K01627": GeneRole.ALDOLASE,
        "K00832": GeneRole.AMINOTRANSFERASE,
        "K01791": GeneRole.EPIMERASE,
        "K01790": GeneRole.EPIMERASE,
        "K02473": GeneRole.EPIMERASE,
        "K00012": GeneRole.OXIDOREDUCTASE,
        "K00067": GeneRole.OXIDOREDUCTASE,
        "K02377": GeneRole.OXIDOREDUCTASE,
        "K00697": GeneRole.TRANSMEMBRANE_SYNTHASE,
        "K01795": GeneRole.TRANSMEMBRANE_SYNTHASE,
        "K00703": GeneRole.GLYCOSYLTRANSFERASE,
        "K01186": GeneRole.GLYCOSYLHYDROLASE,
        "K07483": GeneRole.TRANSPOSASE,
        "K04763": GeneRole.INTEGRASE,
    }

    ROLE_TO_KOS: dict[int, list[str]] = {}
    for _ko, _role in KEGG_KO_MAP.items():
        ROLE_TO_KOS.setdefault(int(_role), []).append(_ko)

    RefSeqAccPattern = re.compile(r"^[A-Z]{2,}_")

    return KEGG_KO_MAP, ROLE_QUERIES, ROLE_TO_KOS, RefSeqAccPattern


@app.cell
def _(
    Alphabet,
    Clusterer,
    ET,
    FastaReader,
    GeneRole,
    GeneRoleClassifier,
    HTTPError,
    Path,
    ROLE_QUERIES,
    ROLE_TO_KOS,
    RefSeqAccPattern,
    Request,
    Sequences,
    ThreadPoolExecutor,
    csr_array,
    gzip,
    mo,
    njit,
    np,
    optuna,
    os,
    quote,
    time,
    urlopen,
):
    # ------------------------------------------------------------------------------------------------------------------
    # Pure In-Memory Multi-Source Sequence Fetcher (with Post-Clustering Top-Up), Pre-Encoder & Optuna TPE Engine
    # ------------------------------------------------------------------------------------------------------------------
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
        """Perform an in-memory HTTP GET request with gzip decompression and exponential backoff."""
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
        """Fetch full-length bacterial protein sequences from UniProtKB REST API into `Sequences`."""
        encoded_q = quote(f"({query}) AND fragment:false")
        url = f"https://rest.uniprot.org/uniprotkb/search?query={encoded_q}&format=fasta&size={limit}"
        raw_fasta = http_get_bytes(url)
        if not raw_fasta or not raw_fasta.strip():
            return Sequences.empty()
        return FastaReader.from_bytes(raw_fasta).load()

    def fetch_uniref50_sequences(query: str, limit: int = 180) -> Sequences:
        """Fetch pre-clustered (<50% identity) bacterial representative sequences from UniRef50."""
        encoded_q = quote(query)
        url = f"https://rest.uniprot.org/uniref/search?query={encoded_q}&format=fasta&size={limit}"
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

        summary_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
            f"db=ipg&id={','.join(id_list)}&retmode=xml{api_param}"
        )
        sum_xml = http_get_bytes(summary_url)
        if not sum_xml:
            return Sequences.empty()

        wp_accs: list[str] = []
        try:
            sum_root = ET.fromstring(sum_xml)
            for doc in sum_root.findall(".//DocumentSummary"):
                acc = doc.findtext("Accession")
                if acc and (acc.startswith("WP_") or acc.startswith("NP_") or RefSeqAccPattern.match(acc)):
                    wp_accs.append(acc)
        except Exception:
            return Sequences.empty()

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

    def fetch_in_memory_dataset(
        target_per_role: int = 80,
        min_identity: float = 80.0,
        k_std: float = 3.5,
        include_ncbi: bool = True,
        include_kegg: bool = True,
        seed: int = 42,
    ) -> tuple[Sequences, np.ndarray]:
        """Fetch, cluster on the main thread, and top-up post-clustering deficits across all 29 roles."""
        api_key = resolve_ncbi_api_key()
        clusterer = Clusterer(min_identity=min_identity, mode="cdhit")
        n_functional = 29

        def _role_target(role_idx: int) -> int:
            q_spec = ROLE_QUERIES.get(role_idx, {})
            mult = float(q_spec.get("quota_multiplier", 1.0))  # type: ignore[arg-type]
            return int(round(target_per_role * mult))

        # Phase 1: Parallel HTTP I/O from UniProtKB (Pfam/InterPro enriched) + UniRef50 (pre-clustered <50% identity)
        gt_excl_uniprot = (
            " NOT xref:pfam-PF00534 NOT xref:pfam-PF00535 NOT xref:pfam-PF13439"
            " NOT xref:pfam-PF13579 NOT xref:pfam-PF01501 NOT protein_name:glycosyltransferase"
        )
        gt_excl_uniref = " NOT name:glycosyltransferase"

        def _fetch_primary_http(role_idx: int) -> Sequences:
            q_spec = ROLE_QUERIES.get(role_idx, {})
            r_target = _role_target(role_idx)
            batches: list[Sequences] = []
            exclude_gt = role_idx not in (
                int(GeneRole.GLYCOSYLTRANSFERASE),
                int(GeneRole.TRANSMEMBRANE_SYNTHASE),
            )

            gt_subs = q_spec.get("gt_subqueries")
            if isinstance(gt_subs, list) and gt_subs:
                sub_limit = max((r_target // len(gt_subs)) + 60, 120)
                for sub_q in gt_subs:
                    sub_seqs = fetch_uniprot_sequences(str(sub_q), limit=sub_limit)
                    if len(sub_seqs) > 0:
                        batches.append(sub_seqs)
            else:
                up_q = q_spec.get("uniprot")
                if up_q:
                    up_q_str = f"({up_q}){gt_excl_uniprot}" if exclude_gt else str(up_q)
                    up_seqs = fetch_uniprot_sequences(up_q_str, limit=max(r_target + 100, 180))
                    if len(up_seqs) > 0:
                        batches.append(up_seqs)

            ur_q = q_spec.get("uniref")
            if ur_q:
                ur_q_str = f"({ur_q}){gt_excl_uniref}" if exclude_gt else str(ur_q)
                ur_seqs = fetch_uniref50_sequences(ur_q_str, limit=max(r_target + 60, 140))
                if len(ur_seqs) > 0:
                    batches.append(ur_seqs)
            return Sequences.concat(batches) if batches else Sequences.empty()

        with ThreadPoolExecutor(max_workers=8) as pool:
            primary_raw_list = list(pool.map(_fetch_primary_http, range(n_functional)))

        # Phase 2: Main-thread Numba parallel=True clustering & POST-CLUSTERING deficit detection
        curated_map: dict[int, Sequences] = {}
        deficit_roles: list[int] = []
        for r_idx, raw_seqs in enumerate(primary_raw_list):
            q_spec = ROLE_QUERIES.get(r_idx, {})
            r_min_len = int(q_spec.get("min_len", 30))  # type: ignore[call-overload]
            r_max_len = int(q_spec.get("max_len", 2500))  # type: ignore[call-overload]
            clustered = filter_and_cluster_role_sequences(
                raw_seqs,
                clusterer=clusterer,
                k_std=k_std,
                min_len=r_min_len,
                max_len=r_max_len,
            )
            curated_map[r_idx] = clustered
            if len(clustered) < _role_target(r_idx):
                deficit_roles.append(r_idx)

        # Phase 3: Post-clustering enrichment from NCBI IPG & KEGG Orthology for any role < role_target
        if deficit_roles and (include_ncbi or include_kegg):

            def _fetch_secondary_http(role_idx: int) -> tuple[int, Sequences]:
                q_spec = ROLE_QUERIES.get(role_idx, {})
                r_target = _role_target(role_idx)
                sec_batches: list[Sequences] = []
                if include_ncbi and "ncbi_ipg" in q_spec:
                    ipg_seqs = fetch_ncbi_ipg_sequences(
                        str(q_spec["ncbi_ipg"]),
                        limit=max(r_target * 2, 100),
                        api_key=api_key,
                    )
                    if len(ipg_seqs) > 0:
                        sec_batches.append(ipg_seqs)
                if include_kegg and role_idx in ROLE_TO_KOS:
                    for ko_id in ROLE_TO_KOS[role_idx]:
                        ko_seqs = fetch_kegg_ko_sequences(ko_id, max_seqs=40)
                        if len(ko_seqs) > 0:
                            sec_batches.append(ko_seqs)
                return role_idx, (Sequences.concat(sec_batches) if sec_batches else Sequences.empty())

            with ThreadPoolExecutor(max_workers=6) as pool:
                sec_results = list(pool.map(_fetch_secondary_http, deficit_roles))

            for r_idx, sec_seqs in sec_results:
                if len(sec_seqs) > 0:
                    q_spec = ROLE_QUERIES.get(r_idx, {})
                    r_min_len = int(q_spec.get("min_len", 30))  # type: ignore[call-overload]
                    r_max_len = int(q_spec.get("max_len", 2500))  # type: ignore[call-overload]
                    combined = Sequences.concat([curated_map[r_idx], sec_seqs])
                    curated_map[r_idx] = filter_and_cluster_role_sequences(
                        combined,
                        clusterer=clusterer,
                        k_std=k_std,
                        min_len=r_min_len,
                        max_len=r_max_len,
                    )

        # Phase 4: Cap each role at _role_target(r_idx) and pack into a single pre-encoded Sequences container
        ordered_batches: list[Sequences] = []
        label_parts: list[np.ndarray] = []
        for r_idx in range(n_functional):
            b = curated_map.get(r_idx, Sequences.empty())
            r_target = _role_target(r_idx)
            if len(b) > r_target:
                role_rng = np.random.default_rng(seed + r_idx)
                chosen = np.sort(role_rng.choice(len(b), size=r_target, replace=False).astype(np.int32))
                b = b[chosen]
            if len(b) > 0:
                ordered_batches.append(b)
                label_parts.append(np.full(len(b), r_idx, dtype=np.int8))

        if not ordered_batches:
            return Sequences.empty(), np.empty(0, dtype=np.int8)

        all_seqs = Sequences.concat(ordered_batches)
        encoded_tokens = np.minimum(Alphabet.AMINO.encode(all_seqs.seqs), np.uint8(22))
        encoded_seqs = Sequences(all_seqs.ids, encoded_tokens, all_seqs.offsets, all_seqs.lengths)
        all_labels = np.concatenate(label_parts)
        return encoded_seqs, all_labels

    @njit(cache=True, nogil=True)
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

    @njit(cache=True, nogil=True)
    def stratified_kfold_assign(
        y: np.ndarray,
        n_folds: int,
        seed: int,
    ) -> np.ndarray:
        """Assign each sample in `y` to a stratified fold ID in `0..n_folds-1`."""
        np.random.seed(seed)
        n = len(y)
        order = np.argsort(y)
        fold_ids = np.empty(n, dtype=np.int32)

        start = 0
        rr = 0
        while start < n:
            c = y[order[start]]
            end = start + 1
            while end < n and y[order[end]] == c:
                end += 1

            count = end - start
            class_indices = order[start:end].copy()
            for i in range(count - 1, 0, -1):
                j = np.random.randint(0, i + 1)
                tmp = class_indices[i]
                class_indices[i] = class_indices[j]
                class_indices[j] = tmp
            for k in range(count):
                fold_ids[class_indices[k]] = rr % n_folds
                rr += 1
            start = end

        return fold_ids

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

    def run_optuna_study(
        X_train: Sequences,
        y_train: np.ndarray,
        n_trials: int,
        cv_folds: int,
        epochs: int,
        seed: int,
    ) -> tuple[optuna.Study, list[dict[str, object]], list[dict[str, object]]]:
        """Run an in-memory Optuna TPE study maximizing inner-validation Macro F1 on `X_train`."""
        if cv_folds <= 1:
            tr_counts = np.bincount(y_train.astype(np.int32), minlength=29)
            max_val_per_class = max(1, int(round(float(np.median(tr_counts[:29])) * 0.20)))
            inner_tr, inner_val = stratified_split_kernel(y_train, 0.20, seed + 101, max_val_per_class)
            splits = [(X_train[inner_tr], y_train[inner_tr], X_train[inner_val], y_train[inner_val])]
        else:
            fold_ids = stratified_kfold_assign(y_train, cv_folds, seed + 101)
            splits = []
            for f in range(cv_folds):
                tr_mask = fold_ids != f
                val_mask = fold_ids == f
                splits.append((X_train[tr_mask], y_train[tr_mask], X_train[val_mask], y_train[val_mask]))

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        with mo.status.progress_bar(
            total=n_trials,
            title="Optuna TPE Hyperparameter Search",
            subtitle="Evaluating candidate 1D-CNN architectures...",
        ) as bar:
            for _ in range(n_trials):
                trial = study.ask()
                embed_dim = trial.suggest_categorical("embed_dim", [16, 24, 32, 48, 64])
                num_filters = trial.suggest_categorical("num_filters", [64, 96, 128, 192, 256])
                kernel_combo = trial.suggest_categorical(
                    "kernel_combo",
                    ["3, 7, 11", "5, 9, 13", "3, 5, 9, 15", "7, 11, 15", "5, 11, 17"],
                )
                lr = trial.suggest_float("learning_rate", 1e-3, 2.5e-2, log=True)
                wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
                bs = trial.suggest_categorical("batch_size", [16, 32, 64])

                k_tuple = tuple(int(k.strip()) for k in kernel_combo.split(","))
                fold_f1s: list[float] = []

                for f_idx, (x_tr, y_tr, x_vl, y_vl) in enumerate(splits):
                    cand_clf = GeneRoleClassifier(
                        vocab_size=23,
                        embed_dim=int(embed_dim),
                        num_filters=int(num_filters),
                        kernel_size=k_tuple,
                        learning_rate=float(lr),
                        epochs=epochs,
                        batch_size=int(bs),
                        weight_decay=float(wd),
                        class_weight="balanced",
                        random_state=seed + f_idx,
                    )
                    cand_clf.fit(x_tr, y_tr)
                    preds_vl = cand_clf.predict(x_vl)
                    m_vl = evaluate_sparse_metrics(y_vl, preds_vl, cand_clf.classes_)
                    fold_f1s.append(float(m_vl["macro_f1"]))

                mean_f1 = float(np.mean(fold_f1s))
                study.tell(trial, mean_f1)
                bar.update(
                    subtitle=f"Trial {trial.number + 1}/{n_trials} — Val Macro F1: {mean_f1 * 100:.2f}% (Best: {study.best_value * 100:.2f}%)"
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
        if len(study.trials) >= 4:
            try:
                imp_map = optuna.importance.get_param_importances(study)
                for p_name, p_val in imp_map.items():
                    importance_records.append({"parameter": p_name, "importance": float(p_val)})
            except Exception:
                pass

        return study, trial_records, importance_records

    return (
        stratified_kfold_assign,
        stratified_split_kernel,
        evaluate_sparse_metrics,
        fetch_in_memory_dataset,
        fetch_kegg_ko_sequences,
        fetch_ncbi_ipg_sequences,
        fetch_uniprot_sequences,
        fetch_uniref50_sequences,
        filter_and_cluster_role_sequences,
        http_get_bytes,
        resolve_ncbi_api_key,
        run_optuna_study,
    )


@app.cell
def _(mo):
    # ------------------------------------------------------------------------------------------------------------------
    # Interactive Control Panel (Dataset Curation, 1D-CNN Hyperparameters & Optuna TPE Tuning)
    # ------------------------------------------------------------------------------------------------------------------
    target_per_role_ui = mo.ui.slider(start=30, stop=250, step=10, value=120, label="Max Sequences / Role")
    min_identity_ui = mo.ui.slider(start=50.0, stop=95.0, step=5.0, value=80.0, label="CD-HIT Identity (%)")
    k_std_ui = mo.ui.slider(start=2.0, stop=5.0, step=0.5, value=3.5, label="Length MAD Threshold (k)")
    test_fraction_ui = mo.ui.slider(start=0.10, stop=0.40, step=0.05, value=0.20, label="Test Split Fraction")
    include_ncbi_ui = mo.ui.checkbox(value=True, label="Include NCBI IPG")
    include_kegg_ui = mo.ui.checkbox(value=True, label="Include KEGG Orthology")

    embed_dim_ui = mo.ui.slider(start=16, stop=64, step=8, value=32, label="Embedding Dim")
    num_filters_ui = mo.ui.slider(start=32, stop=256, step=32, value=128, label="Conv Filters")
    kernel_sizes_ui = mo.ui.text(value="5, 9, 13", label="Multi-Scale Kernel Sizes")
    epochs_ui = mo.ui.slider(start=5, stop=60, step=5, value=30, label="Epochs")
    batch_size_ui = mo.ui.slider(start=8, stop=64, step=8, value=32, label="Mini-Batch Size")
    lr_ui = mo.ui.number(start=0.0005, stop=0.05, step=0.0005, value=0.008, label="AdamW Learning Rate")
    weight_decay_ui = mo.ui.number(start=0.0, stop=0.01, step=0.0001, value=0.0001, label="AdamW Weight Decay")
    seed_ui = mo.ui.number(start=0, stop=99999, step=1, value=42, label="Random Seed")

    n_trials_ui = mo.ui.slider(start=5, stop=80, step=5, value=20, label="Optuna Trials")
    cv_folds_ui = mo.ui.dropdown(
        options={
            "Fast Inner Split (80/20)": 1,
            "3-Fold Stratified CV": 3,
            "5-Fold Stratified CV": 5,
        },
        value="Fast Inner Split (80/20)",
        label="Optuna Inner Validation",
    )
    tune_btn = mo.ui.run_button(label="Run Optuna Hyperparameter Study")

    header_md = mo.md(
        """
        # SeroGen `GeneRoleClassifier` — Unified In-Memory Training, Optuna Tuning & Validation
        Fetches non-redundant bacterial reference proteins across **UniProtKB (Pfam/InterPro)**, **UniRef50**,
        **NCBI IPG**, and **KEGG Orthology** directly into an in-memory
        [`Sequences`](file:///Users/tsta0015/Programming/SeroGen/src/serogen/core/seq/seq.py) container, deduplicates
        homologs via [`Clusterer`](file:///Users/tsta0015/Programming/SeroGen/src/serogen/core/seq/clustering.py) with
        **post-clustering deficit top-up**, optionally tunes with **Optuna TPE**, and evaluates
        [`GeneRoleClassifier`](file:///Users/tsta0015/Programming/SeroGen/src/serogen/classifier.py).
        """
    )

    controls_panel = mo.vstack(
        [
            header_md,
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("### 1. In-Memory Dataset Curation"),
                            target_per_role_ui,
                            min_identity_ui,
                            k_std_ui,
                            test_fraction_ui,
                            mo.hstack([include_ncbi_ui, include_kegg_ui]),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("### 2. Manual 1D-CNN & AdamW Defaults"),
                            embed_dim_ui,
                            num_filters_ui,
                            kernel_sizes_ui,
                            epochs_ui,
                            batch_size_ui,
                            mo.hstack([lr_ui, weight_decay_ui, seed_ui]),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("### 3. Optuna TPE Tuning (In-Memory)"),
                            n_trials_ui,
                            cv_folds_ui,
                            tune_btn,
                        ]
                    ),
                ],
                widths="equal",
            ),
        ]
    )
    mo.output.replace(controls_panel)
    return (
        batch_size_ui,
        controls_panel,
        cv_folds_ui,
        embed_dim_ui,
        epochs_ui,
        header_md,
        include_kegg_ui,
        include_ncbi_ui,
        k_std_ui,
        kernel_sizes_ui,
        lr_ui,
        min_identity_ui,
        n_trials_ui,
        num_filters_ui,
        seed_ui,
        target_per_role_ui,
        test_fraction_ui,
        tune_btn,
        weight_decay_ui,
    )


@app.cell
def _(
    fetch_in_memory_dataset,
    include_kegg_ui,
    include_ncbi_ui,
    k_std_ui,
    min_identity_ui,
    mo,
    seed_ui,
    target_per_role_ui,
    time,
):
    # ------------------------------------------------------------------------------------------------------------------
    # Execute In-Memory Data Fetching, Post-Clustering Top-Up & Token Pre-Encoding
    # ------------------------------------------------------------------------------------------------------------------
    _t0 = time.perf_counter()
    with mo.status.spinner(title="Fetching, clustering & verifying post-clustering role quotas in memory..."):
        dataset_seqs, dataset_labels = fetch_in_memory_dataset(
            target_per_role=int(target_per_role_ui.value),
            min_identity=float(min_identity_ui.value),
            k_std=float(k_std_ui.value),
            include_ncbi=bool(include_ncbi_ui.value),
            include_kegg=bool(include_kegg_ui.value),
            seed=int(seed_ui.value),
        )
    fetch_elapsed = time.perf_counter() - _t0
    return dataset_labels, dataset_seqs, fetch_elapsed


@app.cell
def _(
    GeneRole,
    GeneRoleClassifier,
    Sequences,
    stratified_split_kernel,
    batch_size_ui,
    cv_folds_ui,
    dataset_labels,
    dataset_seqs,
    embed_dim_ui,
    epochs_ui,
    evaluate_sparse_metrics,
    fetch_elapsed,
    kernel_sizes_ui,
    lr_ui,
    mo,
    n_trials_ui,
    np,
    num_filters_ui,
    run_optuna_study,
    seed_ui,
    test_fraction_ui,
    time,
    tune_btn,
    weight_decay_ui,
):
    # ------------------------------------------------------------------------------------------------------------------
    # Hard 29-Role Quota Verification, Ragged Split, Optional Optuna TPE Study & Final Numba AdamW Training
    # ------------------------------------------------------------------------------------------------------------------
    mo.stop(
        len(dataset_seqs) == 0,
        mo.md("> **Warning:** No sequences were fetched. Check network connectivity and re-run."),
    )

    role_counts = np.bincount(dataset_labels.astype(np.int32), minlength=29)
    min_role_support = int(np.min(role_counts[:29]))
    min_role_name = GeneRole(int(np.argmin(role_counts[:29]))).name
    quota_passed = bool(np.all(role_counts[:29] >= 30))

    max_test_per_role = max(1, int(round(float(np.median(role_counts[:29])) * float(test_fraction_ui.value))))
    train_idx, test_idx = stratified_split_kernel(
        dataset_labels,
        float(test_fraction_ui.value),
        int(seed_ui.value),
        max_test_per_role,
    )

    # Zero-loop ragged slicing powered by serogen.core.collections._gather_ragged_kernel
    X_train = dataset_seqs[train_idx]
    y_train = dataset_labels[train_idx]
    X_test = dataset_seqs[test_idx]
    y_test = dataset_labels[test_idx]

    optuna_study = None
    optuna_trial_records: list[dict[str, object]] = []
    optuna_importance_records: list[dict[str, object]] = []

    is_script_mode = not mo.running_in_notebook()

    if tune_btn.value or is_script_mode:
        optuna_study, optuna_trial_records, optuna_importance_records = run_optuna_study(
            X_train=X_train,
            y_train=y_train,
            n_trials=int(n_trials_ui.value),
            cv_folds=int(cv_folds_ui.value),
            epochs=int(epochs_ui.value),
            seed=int(seed_ui.value),
        )
        best_p = optuna_study.best_params
        active_embed_dim = int(best_p["embed_dim"])
        active_num_filters = int(best_p["num_filters"])
        parsed_kernels = tuple(int(k.strip()) for k in str(best_p["kernel_combo"]).split(","))
        active_lr = float(best_p["learning_rate"])
        active_wd = float(best_p["weight_decay"])
        active_bs = int(best_p["batch_size"])
        config_source = (
            f"Optuna Best (Trial #{optuna_study.best_trial.number + 1}, Val F1={optuna_study.best_value * 100:.1f}%)"
        )
    else:
        active_embed_dim = int(embed_dim_ui.value)
        active_num_filters = int(num_filters_ui.value)
        parsed_kernels = tuple(int(k.strip()) for k in kernel_sizes_ui.value.split(",") if k.strip())
        active_lr = float(lr_ui.value)
        active_wd = float(weight_decay_ui.value)
        active_bs = int(batch_size_ui.value)
        config_source = "Manual UI Hyperparameters"

    kernel_arg: int | tuple[int, ...] = parsed_kernels[0] if len(parsed_kernels) == 1 else parsed_kernels

    clf = GeneRoleClassifier(
        vocab_size=23,
        embed_dim=active_embed_dim,
        num_filters=active_num_filters,
        kernel_size=kernel_arg,
        learning_rate=active_lr,
        epochs=int(epochs_ui.value),
        batch_size=active_bs,
        weight_decay=active_wd,
        class_weight="balanced",
        random_state=int(seed_ui.value),
    )

    _train_t0 = time.perf_counter()
    with mo.status.spinner(title=f"Training final Multi-Scale 1D-CNN ({config_source})..."):
        clf.fit(X_train, y_train)
    train_elapsed = time.perf_counter() - _train_t0

    y_pred_test = clf.predict(X_test)
    test_probs = clf.predict_proba(X_test)
    test_max_conf = np.max(test_probs, axis=1)

    eval_metrics = evaluate_sparse_metrics(y_test, y_pred_test, clf.classes_)

    # Generate shuffled-composition decoy sequences in-memory to benchmark Open-Set UNKNOWN (Role 29) rejection
    _rng = np.random.default_rng(int(seed_ui.value) + 999)
    _decoy_bytes = [bytes(_rng.permutation(np.frombuffer(X_test[i].seq, dtype=np.uint8))) for i in range(len(X_test))]
    decoy_seqs = Sequences.from_bytes(_decoy_bytes)
    decoy_probs = clf.predict_proba(decoy_seqs)
    decoy_max_conf = np.max(decoy_probs, axis=1)

    summary_banner = mo.hstack(
        [
            mo.stat(
                label="Curated Sequences",
                value=f"{len(dataset_seqs):,}",
                caption=f"Train: {len(X_train):,} | Test: {len(X_test):,} ({fetch_elapsed:.1f}s)",
            ),
            mo.stat(
                label="29-Role Quota Gate",
                value="PASS" if quota_passed else "DEFICIT",
                caption=f"Min Role: {min_role_name} ({min_role_support} seqs)",
            ),
            mo.stat(
                label="Final Model Fit",
                value=f"{train_elapsed:.2f}s",
                caption=f"{config_source} | Loss: {clf.loss_history_[-1]:.4f}",
            ),
            mo.stat(
                label="Test Accuracy",
                value=f"{float(eval_metrics['accuracy']) * 100:.2f}%",
                caption=f"{len(clf.classes_)} / {len(GeneRole) - 1} Functional Roles",
            ),
            mo.stat(
                label="Macro F1-Score",
                value=f"{float(eval_metrics['macro_f1']) * 100:.2f}%",
                caption=(
                    f"Prec: {float(eval_metrics['macro_precision']) * 100:.1f}% | "
                    f"Rec: {float(eval_metrics['macro_recall']) * 100:.1f}%"
                ),
            ),
        ],
        widths="equal",
    )
    mo.output.replace(summary_banner)
    return (
        X_test,
        X_train,
        active_bs,
        active_embed_dim,
        active_lr,
        active_num_filters,
        active_wd,
        clf,
        config_source,
        decoy_max_conf,
        decoy_probs,
        decoy_seqs,
        eval_metrics,
        is_script_mode,
        kernel_arg,
        min_role_name,
        min_role_support,
        optuna_importance_records,
        optuna_study,
        optuna_trial_records,
        parsed_kernels,
        quota_passed,
        role_counts,
        summary_banner,
        test_idx,
        test_max_conf,
        test_probs,
        train_elapsed,
        train_idx,
        y_pred_test,
        y_test,
        y_train,
    )


@app.cell
def _(
    GeneRole,
    alt,
    clf,
    dataset_labels,
    dataset_seqs,
    decoy_max_conf,
    eval_metrics,
    mo,
    np,
    optuna_importance_records,
    optuna_trial_records,
    test_max_conf,
    y_test,
    y_train,
):
    # ------------------------------------------------------------------------------------------------------------------
    # Altair Diagnostic Visualizations (Zero-DataFrame, Pure NumPy/Dict Values)
    # ------------------------------------------------------------------------------------------------------------------

    # 1. Dataset Split Counts & Length IQR per Role
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
        .properties(title="Curated Sequence Counts per Role (Train vs. Test)", width=380, height=480)
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
        title="Protein Length Distribution (Median & IQR)", width=380, height=480
    )

    # 2. Training Loss Curve
    loss_records = [{"epoch": idx + 1, "loss": float(val)} for idx, val in enumerate(clf.loss_history_)]
    loss_chart = (
        alt.Chart(alt.Data(values=loss_records))
        .mark_line(point=True, strokeWidth=2.5, color="#2b5c8f")
        .encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y("loss:Q", title="Class-Weighted Cross-Entropy Loss"),
            tooltip=["epoch:Q", alt.Tooltip("loss:Q", format=".4f")],
        )
        .properties(title="AdamW Training Loss Convergence", width=360, height=280)
    )

    # 3. Open-Set Confidence & Decoy Rejection Sweep
    thresholds = np.linspace(0.10, 0.99, 45)
    sweep_records: list[dict[str, object]] = []
    for _th in thresholds:
        _tp_retain = float(np.mean(test_max_conf >= _th)) * 100.0
        _decoy_reject = float(np.mean(decoy_max_conf < _th)) * 100.0
        sweep_records.append({"threshold": float(_th), "rate": _tp_retain, "curve": "True Test Retained (%)"})
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
        .properties(title="Open-Set UNKNOWN (Role 29) Rejection vs. Test Retention", width=400, height=280)
    )

    # 4. Per-Role Precision, Recall & F1-Score Bar Chart
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
        .properties(title="Held-Out Test Precision, Recall & F1-Score per Role", width=380, height=520)
    )

    # 5. Sparse Row-Normalized Confusion Matrix Heatmap
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
        .properties(title="Held-Out Test Confusion Matrix (Row-Normalized %)", width=460, height=520)
    )

    dashboard_rows = [
        mo.md("## Diagnostic Visualizations"),
    ]
    vconcat_rows = []

    # 6. Optional Optuna Study Charts (Optimization History + Hyperparameter Importances)
    if optuna_trial_records:
        _hist_base = alt.Chart(alt.Data(values=optuna_trial_records))
        _trial_pts = _hist_base.mark_circle(size=75).encode(
            x=alt.X("trial:Q", title="Optuna Trial"),
            y=alt.Y("macro_f1:Q", title="Inner Validation Macro F1 (%)", scale=alt.Scale(zero=False)),
            color=alt.Color("kernel_combo:N", title="Kernel Sizes"),
            tooltip=[
                "trial:Q",
                alt.Tooltip("macro_f1:Q", format=".2f"),
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
            title="Optuna TPE Optimization History (Validation Macro F1 %)",
            width=400,
            height=260,
        )

        if optuna_importance_records:
            optuna_imp_chart = (
                alt.Chart(alt.Data(values=optuna_importance_records))
                .mark_bar(color="#4e79a7")
                .encode(
                    y=alt.Y("parameter:N", sort="-x", title="Hyperparameter"),
                    x=alt.X("importance:Q", title="Relative fANOVA Importance"),
                    tooltip=["parameter:N", alt.Tooltip("importance:Q", format=".3f")],
                )
                .properties(title="Optuna Hyperparameter Importance", width=360, height=260)
            )
            dashboard_rows.append(mo.hstack([optuna_history_chart, optuna_imp_chart], justify="start"))
            vconcat_rows.append(alt.hconcat(optuna_history_chart, optuna_imp_chart).resolve_scale(color="independent"))
        else:
            dashboard_rows.append(optuna_history_chart)
            vconcat_rows.append(optuna_history_chart)

    vconcat_rows.extend(
        [
            alt.hconcat(counts_chart, length_chart).resolve_scale(color="independent"),
            alt.hconcat(loss_chart, open_set_chart).resolve_scale(color="independent"),
            alt.hconcat(prf_chart, conf_chart).resolve_scale(color="independent"),
        ]
    )
    full_altair_report_chart = alt.vconcat(*vconcat_rows).resolve_scale(color="independent")

    dashboard_rows.extend(
        [
            mo.hstack([counts_chart, length_chart], justify="start"),
            mo.hstack([loss_chart, open_set_chart], justify="start"),
            mo.hstack([prf_chart, conf_chart], justify="start"),
        ]
    )

    dashboard = mo.vstack(dashboard_rows)
    mo.output.replace(dashboard)
    return (
        conf_chart,
        counts_chart,
        dashboard,
        dashboard_rows,
        full_altair_report_chart,
        len_base,
        len_pts,
        len_rule,
        length_chart,
        length_records,
        loss_chart,
        loss_records,
        open_set_chart,
        prf_chart,
        split_records,
        sweep_records,
        thresholds,
        vconcat_rows,
    )


@app.cell
def _(Path, mo):
    # ------------------------------------------------------------------------------------------------------------------
    # Explicit On-Demand / Script-Mode Model & HTML Report Export
    # ------------------------------------------------------------------------------------------------------------------
    default_model_path = Path(__file__).resolve().parent.parent / "src" / "serogen" / "data" / "model.npz"
    default_report_path = Path(__file__).resolve().parent / "train_gene_classifier_report.html"
    export_btn = mo.ui.run_button(
        label=f"Export Trained Weights ({default_model_path.name}) & HTML Report ({default_report_path.name})"
    )
    mo.output.replace(export_btn)
    return default_model_path, default_report_path, export_btn


@app.cell
def _(
    GeneRole,
    GeneRoleClassifier,
    X_test,
    X_train,
    active_bs,
    active_embed_dim,
    active_lr,
    active_num_filters,
    active_wd,
    clf,
    config_source,
    dataset_seqs,
    default_model_path,
    default_report_path,
    eval_metrics,
    export_btn,
    fetch_elapsed,
    full_altair_report_chart,
    is_script_mode,
    min_role_name,
    min_role_support,
    mo,
    np,
    optuna_trial_records,
    parsed_kernels,
    quota_passed,
    role_counts,
    train_elapsed,
    y_pred_test,
):
    mo.stop(
        not (export_btn.value or is_script_mode),
        mo.md(
            "*Model weights and charts are currently held in memory only. "
            "Click the button above (or run as a script) to write `model.npz` and `train_gene_classifier_report.html`.*"
        ),
    )
    mo.stop(
        not quota_passed or len(clf.classes_) < 29,
        mo.md("> **Export Blocked:** Not all 29 functional `GeneRole`s met the minimum 30-sequence quota."),
    )

    default_model_path.parent.mkdir(parents=True, exist_ok=True)
    clf.save(default_model_path)
    reloaded = GeneRoleClassifier.load(default_model_path)
    parity = bool(np.array_equal(reloaded.predict(X_test), y_pred_test))
    size_kb = default_model_path.stat().st_size / 1024.0

    # Build standalone HTML report with embedded Vega-Lite dashboard & summary tables
    charts_embed_html = full_altair_report_chart.to_html(fullhtml=False)

    trial_rows_html = ""
    for _tr in optuna_trial_records:
        trial_rows_html += (
            f"<tr>"
            f"<td>#{_tr['trial']}</td>"
            f"<td><strong>{_tr['macro_f1']:.2f}%</strong></td>"
            f"<td>{_tr['best_macro_f1']:.2f}%</td>"
            f"<td>{_tr['embed_dim']}</td>"
            f"<td>{_tr['num_filters']}</td>"
            f"<td><code>{_tr['kernel_combo']}</code></td>"
            f"<td>{_tr['learning_rate']:.5f}</td>"
            f"<td>{_tr['weight_decay']:.6f}</td>"
            f"<td>{_tr['batch_size']}</td>"
            f"</tr>\n"
        )

    role_rows_html = ""
    for _c_idx in range(29):
        _r = GeneRole(_c_idx)
        _tot_cnt = int(role_counts[_c_idx])
        _pr_items = [row for row in eval_metrics["per_role"] if row["role_name"] == _r.name]
        _p_val = next((row["score"] for row in _pr_items if row["metric"] == "Precision"), 0.0)
        _r_val = next((row["score"] for row in _pr_items if row["metric"] == "Recall"), 0.0)
        _f_val = next((row["score"] for row in _pr_items if row["metric"] == "F1-Score"), 0.0)
        _sup_val = next((row["support"] for row in _pr_items), 0)
        role_rows_html += (
            f"<tr>"
            f"<td>{_c_idx}</td>"
            f"<td><code>{_r.abbreviation}</code></td>"
            f"<td>{_r.name}</td>"
            f"<td>{_r.category.value}</td>"
            f"<td>{_tot_cnt}</td>"
            f"<td>{_sup_val}</td>"
            f"<td>{_p_val * 100:.1f}%</td>"
            f"<td>{_r_val * 100:.1f}%</td>"
            f"<td><strong>{_f_val * 100:.1f}%</strong></td>"
            f"</tr>\n"
        )

    optuna_table_section = (
        f"""
        <h2>Optuna TPE Hyperparameter Study ({len(optuna_trial_records)} Trials)</h2>
        <table>
          <thead>
            <tr>
              <th>Trial</th><th>Inner Val Macro F1</th><th>Best So Far</th>
              <th>Embed Dim</th><th>Filters</th><th>Kernels</th>
              <th>Learning Rate</th><th>Weight Decay</th><th>Batch Size</th>
            </tr>
          </thead>
          <tbody>{trial_rows_html}</tbody>
        </table>
        """
        if trial_rows_html
        else ""
    )

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>SeroGen GeneRoleClassifier — Training, Optuna Tuning & Validation Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 24px auto;
      max-width: 1180px;
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
  <h1>SeroGen <code>GeneRoleClassifier</code> — Training, Optuna Tuning &amp; Validation Report</h1>
  <p>
    <strong>Selected Configuration:</strong> {config_source} &mdash;
    <code>embed_dim={active_embed_dim}</code>, <code>num_filters={active_num_filters}</code>,
    <code>kernel_size={parsed_kernels}</code>, <code>lr={active_lr:.5f}</code>,
    <code>weight_decay={active_wd:.6f}</code>, <code>batch_size={active_bs}</code>
  </p>
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Curated Sequences</div>
      <div class="kpi-value">{len(dataset_seqs):,}</div>
      <div class="kpi-sub">Train: {len(X_train):,} | Test: {len(X_test):,} ({fetch_elapsed:.1f}s)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">29-Role Quota Gate</div>
      <div class="kpi-value">{"PASS" if quota_passed else "DEFICIT"}</div>
      <div class="kpi-sub">Min Role: {min_role_name} ({min_role_support} seqs)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Final Model Fit</div>
      <div class="kpi-value">{train_elapsed:.2f}s</div>
      <div class="kpi-sub">Final Loss: {clf.loss_history_[-1]:.4f} | {size_kb:.1f} KB</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Held-Out Test Accuracy</div>
      <div class="kpi-value">{float(eval_metrics["accuracy"]) * 100:.2f}%</div>
      <div class="kpi-sub">{len(clf.classes_)} / 29 Functional Roles</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Held-Out Macro F1</div>
      <div class="kpi-value">{float(eval_metrics["macro_f1"]) * 100:.2f}%</div>
      <div class="kpi-sub">Prec: {float(eval_metrics["macro_precision"]) * 100:.1f}% | Rec: {float(eval_metrics["macro_recall"]) * 100:.1f}%</div>
    </div>
  </div>

  <h2>Interactive Diagnostic Charts (Altair / Vega-Lite)</h2>
  <div class="chart-box">
    {charts_embed_html}
  </div>

  {optuna_table_section}

  <h2>Per-Role Post-Clustering Support &amp; Held-Out Test Performance (29 Functional Roles)</h2>
  <table>
    <thead>
      <tr>
        <th>ID</th><th>Abbrev</th><th>Gene Role</th><th>Category</th>
        <th>Post-CD-HIT Total</th><th>Test Support</th>
        <th>Precision</th><th>Recall</th><th>F1-Score</th>
      </tr>
    </thead>
    <tbody>{role_rows_html}</tbody>
  </table>
</body>
</html>
"""
    default_report_path.write_text(report_html, encoding="utf-8")
    report_kb = default_report_path.stat().st_size / 1024.0

    if is_script_mode:
        print(
            f"[SeroGen] Curated {len(dataset_seqs):,} non-redundant sequences across 29 roles "
            f"(min role: {min_role_name}={min_role_support}) in {fetch_elapsed:.1f}s"
        )
        print(f"[SeroGen] Configuration: {config_source}")
        print(
            f"[SeroGen] Held-Out Test Accuracy: {float(eval_metrics['accuracy']) * 100:.2f}% | "
            f"Macro F1: {float(eval_metrics['macro_f1']) * 100:.2f}% "
            f"(Prec: {float(eval_metrics['macro_precision']) * 100:.1f}%, "
            f"Rec: {float(eval_metrics['macro_recall']) * 100:.1f}%)"
        )
        print(f"[SeroGen] Saved weights -> {default_model_path} ({size_kb:.1f} KB, parity={parity})")
        print(f"[SeroGen] Saved HTML report -> {default_report_path} ({report_kb:.1f} KB)")

    mo.output.replace(
        mo.md(
            f"**Saved** `{default_model_path}` (`{size_kb:.1f} KB`, reload parity: `{parity}`) and "
            f"**HTML report** `{default_report_path}` (`{report_kb:.1f} KB`)."
        )
    )
    return (
        charts_embed_html,
        optuna_table_section,
        parity,
        reloaded,
        report_html,
        report_kb,
        role_rows_html,
        size_kb,
        trial_rows_html,
    )


if __name__ == "__main__":
    app.run()
