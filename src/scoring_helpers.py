"""
AlphaGenome score extraction helpers shared by scoring and inference.

Derived from v1 src/scoring.py — the inference-time logic and training-time
aggregation must agree exactly so that predictions at serving time are a
proper drop-in replacement for cached scores.
"""
import numpy as np
import pandas as pd


RESULT_MAP = {
    0:  {'output_type': 'ATAC',              'scorer': 'center_mask'},
    1:  {'output_type': 'CONTACT_MAPS',      'scorer': 'contact_map'},
    2:  {'output_type': 'DNASE',             'scorer': 'center_mask'},
    3:  {'output_type': 'CHIP_TF',           'scorer': 'center_mask'},
    4:  {'output_type': 'CHIP_HISTONE',      'scorer': 'center_mask'},
    5:  {'output_type': 'CAGE',              'scorer': 'center_mask'},
    6:  {'output_type': 'PROCAP',            'scorer': 'center_mask'},
    7:  {'output_type': 'RNA_SEQ',           'scorer': 'gene_mask_lfc'},
    8:  {'output_type': 'RNA_SEQ_ACTIVE',    'scorer': 'gene_mask_active'},
    9:  {'output_type': 'SPLICE_SITES',      'scorer': 'gene_mask_splicing'},
    10: {'output_type': 'SPLICE_SITE_USAGE', 'scorer': 'gene_mask_splicing'},
    11: {'output_type': 'SPLICE_JUNCTIONS',  'scorer': 'splice_junction'},
    12: {'output_type': 'POLYADENYLATION',   'scorer': 'gene_mask_lfc'},
    13: {'output_type': 'ATAC_ACTIVE',       'scorer': 'center_mask_active'},
    14: {'output_type': 'DNASE_ACTIVE',      'scorer': 'center_mask_active'},
    15: {'output_type': 'CHIP_TF_ACTIVE',    'scorer': 'center_mask_active'},
    16: {'output_type': 'CHIP_HISTONE_ACTIVE', 'scorer': 'center_mask_active'},
    17: {'output_type': 'CAGE_ACTIVE',       'scorer': 'center_mask_active'},
    18: {'output_type': 'PROCAP_ACTIVE',     'scorer': 'center_mask_active'},
}


AGING_TISSUES = {
    'brain':   ['brain', 'cortex', 'hippocampus', 'cerebellum', 'frontal'],
    'heart':   ['heart', 'ventricle', 'cardiac', 'aorta'],
    'liver':   ['liver', 'hepatocyte'],
    'blood':   ['blood', 'monocyte', 'T cell', 'B cell', 'NK cell',
                'neutrophil', 'macrophage', 'PBMC'],
    'kidney':  ['kidney', 'renal'],
    'muscle':  ['muscle', 'skeletal muscle', 'psoas', 'myotube'],
    'adipose': ['adipose', 'fat'],
    'skin':    ['skin', 'keratinocyte', 'fibroblast', 'foreskin'],
    'colon':   ['colon', 'sigmoid', 'transverse colon', 'intestin'],
    'lung':    ['lung', 'bronch', 'alveol'],
}


def classify_tissue(biosample_name):
    name_lower = str(biosample_name).lower()
    for category, keywords in AGING_TISSUES.items():
        for kw in keywords:
            if kw.lower() in name_lower:
                return category
    return 'other'


def extract_scores_from_result(adata, result_idx, rsid):
    """Extract a tidy list of row-dicts from one AnnData result."""
    info = RESULT_MAP.get(result_idx, {})
    output_type = info.get('output_type', f'unknown_{result_idx}')
    scorer = info.get('scorer', 'unknown')

    rows = []
    has_genes = 'gene_name' in adata.obs.columns
    n_obs, n_var = adata.shape

    if n_obs == 0 or n_var == 0:
        return rows

    for var_idx in range(n_var):
        tissue = (adata.var['biosample_name'].iloc[var_idx]
                  if 'biosample_name' in adata.var.columns else 'unknown')
        ontology = (adata.var['ontology_curie'].iloc[var_idx]
                    if 'ontology_curie' in adata.var.columns else '')

        extra = {}
        if 'histone_mark' in adata.var.columns:
            extra['histone_mark'] = adata.var['histone_mark'].iloc[var_idx]
        if 'transcription_factor' in adata.var.columns:
            extra['transcription_factor'] = adata.var['transcription_factor'].iloc[var_idx]

        if has_genes:
            for obs_idx in range(n_obs):
                score = float(adata.X[obs_idx, var_idx])
                gene_name = adata.obs['gene_name'].iloc[obs_idx]
                rows.append({
                    'rsid': rsid,
                    'output_type': output_type,
                    'scorer': scorer,
                    'gene_name': gene_name,
                    'biosample_name': tissue,
                    'ontology_curie': ontology,
                    'raw_score': score,
                    **extra,
                })
        else:
            score = float(adata.X[0, var_idx])
            rows.append({
                'rsid': rsid,
                'output_type': output_type,
                'scorer': scorer,
                'gene_name': '',
                'biosample_name': tissue,
                'ontology_curie': ontology,
                'raw_score': score,
                **extra,
            })

    return rows
