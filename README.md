cat > README.md << 'EOF'
# Potential evaporation and the Bouchet–Morton complementary relationship

This repository contains the code and (processed) plot data needed to reproduce the figures and tables in the accompanying HESS manuscript.

## Repository contents
- `paper/`: submitted manuscript PDF and submission text (abstract, short summary)
- `src/`: model + analysis code
- `scripts/`: scripts to regenerate figures/tables
- `data/`: processed plot data (raw data are obtained separately; see `data/README.md`)
- `outputs/`: optional pre-generated figures/tables

## Quickstart
1. Create a Python environment (see `environment/` or `requirements.txt`).
2. Install requirements.
3. Run:
   - `python scripts/99_make_all.py`

## Data access
See `data/README.md` for how to obtain the input data used in the manuscript.

## Citation
See `CITATION.cff` (and Zenodo DOI once minted).
EOF

