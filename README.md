# pnn-citation-rec — notebooks

Companion notebooks for the PNN final-project report. Each notebook is
self-contained: it loads the cached data, defines its own encoder and
decoder classes, trains, and prints test metrics in the cell outputs.

## Setup

Python 3.10 or newer; a CUDA-capable GPU is needed for any of the
training notebooks (`10_*` through `24_*`).

```bash
# uv (recommended)
uv sync
source .venv/bin/activate

# or pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install jupyterlab matplotlib pandas seaborn scikit-learn tqdm
```

## Run order

`00_data_preparation.ipynb` must be run once to build the train, val,
and test caches. After that, every other notebook can be run
independently.

```
00 ─→ (any of)  01, 02, 03, 04, 1*, 2*
```

Numeric prefixes encode the layout, not a strict dependency:

- `0*` — data preparation, dataset analysis, figure generation, no-graph baselines
- `1*` — homogeneous GNN encoders (GraphSAGE, GAT, GATv2)
- `2*` — heterogeneous GNN encoders and the R-GAT control

## Notebook index

| File | What it does |
|---|---|
| `00_data_preparation.ipynb` | Downloads/loads OAG-CS, builds per-year graphs and supervision examples, saves `processed/*.pt` caches |
| `01_data_analysis.ipynb` | Dataset stats, sanity checks |
| `02_figure_eda_section2.ipynb` | Generates the dataset figure used in the report |
| `03_figure_training_example.ipynb` | Generates the training-example figure used in the report |
| `04_baselines_cosine_mlp.ipynb` | Cosine-similarity and MLP-only baselines (no graph) |
| `10_graphsage_2L.ipynb` | GraphSAGE, 2 layers |
| `11_gat_2L.ipynb` | GAT, 2 layers |
| `12_gat_3L.ipynb` | GAT, 3 layers (multi-seed) |
| `13_gat_4L.ipynb` | GAT, 4 layers (multi-seed) |
| `14_gat_5L.ipynb` | GAT, 5 layers |
| `15_gatv2_4L.ipynb` | GATv2, 4 layers |
| `20_hetgat_2L.ipynb` | HetGAT (per-relation + HAN softmax), 2 layers |
| `21_hetgat_3L.ipynb` | HetGAT, 3 layers |
| `22_hetgat_4L.ipynb` | HetGAT, 4 layers (multi-seed) |
| `23_hetgat_5L.ipynb` | HetGAT, 5 layers (multi-seed) |
| `24_rgat_3L.ipynb` | R-GAT (per-relation + unweighted-sum aggregator), 3 layers |

## Multi-seed runs

Notebooks marked "multi-seed" in the table were run with `SEED ∈ {42, 43, 44}`
to produce the `mean ± std` numbers. The default value in each notebook is
seed 42; change `SEED` at the top of the config cell to reproduce the others.
