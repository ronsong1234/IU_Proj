# PneumoniaMNIST: metadata pipeline, analysis and baseline classifier

A small, reproducible pipeline that turns the PneumoniaMNIST `.npz` into a queryable
per-image metadata table (DuckDB), validates it, analyses it with SQL, and evaluates
a simple scikit-learn baseline for normal vs. pneumonia.

```
README.md
requirements.txt
Notebooks/
    01_exp.ipynb                end-to-end walkthrough: every step, output, figure and commentary
src/
    download_data.py            download the dataset from the official Zenodo record
    build_metadata.py           .npz -> per-image metadata table (DuckDB); runs validation + export
    validate.py                 data-quality checks, results stored in DuckDB
    export_results.py           result tables -> small CSVs in output/
    baseline.py                 logistic-regression baseline (train / val / test protocol)
sql/
    split_summary.sql           images, normal, pneumonia, % pneumonia per split
    intensity_by_class.sql      normal vs pneumonia: distribution of per-image statistics
    intensity_effect_size.sql   difference and Cohen's d, per split
    unusual_images.sql          unusual images by robust z-score
output/                         committed results (CSV + PNG), readable without running anything
    validation_results.csv      one row per validation check
    duplicate_images.csv        groups of identical images
    split_summary.csv           }
    intensity_by_class.csv      }  results of the matching sql/ queries
    intensity_effect_size.csv   }
    unusual_images.csv          }
    model_metrics.csv           test metrics + "always pneumonia" reference
    intensity_by_class.png      normal vs pneumonia intensity distributions
    unusual_images.png          unusual images next to typical ones
    baseline_evaluation.png     test confusion matrix + ROC curves
    pneumoniamnist.duckdb       generated database (git-ignored)
data/
    pneumoniamnist.npz          downloaded dataset (git-ignored, never committed)
```

## How to run it

Tested with Python 3.13 on Windows. All commands run from the repo root.

**1. Install dependencies**

```
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**2. Obtain the dataset** (PneumoniaMNIST 28x28, official MedMNIST+ record,
https://doi.org/10.5281/zenodo.10519652)

```
python -m src.download_data       # -> data/pneumoniamnist.npz (4.2 MB); skipped if present
```

Alternatives: download `pneumoniamnist.npz` from the Zenodo record by hand into `data/`,
or use the official `medmnist` package (`pip install medmnist`; not in our requirements
because it pulls in PyTorch).

**3. Run the pipeline** (build the metadata table, run the validation checks, export CSVs)

```
python -m src.build_metadata      # -> output/pneumoniamnist.duckdb + output/*.csv; prints check results
python -m src.validate            # optional: re-run the checks alone
```

**4. Run the analysis and model**

```
python -m src.baseline            # model selection on val, one final test evaluation -> output/model_metrics.csv
```

The full analysis (SQL queries, figures, commentary) is in `Notebooks/01_exp.ipynb`.
Open it in VS Code or Jupyter with the `.venv` interpreter as the kernel and
**Run All**. The notebook expects to run from `Notebooks/` (it sets `ROOT` to the
parent folder) and repeats steps 2–4 itself, so it also works on a fresh clone.
Any SQL file can also be run directly:

```
python -c "import duckdb; print(duckdb.connect('output/pneumoniamnist.duckdb').sql(open('sql/split_summary.sql').read()).df())"
```

## Approach

### Data model

The `.npz` holds six arrays: `{train,val,test}_images` (N x 28 x 28, `uint8`) and
`{train,val,test}_labels` (N x 1, values 0/1). The pipeline discovers splits from these
key names and fails loudly on unpaired or unexpected arrays rather than assuming them.

One DuckDB file, `output/pneumoniamnist.duckdb`, with one row per image in
`image_metadata`. The pixels stay in the `.npz`; the table points back to them via
`(split, split_index)`.

| Column | Description |
|---|---|
| `image_id` | `<split>_<index>`, e.g. `train_00042` (primary key) |
| `split`, `split_index` | source split and position in that split's array |
| `label`, `label_name` | 0 = `normal`, 1 = `pneumonia` |
| `width`, `height`, `channels`, `dtype` | geometry and pixel type |
| `mean_intensity`, `std_intensity`, `min_intensity`, `max_intensity`, `median_intensity` | per-image pixel statistics (0–255) |
| `frac_black`, `frac_white` | fraction of pixels at 0 / 255 |
| `pixel_sha1` | SHA-1 of the raw pixel bytes, for exact-duplicate detection |
| `source_file` | source `.npz` name |

Validation writes three more tables: `validation_results` (one row per check),
`low_variance_images` and `duplicate_images` (one row per group of identical images).

### Important assumptions

- The label mapping 0 = normal, 1 = pneumonia comes from the MedMNIST dataset info; the
  `.npz` itself does not record it.
- Expected split sizes (4,708 / 524 / 624) are the official PneumoniaMNIST sizes.
- `image_id` is positional. It is stable only as long as the `.npz` does not change.
- The images are already preprocessed by MedMNIST (resized to 28x28, 8-bit). We analyse
  them as given; statistics say nothing about the original radiographs.
- Each image is treated as independent. There are no patient IDs to check this.

### Validation checks

Run by `src/validate.py`. The first four checks FAIL the run if violated; the last two
are reported as WARN, for human review.

| Check | Rule | Result |
|---|---|---|
| split counts | train/val/test = 4708/524/624 | PASS |
| label values | only {0, 1}; every split has both classes | PASS |
| image dimensions | raw arrays are 28x28, single channel | PASS |
| missing / invalid | no nulls or duplicate IDs; stats finite, in 0–255, min ≤ mean ≤ max; arrays `uint8` | PASS |
| low variance | `std_intensity` < Q1 − 1.5·IQR (= 14.2) | WARN: 46 images |
| exact duplicates | identical `pixel_sha1` | WARN: 27 groups (56 images); **8 groups span train and val**; no label conflicts |

Each FAIL check was tested by corrupting data on purpose (bad label, NaN, out-of-range
value, missing rows).

### Analytical findings

- **Class balance** (`sql/split_summary.sql`): pneumonia is 74.2% of train, 74.2% of val,
  but only **62.5% of test**.
- **Normal vs pneumonia** (`sql/intensity_by_class.sql`, `sql/intensity_effect_size.sql`):
  pneumonia images are slightly brighter (mean 147.6 vs 140.2, Cohen's d = 0.38). The gap
  shrinks from train (d 0.41) to test (d 0.20). They have much **lower contrast** (std 34.4
  vs 44.1, d = −1.21), and this holds in every split. Pneumonia images also span a
  narrower intensity range (higher minimum, lower maximum).
- **Unusual images** (`sql/unusual_images.sql`): a robust z-score (median/MAD, with
  |z| > 3.5) on mean and std intensity flags 32 images (0.5%). These are 10 very dark,
  1 very bright, 1 very low-contrast and 21 very high-contrast. The most extreme,
  `test_00165`, is an X-ray inset in a thick black border, which looks like an
  acquisition or cropping artefact. Figure: `output/unusual_images.png`.

### Model results

Logistic regression on the 784 standardised pixels, with `class_weight="balanced"`
(`src/baseline.py`).

- **Train** fits the model.
- **Validation** picks C ∈ {0.001, 0.01, 0.1, 1} by ROC AUC (C = 0.01), then the
  threshold by Youden's J (0.588). The 8 validation images duplicated in train are
  excluded from this.
- **Test** is evaluated once.

| | Accuracy | Sensitivity | Specificity | Balanced acc. | ROC AUC |
|---|---|---|---|---|---|
| **Test** | 0.885 | 0.964 | 0.752 | 0.858 | 0.930 |
| Always predict pneumonia (test) | 0.625 | 1.000 | 0.000 | 0.500 | 0.500 |
| Validation (reference) | 0.950 | 0.937 | 0.985 | 0.961 | 0.992 |

Test confusion matrix: of 234 normal images, 176 were correct and 58 were false
positives; of 390 pneumonia images, 376 were correct and 14 were missed. Figure:
`output/baseline_evaluation.png` (confusion matrix and ROC curves).

**Why accuracy alone misleads:** a model that always answers "pneumonia" gets 62.5% test
accuracy with 0% specificity. Accuracy weights each error type by how common its class
is. It therefore hides which class is failing (here, a quarter of normal images are
false alarms), and it changes with prevalence even when the model does not.

### Important limitations

- **Validation overstates performance.** Test AUC is 0.930 against 0.992 on validation.
  In MedMNIST, val is split off the same source data as train, while test comes from a
  separate source set. The test figures are the honest estimate.
- **Shortcut risk.** A logistic regression on just five global statistics (mean, std,
  min, max and median intensity) reaches test AUC 0.893, close to the pixel model's
  0.930. Much of the signal is global brightness and contrast, which may reflect how
  images were acquired rather than the disease (see Part 4, question 4).
- 28x28 images discard most diagnostic detail. The results say nothing about clinical
  usefulness.
- Duplicate detection is exact only. Near-duplicates (re-crops or re-encodes) are not
  caught.
- No patient, site or date metadata, so patient-level leakage cannot be ruled out
  (Part 4, question 1).
- Only one model, one seed, and no confidence intervals. With 234 normal test images,
  specificity has a 95% CI of roughly ±5.5 percentage points.

## Part 4: Research / data engineering judgment

### 1. Data leakage

What I would worry about in medical imaging:

- **Patient-level leakage.** The same patient, or several studies from one patient,
  lands in both train and test. The model then recognises the patient, not the disease.
  This is the most common and most damaging kind.
- **Duplicate or near-duplicate images across splits.** These are the same scan,
  re-exported, re-cropped or resized.
- **Acquisition shortcuts.** Label correlates with hospital, scanner, protocol,
  patient age, view (AP vs PA), burned-in text or markers, or borders. The model then
  learns *where or how* an image was taken.
- **Label leakage via preprocessing.** Examples: per-class preprocessing pipelines,
  or labels derived from reports the model indirectly sees.
- **Evaluation leakage.** Fitting scalers on all data, or picking thresholds or models
  on the test set. We avoided both: scaling is fitted on train inside the pipeline, and
  selection uses val only.

**Can this dataset rule them out? No.** We can detect exact duplicates (8 train/val
groups found; none touch test). The `.npz` contains only pixels and labels, with no
patient ID, study ID, site, scanner, view or date. So patient-level leakage and
site/acquisition confounding cannot be ruled out. The source dataset's file names carry
patient identifiers for some images, but these don't survive into the `.npz`. The strong
class difference in global contrast is consistent with an acquisition shortcut, and
the data cannot tell us which explanation is true.

### 2. Scaling to 1,000,000 images from several hospitals

1. **Process incrementally and in a streaming fashion, keyed by content hash.** Today the
   pipeline loads one `.npz` into memory and rebuilds the table with `CREATE OR REPLACE`.
   Instead, process images in batches as they arrive. Compute each image's metadata once,
   keyed by a hash of the decoded pixels, and append to partitioned Parquet (by hospital
   and ingest date), with DuckDB as the query engine on top. This also means handling
   real inputs, such as DICOM at varying sizes and 12–16-bit depth, rather than assuming
   28x28 `uint8`.
2. **Record provenance and join keys from day one.** Store hospital, device, a
   pseudonymised patient ID, study date, source URI, pipeline version and processing
   status per image. Without patient and site IDs you cannot build leakage-safe
   (patient- or site-grouped) splits, audit results, or debug one hospital's bad feed.
   It is much harder to add these later.
3. **Make validation per batch and per source, and quarantine failures.** The current
   checks assume one fixed dataset (for example, hard-coded split sizes). At scale, run
   schema and value checks on each batch, and track per-hospital distributions of
   intensity, label rate and image size over time to catch drift. Send failing images to
   a quarantine table with the error, instead of failing the whole run.

### 3. Incremental processing (10,000 new images tomorrow)

Keep a **processing ledger** table with one row per unique image:

- a content hash (SHA-256 of the decoded pixel array plus its shape and dtype, so
  identical pixels match even if file headers differ);
- a list of source URIs;
- `status` (`ok` / `failed`), error message and attempt count;
- the pipeline version that produced the metadata;
- timestamps.

Metadata rows reference the hash. For each incoming image:

- **New image:** its hash is not in the ledger. Compute the metadata, append it, and
  mark it `ok`. The existing 1,000,000 rows are never touched. The only work is one hash
  lookup per new image, plus metadata for genuinely new images.
- **Exact duplicate:** its hash is already present with status `ok`. Don't recompute;
  just add the new source URI to that entry. We still keep the duplicate as a fact,
  because a duplicate that lands in both train and test is a leakage problem (and
  duplicates with conflicting labels are a label-quality problem).
- **Previously failed:** the ledger has the hash, or the source URI if decoding failed
  before hashing, with status `failed`. Retry these, with a capped attempt count and the
  last error kept.

If the metadata code changes, bump the pipeline version and recompute only the rows
with an older version.

### 4. A run that succeeds but misleads

**Scenario: the classifier learns acquisition differences instead of disease.** In this
dataset the classes differ strongly in global contrast (d = −1.21). A model given only
five summary statistics already reaches test AUC 0.893, against 0.930 for all pixels.
If normal and pneumonia images partly come from different sources, scanners or
protocols, the pipeline would run cleanly, all validation checks would pass, and the
AUC would look good. Yet the model could fail entirely at a hospital whose images look
different. Something like this is already visible: performance drops from validation
(AUC 0.992) to the differently sourced test set (0.930).

**Detect it:**

- Compare against a "trivial features" baseline, as we did: if global statistics alone
  get close to the full model, be suspicious.
- Report performance separately for each hospital, scanner and time period, not just
  pooled.
- Check whether hospital or scanner can itself be predicted from the images.
- Inspect misclassified images and saliency maps for borders, text or markers.

**Prevent it:**

- Split by patient and hold out whole sites for testing.
- Balance or match class composition across sources.
- Standardise intensities per source.
- Require an external, differently sourced test set before claiming performance.

## AI assistance

1. **Tool:** Claude Code (Anthropic's coding assistant, Claude Opus 5.5 model) in VS Code.
2. **What for:** setting up the environment and fixing the dataset download; writing the
   pipeline, validation, SQL queries, baseline model and plotting code; drafting parts
   of this README. I directed each step, ran the code, and reviewed the outputs and
   figures.
3. **Checked, changed or rejected:**
   - *Rejected:* the assistant's first plan was to install `requirements.txt` as it
     was, including `medmnist`, which pulls in PyTorch (hundreds of MB) just to download
     one 4 MB file. I rejected this and had it remove `medmnist`. The dataset is now
     downloaded directly from the official Zenodo record with the standard library
     (`src/download_data.py`).
   - *Checked:* the assistant's first diagnosis of the HTTP 403 download error was a
     server-side block, and it switched to a different Zenodo URL. Testing showed the
     real cause was the fake browser `User-Agent` header in the download code; the
     original URL works without it. The fix and a comment explaining it are in
     `src/download_data.py`.
