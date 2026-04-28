# Dataset guide: LIBERO, RLDS, HuggingFace, and ReconVLA

This note explains **which robot manipulation data** LA-ReconVLA uses in our experiment plan, **why**, **where to get it**, **approximate size**, **streaming vs download**, and how this relates to **ReconVLA** [1].

---

## 1. What we use in this project

| Role | Dataset / format | In our pipeline |
|------|------------------|-----------------|
| Primary benchmark & training demos | **LIBERO** (simulation), **LIBERO-Spatial** suite | `configs/base.yaml` → `data.libero_variant: spatial` |
| Convenient HuggingFace mirror (RLDS) | **`openvla/modified_libero_rlds`** | `libero_spatial_no_noops` split |
| Optional native LIBERO format | HDF5 demos via **LIBERO** Python package | Local `data/libero_spatial/` after official download |

Our Part 2 / Colab plan targets **3 tasks × 50 demos** for speed; the **full** LIBERO-Spatial suite is larger (see §4).

---

## 2. What LIBERO is

**LIBERO** (Lifelong Robot Learning Benchmark) is a **simulated** manipulation benchmark with multiple task suites [2]. Each suite is a set of language-conditioned tasks with human-collected demonstrations.

Suites commonly referenced in papers:

| Suite | Focus (informal) |
|-------|-------------------|
| LIBERO-Spatial | Spatial relations (e.g., put object in drawer) |
| LIBERO-Object | Object generalisation |
| LIBERO-Goal | Goal-conditioned behaviour |
| LIBERO-Long | Long-horizon |
| LIBERO-100 / aggregated | Broader coverage |

**Why we use LIBERO-Spatial**

- Public, reproducible, no real robot required.
- Matches the **evaluation** setting cited in ReconVLA [1] (among others).
- Tasks mix **single-object** and **relational** manipulation, which matters for attention-based masking (see `experiment/01_theoretical_analysis.md`).

---

## 3. HuggingFace: `openvla/modified_libero_rlds`

**URL:** [https://huggingface.co/datasets/openvla/modified_libero_rlds](https://huggingface.co/datasets/openvla/modified_libero_rlds)

**What it is**

- Robot demonstrations in **RLDS** (TensorFlow Datasets–style episodic format), **modified for OpenVLA** fine-tuning.
- **“no_noops”** variants strip no-op actions so trajectories are denser in meaningful steps (better for imitation learning).

**Relevant config names** (typical)

- `libero_spatial_no_noops` — **our default** for LA-ReconVLA experiments aligned with “LIBERO-Spatial”.
- `libero_object_no_noops`, `libero_goal_no_noops`, `libero_10_no_noops` — other suites in the same repo.

**Rough sizes** (order of magnitude; HF may update)

- A single split such as `libero_spatial_no_noops` is often on the order of **hundreds of MB to a few GB** depending on resolution and compression.
- Full `modified_libero_rlds` multi-split download can be **several GB**.

Always check the dataset card on HuggingFace for current shard sizes.

---

## 4. Stream vs download locally

| Approach | Pros | Cons |
|----------|------|------|
| **Download / snapshot** (`snapshot_download`, or TFDS cache) | Stable offline training; predictable Colab session behaviour | Disk usage; first-time wait |
| **Streaming** (`datasets.load_dataset(..., streaming=True)`) | Low upfront disk; good for inspection | Slower random access; harder reproducibility unless you fix shuffle seeds and iteration order |

**Recommendation for this project**

- **Development / quick inspection:** streaming or a **tiny slice** (first N episodes).
- **Actual training runs (20 epochs, 3×50 demos):** **local snapshot** of the subset you need, or HDF5 subset from LIBERO — avoids mid-training network stalls.

Script: `scripts/download_data_hf.py` (snapshot or list files).

---

## 5. Official LIBERO (non-HF) path

**Repo:** [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)

- Demos are often distributed as **HDF5** per task.
- The **Part 2 implementation guide** (`doc/PART2_IMPLEMENTATION_GUIDE.md`) assumes paths like `data/libero_spatial/{task}_demo.hdf5`.

**When to prefer HDF5**

- You already use **robosuite** / LIBERO env for rollouts.
- You want exact alignment with **LIBERO** evaluation harness.

**When to prefer HF RLDS**

- You align with **OpenVLA** fine-tuning tooling.
- You want a single `datasets`-style loader.

Our **YAML config** can point to either `data.source: huggingface` or `data.source: libero_hdf5` once the dataset module is implemented; the plans per experiment stay the same.

---

## 6. Our filtration / subset (experiment protocol)

From `experiment/02_experiment_plan.md`:

| Filter | Value |
|--------|--------|
| Suite | LIBERO-Spatial |
| Tasks | 3 fixed tasks (T1 place, T2–T3 single-object) |
| Demos per task | 50 (pilot); full suite ≈ ~130 demos/task |
| Train/val split | 85% / 15% **within each task’s demos** |
| Seeds | 42, 123, 7 |

**Task IDs** (string keys in config)

1. `KITCHEN_SCENE1_put_the_black_bowl_in_the_top_drawer_of_the_cabinet`
2. `KITCHEN_SCENE2_open_the_bottom_drawer_of_the_cabinet`
3. `KITCHEN_SCENE3_turn_on_the_stove`

**Why these three**

- T1 stresses **two-region** grounding (object + placement).
- T2–T3 stress **single-region** grounding.
- Lets you read per-task metrics (see `experiment/04_evaluation_benchmarking.md`).

---

## 7. How ReconVLA used LIBERO (and related data)

From ReconVLA [1] (as summarised in your Part 1 report and `hypothesis.md`):

- **Pretraining** combines large-scale robot data: **BridgeData V2**, **LIBERO**, **CALVIN**, totalling on the order of **100k+ trajectories** and **~2M samples**.
- **Evaluation** includes **LIBERO-Spatial**, **LIBERO-Long**, and **CALVIN** benchmarks.

**Implication for your work**

- You are **not** reproducing ReconVLA’s full pretraining (8× A100 scale).
- You **are** using the **same benchmark family** (LIBERO-Spatial) to test whether **language-attention masked reconstruction** helps under a **small-data, single-GPU** regime — a deliberate scope choice (see `feedback.md`).

---

## 8. Citations

- [1] W. Song et al., "ReconVLA: Reconstructive vision-language-action model as effective robot perceiver," arXiv:2508.10333, 2025. Available: https://arxiv.org/abs/2508.10333
- [2] LIBERO benchmark — see official LIBERO repository and associated publications linked from the project page.
- [3] M. J. Kim et al., "OpenVLA: An open-source vision-language-action model," arXiv:2406.09246, 2024. (HuggingFace dataset `openvla/modified_libero_rlds`.)

---

## 9. Quick reference commands

```bash
# List HF dataset splits (requires huggingface_hub)
python scripts/download_data_hf.py --list-only

# Snapshot spatial split to ./data/hf_libero_spatial (for local training)
python scripts/download_data_hf.py --split libero_spatial_no_noops --local-dir ./data/hf_libero_spatial
```

See `scripts/download_data_hf.py` for implementation details.
