# How Much is Left? LLMs Linearly Encode Their Remaining Output Length

**Mohamed Amine Merzouk**<sup>1,2,\*</sup>, **Dmitri Carpov**<sup>3,\*</sup>, **Mirko Bronzi**<sup>3,\*</sup>, **Damiano Fornasiere**<sup>3,⋄</sup>, **Adam Oberman**<sup>1,2,3,⋄</sup>
<br><sup>1</sup> Mila – Quebec AI Institute · <sup>2</sup> McGill University · <sup>3</sup> LawZero
<br><sup>\*</sup> Equal contribution · <sup>⋄</sup> Equal advising

**NeurIPS 2026** · [Paper (arXiv)](https://arxiv.org/abs/2607.05316)

Large language models generate one token at a time, yet their responses have consistent lengths. We ask whether a model carries an internal estimate of how much of its response remains. We train single-linear-layer probes on the frozen hidden states of open-weight instruction-tuned models (Llama-3.1-8B, Olmo-3-7B and Mistral-7B, plus Qwen3-4B and Qwen3-32B) across seven datasets and find that:

- total response length is linearly decodable from the prompt's last hidden state, before any token is generated;
- probes trained on natural-language data transfer to other datasets, while probes trained on synthetic counting data do not;
- the probe's estimate jumps upward when the model retracts a partial solution ("Wait");
- adding the probe direction to the residual stream lengthens or shortens generations in a dose-dependent way.

<p align="center">
  <img src="assets/overview.png" width="720" alt="Method overview: a frozen LLM reads the prompt and its own response; a linear probe on the concatenated hidden states of all layers predicts the total response length at the end of the prompt (Completion Length Probe) and the number of tokens left at every response token (Remaining Count Probe).">
</p>

## Installation

Python 3.12 and a CUDA GPU are required.

```bash
git clone https://github.com/mamerzouk/llm-output-length.git
cd llm-output-length
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

`requirements.txt` pins the exact versions this release was tested with, including PyTorch 2.9.1 from PyPI (built for CUDA 12.8). For another CUDA version, install `torch==2.9.1` from the [PyTorch index](https://pytorch.org/get-started/locally/) first. `pip install -e ".[dev]"` on its own installs the latest compatible versions instead.

**Gated models.** Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3 require accepting their licenses on Hugging Face. Then authenticate with `huggingface-cli login` or by setting `HF_TOKEN`.

**Weights & Biases.** Training logs per-probe curves to W&B (project `token-count-prediction`). Run `wandb login`, set `WANDB_MODE=offline`, or pass `wandb.enabled=false`. The evaluation scripts below do not need W&B.

Run the unit tests with `pytest tests/`.

## Data

No manual download is needed. The extraction script loads each dataset from the Hugging Face Hub and formats every prompt with the model's chat template and the system prompt in `src/miol/conf/data/<dataset>.yaml`.

| Dataset | Config | Source | Train / eval |
|---|---|---|---|
| Count | `count` | synthetic: "Count from 0 to {n}…", n ∈ 1..300 | 299 / 299 |
| Countdown | `countdown` | synthetic: "Count down from {n} to 0…", n ∈ 1..300 | 299 / 299 |
| GSM8K | `gsm8k` | `openai/gsm8k` (main), train / test | 7,473 / 1,319 |
| MATH | `math` | `EleutherAI/hendrycks_math`, 7 subjects, train / test | 7,500 / 5,000 |
| MMLU-Pro | `mmlu_pro` | `TIGER-Lab/MMLU-Pro` test: first 9,776 / last 2,256 | 9,776 / 2,256 |
| OpenThoughts-1k | `openthoughts-1k` | `ryanmarten/OpenThoughts-1k-sample`: first 800 / last 200 | 800 / 200 |
| TriviaQA | `trivia_qa` | `mandarjoshi/trivia_qa` (rc.nocontext): first 10,000 of train / last 2,000 of test | 10,000 / 2,000 |

Models (`src/miol/conf/model/`): `llama3.1_8b`, `olmo3_7b`, `mistral_7b`, `qwen3_4b` and `qwen3_32b`. All run in non-thinking mode.

## Usage

**Hidden states and probes are not distributed. Every step below regenerates them.** Completions are sampled (T = 0.7, top-p 0.8, top-k 20; Olmo-3 uses its own T = 0.6, top-p 0.95), so regenerated caches and numbers will differ slightly from the paper's.

The pipeline is: extract hidden states → train probes → evaluate → analyses. Commands are run from the repository root, and outputs go under `outputs/`, `evals/` and `reports/`. The examples use one cell; loop over the models, datasets and seeds you need.

```bash
MODEL=llama3.1_8b
DATASET=gsm8k
```

### 1. Extract hidden states

Each prompt gets one sampled completion (up to 1,024 new tokens). The script then runs a frozen forward pass over prompt + completion and saves the residual stream at every layer and position. Only completions that end with EOS keep their tensors; the per-sample `.json` metadata is always written.

```bash
python -m torch.distributed.run --nproc_per_node=gpu src/miol/extract_hidden_states.py \
    model=$MODEL data=$DATASET output_dir=outputs/$MODEL-$DATASET
```

This writes `outputs/$MODEL-$DATASET/hidden_states/{train,eval}/sample_*.{json,safetensors}`. Each GPU holds a full copy of the model.

### 2. Train probes

All probe heads (Remaining Count Probe `count`, Completion Length Probe `prompt_only_count`, the K-way classifiers `cls_K`, and the shuffled-label controls) train together from the cache. The paper uses seeds 0, 1 and 2.

```bash
for SEED in 0 1 2; do
  python -m miol.train_offline model=$MODEL data=hidden_states data.dataset_name=$DATASET \
      data.hidden_states_dir=outputs/$MODEL-$DATASET/hidden_states \
      output_dir=outputs/$MODEL-$DATASET wandb.run_name=$MODEL-$DATASET-$SEED seed=$SEED
done
```

Probes are saved to `outputs/$MODEL-$DATASET/$MODEL-$DATASET-$SEED/seed_$SEED/trained_probe/`. The defaults in `src/miol/conf/config.yaml` are the paper's settings: AdamW at 2e-4, ReduceLROnPlateau on eval MAE, 4,000 steps, batch size 8, evaluation every 100 steps, a checkpoint every 100 steps.

Train on one GPU to match the paper's batch size of 8. Training also runs under `python -m torch.distributed.run --nproc_per_node=gpu`. Probe gradients are then averaged across GPUs, so the effective batch size is 8 × the number of GPUs, and the per-probe W&B metrics cover only the first process's share of the evaluation set. Use `evaluate_probes` (step 3) for reported numbers.

### 3. Evaluate

```bash
# Per-example predictions and metrics for one seed (input to the dashboard and the retraction analysis)
python -m miol.evaluate_probes \
    --probe-dir outputs/$MODEL-$DATASET/$MODEL-$DATASET-0/seed_0/trained_probe \
    --hidden-states-dir outputs/$MODEL-$DATASET/hidden_states \
    --output evals/$MODEL-$DATASET-0.json

# Train-on-A / evaluate-on-B matrix, mean ± std over seeds, plus the ratio to each dataset's median floor
python -m miol.analysis.cross_dataset_eval --model $MODEL \
    --datasets count countdown gsm8k mmlu_pro openthoughts-1k --seeds 0 1 2 \
    --probe count --metric mae \
    --json-dir reports/cross_dataset/$MODEL --output reports/cross_dataset/${MODEL}_count_mae.pdf

# Constant-median (and other) statistical baselines for the same grid
python -m miol.analysis.cross_dataset_baseline_eval \
    --datasets count countdown gsm8k mmlu_pro openthoughts-1k \
    --metadata-pattern "outputs/$MODEL-{dataset}/hidden_states" \
    --output-dir reports/cross_baseline/$MODEL
```

The diagonal of the cross-dataset matrix is the in-distribution result for each dataset. Use `--probe prompt_only_count --metric prompt_mae` for the Completion Length Probe.

### 4. Analyses

```bash
# Aggregate retraction analysis and discovery scan (Qwen3-4B / GSM8K, seed 0)
python -m miol.evaluate_probes --probe-dir outputs/qwen3_4b-gsm8k/qwen3_4b-gsm8k-0/seed_0/trained_probe \
    --hidden-states-dir outputs/qwen3_4b-gsm8k/hidden_states --output evals/qwen3_4b-gsm8k.json
python -m miol.analysis.token_delta_report --eval-json evals/qwen3_4b-gsm8k.json \
    --output reports/token_delta_qwen3_4b-gsm8k.pdf
python -m miol.analysis.retraction_control_analysis --eval-json evals/qwen3_4b-gsm8k.json \
    --output-dir reports/retraction_qwen3_4b-gsm8k

# Steering along the probe direction (Llama-3.1-8B / GSM8K, all 1,319 test prompts, α ∈ {-12..12})
for LAYER in 8 16 24; do
  python -m torch.distributed.run --nproc_per_node=gpu src/miol/analysis/steering_experiment.py \
      --model-name meta-llama/Llama-3.1-8B-Instruct \
      --probe-dir outputs/llama3.1_8b-gsm8k/llama3.1_8b-gsm8k-0/seed_0/trained_probe \
      --steer-index $LAYER --output-dir reports/steering_llama3.1_8b-gsm8k_layer_$LAYER
done

# Sampling-noise floor (10 completions per GSM8K test prompt)
python -m torch.distributed.run --nproc_per_node=gpu src/miol/analysis/sampling_noise_floor.py \
    --model-name meta-llama/Llama-3.1-8B-Instruct --samples-per-prompt 10 --probe-mae 42.29 \
    --output-dir reports/sampling_floor_llama3.1_8b-gsm8k

# Surface-feature ridge baseline and the prompt-end constant-median floor
python -m miol.analysis.surface_feature_baseline --hidden-states-dir outputs/llama3.1_8b-gsm8k/hidden_states \
    --probe-mae 42.29 --output reports/surface_ridge/llama3.1_8b-gsm8k.json

# Exclusion rates of the naturally-terminated filter, for every cell under outputs/
python -m miol.analysis.exclusion_rates --outputs-dir outputs --output reports/exclusion_rates.json

# Per-layer sweep (Qwen3-4B / GSM8K): one count probe per layer, curves logged to W&B as count_L<k>/eval_mae
python -m miol.train_offline model=qwen3_4b data=hidden_states data.dataset_name=gsm8k \
    data.hidden_states_dir=outputs/qwen3_4b-gsm8k/hidden_states output_dir=outputs/qwen3_4b-gsm8k \
    wandb.run_name=qwen3_4b-gsm8k-layers seed=0 data.layers_to_train=all 'optimization.probes=[count]'
```

**Token-by-token explorer.** `src/miol/dashboard/` is a static page for browsing an `evaluate_probes` JSON (per-token predictions, ground truth, sorting by MAE). Serve it with `python -m http.server` from that directory, open `index.html` and load the JSON with the file picker. It loads Chart.js from a CDN.

## Reproducing the paper

`$M` and `$D` stand for a model and dataset config. "cross-dataset" refers to the `cross_dataset_eval` / `cross_dataset_baseline_eval` commands in step 3.

| Paper item | How to reproduce |
|---|---|
| Fig. 1 (overview) | Illustration, not generated by code |
| Table 1: per-token MAE (baseline, exact countdown, Remaining Count Probe) | Steps 1–2 for each cell. Count probe: diagonal of the cross-dataset `count`/`mae` matrix. Baseline: diagonal of `baseline_constant_median__mae_mean.csv`. Exact countdown: W&B `prompt_only_count/eval_mae_static_decay` at step 4,000, or `static_decay_mae_per_token` in the `evaluate_probes` JSON |
| Prompt-end table (Completion Length Probe vs. baseline) | Probe: cross-dataset `--probe prompt_only_count --metric prompt_mae`. Baseline: `median_floor_eval_mae` from `surface_feature_baseline` |
| Relative-MAE tables | The two tables above divided by their baseline rows |
| Section 5 cross-dataset tables, eval-median floor rows, ratio table | Cross-dataset commands per model. Llama and Mistral use 5 datasets; Olmo uses all 7. Ratios are in `*__rel_baseline_constant_median_mae_mean.csv` |
| Pooled transfer-asymmetry test | Off-diagonal ratio cells below 1, natural vs. synthetic training rows; one-sided Fisher exact test |
| Section 4.3 aggregate retraction analysis, histogram, discovery scan | `retraction_control_analysis` and `token_delta_report` (step 4) |
| Retraction example, good/bad examples, retraction gallery | Dashboard views of `evals/qwen3_4b-gsm8k.json`, sorted by count-probe MAE |
| Section 4.4 steering tables and figure | `steering_experiment` at layers 8, 16 and 24 (`steering_results.json` per layer) |
| Section 4.5 Qwen3-32B | Steps 1–3 with `model=qwen3_32b data=gsm8k`, seed 0, and `max_steps=3200` added to the training command. The reported probe is the step-3,200 checkpoint; ReduceLROnPlateau does not depend on the step budget, so stopping there gives the same probe |
| Per-layer probe sweep | Per-layer command in step 4 (W&B `count_L<k>/eval_mae`) |
| MAE by completion length | W&B image `count/mae_per_bin/eval` at step 4,000 of the Llama-3.1-8B / GSM8K run |
| Hyperparameters, dataset sizes, prompt templates | `src/miol/conf/config.yaml`, `src/miol/conf/data/*.yaml` |
| Exclusion rates | `exclusion_rates` (step 4) on the Llama-3.1-8B cells |
| Control experiments: surface ridge | `surface_feature_baseline` (step 4) |
| Control experiments: shuffled labels | Cross-dataset diagonal with `--probe count_shuffled` and `--probe count_shuffled_within` (Llama-3.1-8B / GSM8K) |
| Sampling-noise floor and figure | `sampling_noise_floor` (step 4) |
| Classification probes (Cohen's κ) | `metadata.dataset_cohen_kappa` in the `evaluate_probes` JSON, averaged over seeds |

## Repository structure

```
src/miol/
├── extract_hidden_states.py    # step 1: generate completions, save hidden states
├── train_offline.py            # step 2: train all probe heads from the cache
├── evaluate_probes.py          # step 3: per-example predictions and metrics
├── statistical_baseline.py     # constant / position baselines for one cell
├── interactive_probe_inference.py  # probe predictions for a typed prompt
├── analysis/                   # cross-dataset matrices, retraction, steering, controls, sampling floor
├── conf/                       # Hydra configs: config.yaml, data/, model/, scheduler/
├── data/                       # dataset builders, tokenization, cached hidden-state dataset
├── model/                      # probe heads, multi-probe trainer, losses, generation utilities
└── dashboard/                  # static token-by-token explorer
tests/                          # unit tests
requirements.txt                # pinned dependency versions
```

## Citation

```bibtex
@inproceedings{merzouk2026howmuch,
  title     = {How Much is Left? {LLMs} Linearly Encode Their Remaining Output Length},
  author    = {Merzouk, Mohamed Amine and Carpov, Dmitri and Bronzi, Mirko and Fornasiere, Damiano and Oberman, Adam},
  booktitle = {40th Conference on Neural Information Processing Systems (NeurIPS 2026)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2607.05316}
}
```

## License

MIT; see [LICENSE](LICENSE).
