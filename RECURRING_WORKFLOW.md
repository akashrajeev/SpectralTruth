# Recurring Model Improvement Workflow

Use `scripts/recurring_workflow.py` to run a full recurring improvement cycle:

1. Evaluate current model and collect misclassified clips.
2. Group failures by language, generator, speaker, and artifact type.
3. Merge curated examples into the training dataset manifest.
4. Retrain candidate model and compare with previous production model on fixed benchmark.
5. Promote candidate only when both **macro-F1** and **worst-language recall** improve.

## Required manifest format (CSV)

All manifest files (`evaluation`, `benchmark`, `dataset`, `curated`) must include:

- `path`
- `label` (`ai` or `human`)
- `language`
- `generator`
- `speaker`
- `artifact_type`

## Example

```bash
python scripts/recurring_workflow.py \
  --run-name weekly-audio-refresh \
  --current-model model/model-1.h5 \
  --previous-model model/model-1.h5 \
  --evaluation-manifest manifests/global_eval.csv \
  --benchmark-manifest manifests/fixed_benchmark.csv \
  --dataset-manifest manifests/train_dataset.csv \
  --curated-manifest manifests/curated_additions.csv \
  --retrain-command "python scripts/retrain.py --dataset {dataset_manifest} --base {base_model} --out {output_model}" \
  --promote-model-path model/model-promoted.h5
```

If `--retrain-command` is omitted, the workflow copies `--current-model` into the candidate slot so logging and promotion checks still run.

## Outputs

Each run writes to:

`experiments/YYYY-MM-DD-<run-name>/`

Artifacts include:

- `global_evaluation.csv`
- `misclassified_clips.csv`
- `failure_tags.json`
- `dataset_with_curated.csv`
- `candidate_model.h5`
- `metrics.json`

`metrics.json` includes benchmark deltas and promotion decision.
