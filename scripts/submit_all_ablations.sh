#!/usr/bin/env bash
# Submits the experiment grid as separate Slurm jobs through scripts/job.sbatch (one GPU each, so they run in
# parallel where the queue allows). The paraphrase caches are built first; every other job waits for that one.
#
#   bash scripts/submit_all_ablations.sh                 # from the repository root on xlogin
#   GPU=a100-40 bash scripts/submit_all_ablations.sh     # other GPU type (see `gpufree` / docs/SoC_Cluster_Guide.pdf)
#   SKIP_OT=1 bash scripts/submit_all_ablations.sh       # leave out the optimal-transport ablations
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-h100-47}
mkdir -p logs/slurm runs weights

PARA=$(sbatch --parsable --job-name=g17_para --time=02:00:00 --gpus="$GPU:1" \
  --export=ALL,STAGES=paraphrases scripts/job.sbatch)
echo "paraphrase caches     -> job $PARA"

submit() {  # submit NAME MODEL RUN_PREFIX MODES [KL_LAMBDAS]
  sbatch --parsable --dependency=afterok:"$PARA" --job-name="$1" --gpus="$GPU:1" \
    --export=ALL,MODEL="$2",RUN_PREFIX="$3",MODES="$4",KL_LAMBDAS="${5:-}",STAGES="train reranker eval" scripts/job.sbatch
}
echo "bert ce               -> job $(submit g17_bert_ce  bert-base-uncased bert ce)"
echo "bert aug              -> job $(submit g17_bert_aug bert-base-uncased bert aug)"
for L in ${KL_LAMBDAS:-0.5 1.0 2.0}; do   # one job per lambda so each stays well inside the time limit
  echo "bert kl lambda=$L     -> job $(submit "g17_bert_kl_lam$L" bert-base-uncased bert kl "$L")"
done
echo "deberta ce            -> job $(submit g17_deb_ce   microsoft/deberta-v3-base deb ce)"
echo "deberta kl            -> job $(submit g17_deb_kl   microsoft/deberta-v3-base deb kl)"
if [ -z "${SKIP_OT:-}" ]; then
  echo "bert ot               -> job $(submit g17_bert_ot   bert-base-uncased bert ot)"
  echo "bert dual             -> job $(submit g17_bert_dual bert-base-uncased bert dual)"
fi
echo
echo "monitor:  squeue --me        logs: tail -f logs/slurm/<job-name>-<job-id>.out"
echo "results:  runs/<run>/eval_<set>/results.json ; then  python visualize.py --best <run>  (figures stage)"
