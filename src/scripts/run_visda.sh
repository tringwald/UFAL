#!/bin/bash
exit_script() {
  echo "Killing child processes."
  trap - INT TERM # clear the trap
  kill -- -$$     # Sends SIGTERM to child/sub processes
}
trap exit_script INT TERM

###############################################################################################################################
_ADD_FLAGS=(
"--epochs" "1"
"--test-every" "1"
"--batch-size" "240"
"--use-lsm-for" "SOURCE"
"--test-batch-size" "400"
"--base-lr" "0.0005"
"--imp-lr" "0.00025"
"--weight-decay" "0.00005"
"--sample-num-instances" "5" "4" "3"
"--sample-num-mini-batches" "50"
"--regen-every" "5"
"--prob-lower-bound" "0.05"
"--mc-iterations" "20"
"--mc-dropout-rate" "0.85"
"--use-filtering"
"--use-feature-loss"
"--use-uncertain-features"
"--imp-epochs" "15"
"--top-k" "12"
"--sampling-mode" "UNCERTAINTY"
"--batch-construction" "SMART_BATCH_LAYOUT"
"--experiment-subdir" "VisDA/final"
"--seed" "1"
"--show-per-class-stats"
)
_COMMENT="rerun"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset VISDA17 --test-dataset VISDA17 --architecture Resnet101 --gpus 0 1 2 3 --comment $_COMMENT "${_ADD_FLAGS[@]}"
