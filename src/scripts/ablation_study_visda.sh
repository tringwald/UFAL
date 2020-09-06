#!/bin/bash
exit_script() {
  echo "Killing child processes."
  trap - INT TERM # clear the trap
  kill -- -$$     # Sends SIGTERM to child/sub processes
}
trap exit_script INT TERM

###############################################################################################################################
_ADD_FLAGS=(
"--dataset" "VISDA17"
"--test-dataset" "VISDA17"
"--architecture" "Resnet101"
"--gpus" "0" "1" "2" "3"
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
"--imp-epochs" "15"
"--top-k" "12"
"--experiment-subdir" "Syn2Real/ablation"
"--seed" "1"
)

_COMMENT="SOURCE_ONLY"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --comment $_COMMENT "${_ADD_FLAGS[@]}" --source-only --show-per-class-stats --no-feature-loss
_COMMENT="TARGET_ONLY"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --comment $_COMMENT "${_ADD_FLAGS[@]}" --sampling-mode TARGET_ONLY --batch-construction RANDOM --no-feature-loss
_COMMENT="BASIC+SOURCE_FIRST"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --comment $_COMMENT "${_ADD_FLAGS[@]}" --sampling-mode BASIC_SAMPLING --batch-construction SOURCE_FIRST --no-feature-loss
_COMMENT="BASIC+TARGET_FIRST"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --comment $_COMMENT "${_ADD_FLAGS[@]}" --sampling-mode BASIC_SAMPLING --batch-construction TARGET_FIRST --no-feature-loss
_COMMENT="BASIC+RANDOM"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --comment $_COMMENT "${_ADD_FLAGS[@]}" --sampling-mode BASIC_SAMPLING --batch-construction RANDOM --no-feature-loss
_COMMENT="BASIC+RBL"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --comment $_COMMENT "${_ADD_FLAGS[@]}" --sampling-mode BASIC_SAMPLING --batch-construction RANDOM_BATCH_LAYOUT --no-feature-loss
_COMMENT="BASIC+SBL"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --comment $_COMMENT "${_ADD_FLAGS[@]}" --sampling-mode BASIC_SAMPLING --batch-construction SMART_BATCH_LAYOUT --no-feature-loss
_COMMENT="BASIC+SBL+UFL"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --comment $_COMMENT "${_ADD_FLAGS[@]}" --sampling-mode UNCERTAINTY --batch-construction SMART_BATCH_LAYOUT --use-feature-loss --use-uncertain-features --no-filtering
_COMMENT="BASIC+SBL+UBF"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --comment $_COMMENT "${_ADD_FLAGS[@]}" --sampling-mode UNCERTAINTY --batch-construction SMART_BATCH_LAYOUT --use-uncertain-features --use-filtering --no-feature-loss
_COMMENT="BASIC+SBL+UFL+UBF+no_uncertain_features"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --comment $_COMMENT "${_ADD_FLAGS[@]}" --sampling-mode UNCERTAINTY --batch-construction SMART_BATCH_LAYOUT --use-feature-loss --no-uncertain-features --use-filtering
_COMMENT="BASIC+SBL+UFL+UBF"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --comment $_COMMENT "${_ADD_FLAGS[@]}" --sampling-mode UNCERTAINTY --batch-construction SMART_BATCH_LAYOUT --use-feature-loss --use-uncertain-features --use-filtering
