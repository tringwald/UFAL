#!/bin/bash
exit_script() {
  echo "Killing child processes."
  trap - INT TERM # clear the trap
  kill -- -$$     # Sends SIGTERM to child/sub processes
}
trap exit_script INT TERM

###############################################################################################################################
_ADD_FLAGS=(
"--epochs" "50"
"--test-every" "5"
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
"--imp-epochs" "75"
"--top-k" "65"
"--gpus" "0" "1" "2" "3"
"--architecture" "Resnet50"
"--use-filtering"
"--use-uncertain-features"
"--use-feature-loss"
"--sampling-mode" "UNCERTAINTY"
"--batch-construction" "SMART_BATCH_LAYOUT"
"--experiment-subdir" "OfficeHome/final"
"--seed" "1"
)
_COMMENT="rerun"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMEREAL --test-dataset OFFICEHOMEART --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMEREAL --test-dataset OFFICEHOMECLIPART --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMEREAL --test-dataset OFFICEHOMEPRODUCT --comment $_COMMENT "${_ADD_FLAGS[@]}"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMEPRODUCT --test-dataset OFFICEHOMEART --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMEPRODUCT --test-dataset OFFICEHOMEREAL --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMEPRODUCT --test-dataset OFFICEHOMECLIPART --comment $_COMMENT "${_ADD_FLAGS[@]}"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMECLIPART --test-dataset OFFICEHOMEART --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMECLIPART --test-dataset OFFICEHOMEREAL --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMECLIPART --test-dataset OFFICEHOMEPRODUCT --comment $_COMMENT "${_ADD_FLAGS[@]}"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMEART --test-dataset OFFICEHOMEPRODUCT --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMEART --test-dataset OFFICEHOMEREAL --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMEART --test-dataset OFFICEHOMECLIPART --comment $_COMMENT "${_ADD_FLAGS[@]}"


_ADD_FLAGS=(
"--epochs" "50"
"--test-every" "5"
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
"--imp-epochs" "60"
"--top-k" "65"
"--gpus" "0" "1" "2" "3"
"--architecture" "Resnet50"
"--use-filtering"
"--use-uncertain-features"
"--use-feature-loss"
"--sampling-mode" "UNCERTAINTY"
"--batch-construction" "SMART_BATCH_LAYOUT"
"--experiment-subdir" "OfficeHome/source_only"
"--seed" "1"
"--source-only"
)
_COMMENT="rerun"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMEREAL --test-dataset OFFICEHOMEART --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMEREAL --test-dataset OFFICEHOMECLIPART --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMEREAL --test-dataset OFFICEHOMEPRODUCT --comment $_COMMENT "${_ADD_FLAGS[@]}"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMEPRODUCT --test-dataset OFFICEHOMEART --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMEPRODUCT --test-dataset OFFICEHOMEREAL --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMEPRODUCT --test-dataset OFFICEHOMECLIPART --comment $_COMMENT "${_ADD_FLAGS[@]}"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMECLIPART --test-dataset OFFICEHOMEART --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMECLIPART --test-dataset OFFICEHOMEREAL --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMECLIPART --test-dataset OFFICEHOMEPRODUCT --comment $_COMMENT "${_ADD_FLAGS[@]}"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMEART --test-dataset OFFICEHOMEPRODUCT --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMEART --test-dataset OFFICEHOMEREAL --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OFFICEHOMEART --test-dataset OFFICEHOMECLIPART --comment $_COMMENT "${_ADD_FLAGS[@]}" 
