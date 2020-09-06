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
"--architecture" "Resnet50"
"--gpus" "0" "1" "2" "3"
"--use-filtering"
"--use-feature-loss"
"--use-uncertain-features"
"--imp-epochs" "15"
"--top-k" "10"
"--sampling-mode" "UNCERTAINTY"
"--batch-construction" "SMART_BATCH_LAYOUT"
"--experiment-subdir" "OfficeCaltech/final"
"--seed" "1"
)
_COMMENT="rerun"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCAMAZON --test-dataset OCCALTECH --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCAMAZON --test-dataset OCDSLR --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCAMAZON --test-dataset OCWEBCAM --comment $_COMMENT "${_ADD_FLAGS[@]}"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCDSLR --test-dataset OCAMAZON --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCDSLR --test-dataset OCCALTECH --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCDSLR --test-dataset OCWEBCAM --comment $_COMMENT "${_ADD_FLAGS[@]}"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCCALTECH --test-dataset OCAMAZON --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCCALTECH --test-dataset OCDSLR --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCCALTECH --test-dataset OCWEBCAM --comment $_COMMENT "${_ADD_FLAGS[@]}"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCWEBCAM --test-dataset OCAMAZON --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCWEBCAM --test-dataset OCCALTECH --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCWEBCAM --test-dataset OCDSLR --comment $_COMMENT "${_ADD_FLAGS[@]}"





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
"--architecture" "Resnet50"
"--gpus" "0" "1" "2" "3"
"--use-filtering"
"--use-feature-loss"
"--use-uncertain-features"
"--imp-epochs" "15"
"--top-k" "10"
"--sampling-mode" "UNCERTAINTY"
"--batch-construction" "SMART_BATCH_LAYOUT"
"--experiment-subdir" "OfficeCaltech/source_only"
"--seed" "1"
"--source-only"
)
_COMMENT="rerun"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCAMAZON --test-dataset OCCALTECH --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCAMAZON --test-dataset OCDSLR --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCAMAZON --test-dataset OCWEBCAM --comment $_COMMENT "${_ADD_FLAGS[@]}"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCDSLR --test-dataset OCAMAZON --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCDSLR --test-dataset OCCALTECH --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCDSLR --test-dataset OCWEBCAM --comment $_COMMENT "${_ADD_FLAGS[@]}"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCCALTECH --test-dataset OCAMAZON --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCCALTECH --test-dataset OCDSLR --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCCALTECH --test-dataset OCWEBCAM --comment $_COMMENT "${_ADD_FLAGS[@]}"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCWEBCAM --test-dataset OCAMAZON --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCWEBCAM --test-dataset OCCALTECH --comment $_COMMENT "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --dataset OCWEBCAM --test-dataset OCDSLR --comment $_COMMENT "${_ADD_FLAGS[@]}"

