#!/bin/bash

# -- user specific
codepath=$(cd $(dirname $(dirname "${BASH_SOURCE[0]}")) && pwd) # path to the main directory of the code repo
MAIN_DIR=$codepath # can be different from the codepath, but should be the same as in config.py
CKPT_DIR="${MAIN_DIR}/ckpt" # should be the same as in config.py

# -- experiment specific
PHASE="$1" # choose: train / eval / heatmaps / ptflops
MODEL="$2" # choose: ARTEMIS / TIRG / cross-modal / visual-search / late-fusion / EM-only / IS-only
DATASET="$3" # choose: fashionIQ / shoes / cirr / fashion200k
RUN=0 # if you want to train the same model several times, use this as a form of identifier

exp_name="artemis_experiments/${DATASET}-${MODEL}-${RUN}"

exp_args=(
    --model_version ${MODEL}
    --data_name ${DATASET}
    --exp_name ${exp_name}
)

# -- update some arguments depending on the dataset
specific_args=()

if [ "$DATASET" == "fashionIQ" ] ; then
    specific_args=(
        --validate "test-val" # comment this line if you do not have the test split annotations
        #--validate "val" # uncomment this line if you do not have the test split annotations
    )
elif [ "$DATASET" == "cirr" ] ; then
    specific_args=(
        --load_image_feature 2048 # loaded feature size (usually: 0)
        --cnn_type 'resnet152'
    )
elif [ "$DATASET" == "fashion200K" ] ; then
    specific_args=(
        --batch_size 128 # default: 32
        --num_epochs 100 # default: 50
        --step_lr 20 # default: 10
        --cnn_type 'resnet18'
        --validate 'test'
    )
fi

# -- proceeding with the task (training/evaluation/heatmap generation)
if [ "$PHASE" == "train" ] ; then

    if [ "$DATASET" == "fashionIQ" ] ; then 
        echo "If you do not have the test split annotations (or do not wish to cross-validate on the test split), change 'test-val' to 'val' in the main script (FashionIQ case)."
    fi

    python3 ${codepath}/train.py "${exp_args[@]}" "${specific_args[@]}" \
    --num_epochs 8 --step_lr 4 --learn_temperature

    python3 ${codepath}/train.py "${exp_args[@]}" "${specific_args[@]}" \
    --img_finetune --txt_finetune --learn_temperature \
    --ckpt ${CKPT_DIR}/${exp_name}/ckpt.pth

elif [ "$PHASE" == "eval" ] ; then

    evaluate () {
        python3 ${codepath}/evaluate.py "${exp_args[@]}" "${specific_args[@]}" \
        --studied_split $1 \
        --ckpt ${CKPT_DIR}/${exp_name}/$2
    } 

    if [ "$DATASET" == "fashionIQ" ] ; then
        echo "Evaluate the last checkpoint on the val split"
        echo ""
        evaluate "val" "ckpt.pth"
        echo "Cross-validation: evaluate the best (val) checkpoint on test and the best (test) checkpoint on val. "
	echo "This will fail if you do not have the test split annotations. "
        echo ""
        evaluate "val" "test/model_best.pth"
        evaluate "test" "val/model_best.pth"
    elif [ "$DATASET" == "shoes" ] ; then
        evaluate "val" "ckpt.pth"
    elif [ "$DATASET" == "cirr" ] ; then
        evaluate "test" "val/model_best.pth"
    fi
    
elif [ "$PHASE" == "heatmaps" ] ; then

    heatmaps () {
        python ${codepath}/generate_heatmaps.py "${exp_args[@]}" "${specific_args[@]}" \
        --studied_split $1 --gradcam \
        --ckpt ${CKPT_DIR}/${exp_name}/$2
    }

    if [ "$DATASET" == "fashionIQ" ] ; then
        heatmaps "test" "val/model_best.pth"
    elif [ "$DATASET" == "shoes" ] ; then
        heatmaps "val" "ckpt.pth"
    elif [ "$DATASET" == "cirr" ] ; then
        echo "Cannot generate heatmaps for CIRR (model starts from image features instead of raw images)."
    fi

elif [ "$PHASE" == "ptflops" ] ; then

    CUDA_VISIBLE_DEVICES=-1 python ${codepath}/evaluate_ptflops.py --txt_enc_type lstm "${exp_args[@]}" "${specific_args[@]}"

fi
