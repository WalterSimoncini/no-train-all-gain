source activate ssast

BATCH_SIZE=32
MODEL=patch-ssast
EMBEDDINGS_DIM=768
DATASETS=(esc-50 speech-commands)

DATASET_CACHE_DIR=/scratch-shared/$USER/cache/
OUTPUT_DIR=/scratch-shared/$USER/embeddings/$DATASET/$MODEL

for DATASET in "${DATASETS[@]}"; do
    mkdir -p $OUTPUT_DIR

    srun python $PWD/extract_features.py \
        --cache-dir $DATASET_CACHE_DIR \
        --output-path $OUTPUT_DIR/train.h5 \
        --batch-size $BATCH_SIZE \
        --dataset $DATASET \
        --dataset-split train \
        --embeddings-dim $EMBEDDINGS_DIM \
        --model $MODEL

    srun python $PWD/extract_features.py \
        --cache-dir $DATASET_CACHE_DIR \
        --output-path $OUTPUT_DIR/test.h5 \
        --batch-size $BATCH_SIZE \
        --dataset $DATASET \
        --dataset-split test \
        --embeddings-dim $EMBEDDINGS_DIM \
        --model $MODEL

    srun python $PWD/extract_features.py \
        --cache-dir $DATASET_CACHE_DIR \
        --output-path $OUTPUT_DIR/valid.h5 \
        --batch-size $BATCH_SIZE \
        --dataset $DATASET \
        --dataset-split valid \
        --embeddings-dim $EMBEDDINGS_DIM \
        --model $MODEL
done