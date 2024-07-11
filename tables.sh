################# BEGIN knn
BACKBONES=(vit-b16 vit-b32 vit-l16 mae-vit-b16 dino_vitb16 dinov2_vitb14 clip-vit-b16 eva-clip-vit-b16 vit-b16-augreg-in1k vit-b16-augreg-in21k vit-l16-augreg-in21k)

mkdir -p data/knn

for BACKBONE_NAME in "${BACKBONES[@]}"; do
    python generate_model_table.py --logs-folder /scratch-shared/$USER/logs/$BACKBONE_NAME/ --output-path data/knn/$BACKBONE_NAME.csv
    python generate_model_table.py --logs-folder /scratch-shared/$USER/logs/$BACKBONE_NAME/5-shots --output-path data/knn/$BACKBONE_NAME-5-shots.csv
done
################# END knn

################# BEGIN retrieval
BACKBONES=(
    clip-vit-b16
    dinov2_vitb14
    dino_vitb16
    eva-clip-vit-b16
    vit-b16
)

mkdir -p data/retrieval

for BACKBONE_NAME in "${BACKBONES[@]}"; do
    python generate_model_table_retrieval.py \
        --logs-folder /scratch-shared/$USER/logs/$BACKBONE_NAME/ \
        --output-path data/retrieval/$BACKBONE_NAME-M.csv \
        --metric-name mAP-M

    python generate_model_table_retrieval.py \
        --logs-folder /scratch-shared/$USER/logs/$BACKBONE_NAME/ \
        --output-path data/retrieval/$BACKBONE_NAME-H.csv \
        --metric-name mAP-H
done
################# END retrieval

################# BEGIN linear
BACKBONES=(vit-b16 vit-b32 vit-l16 mae-vit-b16 dino_vitb16 dinov2_vitb14 clip-vit-b16 eva-clip-vit-b16 vit-b16-augreg-in1k vit-b16-augreg-in21k vit-l16-augreg-in21k)

mkdir -p data/linear

for BACKBONE_NAME in "${BACKBONES[@]}"; do
    python generate_model_table.py --logs-folder /scratch-shared/$USER/logs/linear/$BACKBONE_NAME/ --output-path data/linear/$BACKBONE_NAME.csv
    python generate_model_table.py --logs-folder /scratch-shared/$USER/logs/linear/$BACKBONE_NAME/5-shots --output-path data/linear/$BACKBONE_NAME-5-shots.csv
done
################# END linear
