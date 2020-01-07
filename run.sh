# baseline model
python3 train.py \
--manualSeed 2222 \
--train_data data_lmdb_release/training --valid_data data_lmdb_release/validation \
--imgH 100 --imgW 100 \
--batch_max_length 50 \
--select_data / --batch_ratio 1.0 \
--Transformation None --FeatureExtraction OCResNet --SequenceModeling BiLSTM --Prediction Attn