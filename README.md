# LBCE-XGB
1 Predict new data sets
1.1 Extracting BERT embadding based on BERT pre-training model(positive and negative samples are extracted separately)
    bash extract_features.sh
1.2 Extracting embeddings of the token 'CLS'
    python3 extral_CLS_fea.py input output label
    Extract the positive and negative sample files separately and merge them into one file.
1.3 Predict
    python3 predict.py dataset
    Note: The positive and negative samples in this data set folder should be put into two files and named pos.txt and neg.txt. Line 353 of file changes the read path.

2 Training model
2.1 Extracting BERT embadding based on BERT pre-training model(positive and negative samples are extracted separately)
    bash extract_features.sh
2.2 Extracting embeddings of the token 'CLS'
    python3 extral_CLS_fea.py input output label
    Extract the positive and negative sample files separately and merge them into one file.
2.3 Shap feature selection
    python3 xgb_shap.py training test
2.4 Training
    python3 xgb_retrain.py dataset
    Note: Modify xgb_retrain.py file 274 line read BERT feature file, 185 line read shap output sorting file, 187 line modified read BERT feature dimension.
    Note: The positive and negative samples in this data set folder should be put into two files and named pos.txt and neg.txt. Line 430 changes the read path.
