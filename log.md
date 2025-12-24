## T = 0.5

Class distribution:
  LABEL_0: 98351
  LABEL_1: 41056

Train set: 111525, Test set: 27882

==================================================
TEST SET RESULTS
==================================================
Accuracy:  0.7909
ROC-AUC:   0.7305
F1-Score:  0.5591

Classification Report:
              precision    recall  f1-score   support

     LABEL_0       0.80      0.93      0.86     19671
     LABEL_1       0.74      0.45      0.56      8211

    accuracy                           0.79     27882
   macro avg       0.77      0.69      0.71     27882
weighted avg       0.78      0.79      0.77     27882


Confusion Matrix:
[[18354  1317]
 [ 4514  3697]]

## T = 0.1

Dataset size after cleaning: 139407 -> after ai_confidence filter: 59606

Class distribution:
  LABEL_0: 34233
  LABEL_1: 25373

Train set: 47684, Test set: 11922

==================================================
TEST SET RESULTS
==================================================
Accuracy:  0.8192
ROC-AUC:   0.8308
F1-Score:  0.7622

Classification Report:
              precision    recall  f1-score   support

     LABEL_0       0.80      0.92      0.85      6847
     LABEL_1       0.87      0.68      0.76      5075

    accuracy                           0.82     11922
   macro avg       0.83      0.80      0.81     11922
weighted avg       0.83      0.82      0.81     11922


Confusion Matrix:
[[6310  537]
 [1619 3456]]

## T = 0.05

Dataset size after cleaning: 139407 -> after ai_confidence filter: 40223

Class distribution:
  LABEL_0: 17264
  LABEL_1: 22959

Train set: 32178, Test set: 8045

==================================================
TEST SET RESULTS
==================================================
Accuracy:  0.8009
ROC-AUC:   0.8457
F1-Score:  0.8055

Classification Report:
              precision    recall  f1-score   support

     LABEL_0       0.71      0.91      0.80      3453
     LABEL_1       0.91      0.72      0.81      4592

    accuracy                           0.80      8045
   macro avg       0.81      0.81      0.80      8045

    accuracy                           0.80      8045
   macro avg       0.81      0.81      0.80      8045
weighted avg       0.82      0.80      0.80      8045


Confusion Matrix:
[[3125  328]
 [1274 3318]]

# SOTA pseudo-label validation

## SOTA pseudo-label agreement analysis

## SOTA pseudo-label calibration analysis

# User-level features analysis 

## Feature engineering

## Hybrid user-level -- comment-level model

## Results and discussion

================================================================================
SUMMARY OF CLUSTERING RESULTS ON 5% SAMPLED DATASET
================================================================================

  dataset        embedding  n_samples       ari  silhouette  accuracy
Full_Data minilm_embedding       6970 -0.006487    0.039560  0.537733
Full_Data tf_idf_embedding       6970 -0.045046    0.529829  0.639024
Full_Data empath_embedding       6970 -0.046424    0.739607  0.636729
 Conf_0.7 minilm_embedding       5449  0.008487    0.030114  0.546706
 Conf_0.7 tf_idf_embedding       5449 -0.025023    0.500375  0.679207
 Conf_0.7 empath_embedding       5449 -0.027540    0.729806  0.677372
 Conf_0.9 minilm_embedding       2980  0.000376    0.027128  0.514094
 Conf_0.9 tf_idf_embedding       2980  0.006035    0.080595  0.565772
 Conf_0.9 empath_embedding       2980 -0.005011    0.619259  0.580537

 