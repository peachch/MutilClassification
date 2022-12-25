# MutilClassification
多分类任务

##Introduction
Classify user intentions based on real texts. To recognize users’comments intention automatically which can be used to analyze users’ concerns. Using Roberta and Classification fine-tune as baseline, then adding some tricks such as utilizing a vocabulary lookup table and transformed embedding. Finally, the precision improved from 60% of the baseline to 85%.
## Install
```
pip install requirements.txt
```

## Train
```
python train.py
```

## Inference
```
python predict_by_sentence.py
```