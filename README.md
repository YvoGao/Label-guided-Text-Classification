# Boosting Meta-Learning for Few-Shot Text Classification via Label-guided Distance Scaling

## create python env

```
conda create -n LDS python=3.7
source activate LDS
pip install requirements.txt -r
```

## data
```
cd data
unzip data.zip
```

## quick start

```
cd scripts
sh ours_all.sh
```
The specific parameters in the paper are consistent with ours_all.sh.
**Before you start, you should download bert-base-uncased from Huggingface https://huggingface.co/google-bert/bert-base-uncased, and change the path in the ours_all.sh file to your own file path.**
