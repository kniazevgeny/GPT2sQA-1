# ruGPT3 for QA

This repo includes an experiment of fine-tuning ruGPT-3Large for Question Answering (QA). It also runs the model on SQuAD - like dataset: sberquad. It uses Huggingface Inc.'s PyTorch implementation of ruGPT-3 and adapts from their fine-tuning of BERT for QA. 

SQuAD data can be downloaded from: https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset
SberQuAD data can be downloaded from: https://github.com/kniazevgeny/BERT-QA-fine-tuning

To train and validate the model: 
### GPU or CPU

```
python gpt2_squad.py --output_dir=output/ --train_file=data/train-v2.0.json --do_train --train_batch_size=8 --predict_file=data/dev-v2.0.json --do_predict

```
### TPU (Colab)
```
python gpt2_squad_tpu.py --output_dir=output/ --train_file=data/train-v2.0.json --do_train --train_batch_size=32 --predict_file=data/dev-v2.0.json --do_predict

```

To evaluate: 

```

python evaluate-v2.0.py data/dev-v2.0.json output/predictions.json

```
