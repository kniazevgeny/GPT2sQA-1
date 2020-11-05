# ruGPT3 for QA

This repo includes an experiment of fine-tuning ruGPT-3Large for Question Answering (QA). It also runs the model on SQuAD-like dataset: sberquad. It uses Huggingface Inc.'s PyTorch implementation of ruGPT-3 and adapts from their fine-tuning of BERT for QA. 

SQuAD data can be downloaded from: https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset

SberQuAD data can be downloaded from: https://github.com/kniazevgeny/BERT-QA-fine-tuning

## To train and validate the model: 
### GPU or CPU

```
python gpt2_squad.py --output_dir=output/ --train_file=data/train-v2.0.json --do_train --train_batch_size=8 --predict_file=data/dev-v2.0.json --do_predict --model_name=ruGPT3Small

```
Also, you could specify model name. Use --model_name arg. Example: --model_name=ruGPT3Large

Only 3 models are avaliable: ruGPT3Small, ruGPT3Medium and ruGPT3Large
### TPU (Colab)
```
python gpt2_squad_tpu.py --output_dir=output/ --train_file=data/train-v2.0.json --do_train --train_batch_size=32 --predict_file=data/dev-v2.0.json --do_predict

```
#### Arguments:
Required:
- ```--train_file``` SQuAD-like json for training. E.g., train-v1.1.json
- ```--predict_file``` SQuAD-like json for predictions. E.g., dev-v1.1.json or test-v1.1.json
- ```--output_dir``` The output directory where the model checkpoints and predictions will be written.

You may like to change these:

- ```--model_name``` ruGPT3Small or ruGPT3Medium or ruGPT3Large
- ```--with_negative``` dataset version 2 (with negative) or not
- ```--max_seq_length``` The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.
- ```--doc_stride``` When splitting up a long document into chunks, how much stride to take between chunks.
- ```--max_query_length``` The maximum number of tokens for the question. Questions longer than this will be truncated to this length.
- ```--do_train``` Whether to run training.
- ```--do_predict``` Whether to run eval on the dev set.
- ```--train_batch_size``` Total batch size for training.
- ```--predict_batch_size``` Total batch size for predictions.
- ```--learning_rate``` The initial learning rate for Adam.
- ```--num_train_epochs``` Total number of training epochs to perform.
- ```--warmup_proportion``` Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.
- ```--n_best_size``` The total number of n-best predictions to generate in the nbest_predictions.json output file.

And others:
- ```--max_answer_length``` 
- ```--verbose_logging``` 
- ```--no_cuda``` 
- ```--seed``` 
- ```--gradient_accumulation_steps``` 
- ```--do_lower_case``` 
- ```--local_rank``` 
- ```--loss_scale``` 
- ```--null_score_diff_threshold``` 
## To evaluate: 

```

python evaluate-v2.0.py data/dev-v2.0.json output/predictions.json

```
