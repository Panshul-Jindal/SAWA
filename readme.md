# SAWA (Sentiment Analysis With Attention
## Learning Attention Mechanism via Sentiment Analysis

For this task, I implemented various attention mechanisms (Bahdanau Attention, Luong Dot
Attention, and Luong General Attention) in Python, and integrated them with RNN and LSTM
models to build a sentiment analysis model.


## Training

Dataset Used: [Stanford NLP's IMDB Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)

The dataset contains highly polar plain text reviews of movies on IMDB as well as their sentiment
(positive or negative). The dataset is split into a training set of 25000 entries and a test set
of 25000 entries.


## Findings



![](Final_results.csv)

* The models gave F1 Score between 50% and 85.8%.
*In general RNNs performed worse than LSTMs. LSTM and Bidirectional LSTM generally outperform RNNs in F1-Score, even without attention.
* Bidirectional gave better performance than unidirectional.
*  Attention mechanisms significantly improve F1-Score for all base models, especially for RNN and Bidirectional RNN, where Bahdanau and Luong Concat attention show the largest gains.


* max f1 score of 86.88% is obtained by VanillaBidirectionalLSTMWithLuongDotProductAttention**
* Luong Dot Product Attention achieves the highest F1-Score for LSTM and Bidirectional LSTM. While Interestingly the same attention recieved low score for both RNN and Bidirectional RNN

* Training time and parameters increase with attention complexity, especially for Bahdanau and Concat variants.

## Scope for Improvement
 - During tokenization, I truncated the stop words found in english dataset excluding the negations. In the process, words which intensify adjects are lost. Eg very very good is interpreted same as good.
 - There are some bad results being shown as visulaized by attention weights.
 - Implement Modularization, aiming for that each phase of pipeline is modular
 - Implement CLI (It helps in doing things faster)( And is challenging too)





## Notes



- Do refer to report.pdf for extensive info with all the evaluation results, confusion matrices and attention heatmaps.
