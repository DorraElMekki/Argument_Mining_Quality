# Argument Quality, NLP

## Goal
Predict the strength of an argument.

input= an argument (text) + premise type (categorical features).

Target_class= "STRONG"

Used dataset= FinArgQuality corpus (will be published soon).

## Abstract
Argumentation is the key to convincing people. 
This thesis leverages the power of NLP in Argument Mining to predict the quality of company executives' arguments during earnings conference calls.
We design a Bert-based model to predict the Strength argument quality dimension using the FinArgQuality corpus.
We explore the effect of incorporating the categorical features (premise/claim type and relation type) as input to the Bert model.
We use two input formats: features as text and features as a One-Hot encoded vector.
Using the Shapley values, we learn that only the feature “premise type'' contributes positively to predicting the Strength quality dimension.
In addition, it is more efficient to include the features as a One-Hot encoded vector than to include them as text.
With this approach, we achieve consistent results compared to the literature, and we outperform the Bert baseline model by 10\% for the macro F1 score.
![image](https://user-images.githubusercontent.com/34352894/224577818-f8138f0b-c843-418f-8b1a-464e59953a66.png)

### Keywords:
Computational Argumentation,
Argument Mining,
Argument Quality,
Robustness,
Multi-dataset learning,
Cross-domain,
Cross-topic,
Earning Calls.




## Activate the environment:
```shell
pip install -r requirements.txt
```

## Run the code

```shell
python main.py
```
#### input: earningsCall_argQ.csv
#### output: results.csv file and a log file
