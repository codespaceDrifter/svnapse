classification evaluation

|                    | Actual Positive | Actual Negative |
|--------------------|-----------------|---------------|
| Predicted Positive | TP              | FP            |
| Predicted Negative | FN              | TN            |

(true positive, false positive, false negative, true negative)

> Accuracy = (TP+TN) / (TP+TN+FP+FN)

precision: among the predicted positives how many are actually positive

> Precision = TP/(TP+FP)

recall: all the actually positive things how many are predicted as positive

> Recall = TP / (TP + TN)

F1 score: weighted average of precision and recall. more useful than accuracy especially with uneven class distribution

F1 = 2/ ((1/precision) + (1/recall))
> F1 = (2* Precision *Recall) / (Precision + Recall)

F1 ranges from 0 to 1. better than accuracy in case of dataset target inbalance. better than precision or recall alone or a normal average because say precision 0.95 recall 0.35 is problematic but a normal mean would show 0.65 which seems fine. basically F1 score tanks when either precision or recall is low.  

regression evaluation

mean squared error
> $$MSE = (1/N) * Σ(yᵢ - ŷᵢ)²$$

mean absolute error
> $$MAE = (1/N) * Σ|yᵢ - ŷᵢ|$$

both are values that can help select best regression model but doesn't give specific insights. MSE penalizes big values more strongly than MAE.

overfitting and underfitting

an ML model can be thought of as approximating a target function
generalization is how well the model predicts new data  

overfitting happens when model learns the details and noise in the training data that it hurts performance in testing data. if the model does very well on training set but poor on test set it is overfitting. to prevent overfitting, we can train with more data, or do cross validation. 

cross validation: k fold. divide data into k folds. train a model on everything but 1 test fold, gather the k scores, and average them as a eval of the model scores. use this to avoid misleading test sets like all very eazy or very hard predictions.  

underfitting happens when model does not learn the target function. it can happen if the model is stopped early in training. underfitted model does bad on both training and test sets. 
