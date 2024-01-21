# B-I-G-R-A-M (Bigram)

B-I-G-R-A-M is an auto-regressive character-level language model designed to generate new Indian baby names. Trained on a dataset comprising over 55k names, this model provides a creative tool for predicting unique names for newborns.

## Overview

The repository includes a compact 2K parameter transformer as the default model for generating names. Training progress and model checkpoints are saved in the working directory for easy access. The lightweight design ensures compatibility with a wide range of systems – no special hardware is required. However, utilizing a GPU can significantly enhance the training speed.

## Features

- Dataset: The model is trained on a comprehensive dataset containing more than 55,000 Indian baby names, ensuring diversity and richness in predictions.

- Model Architecture: The default model employs a super tiny 2K parameter transformer, striking a balance between performance and resource efficiency.

- Training Configuration: Various training configurations are available to cater to different preferences and requirements. Users can explore and customize the training process according to their needs.

## Implementation Details

### Probability Matrix Approach

The B-I-G-R-A-M model is implemented using a probability matrix for each character. This approach involves calculating the probability distribution of characters based on the dataset, providing a foundation for generating new names.

![Probability Distribution](Bigram(namePrediction)/_extensions/probDist.svg)

 **Note:** 
-  Model smoothing is a hyperparamter, that can be used for better results 

### Neural Network Approach

An alternative implementation of the B-I-G-R-A-M model leverages neural network architecture . This approach utilizes deep learning techniques to learn and predict character-level patterns in Indian baby names, offering a dynamic and data-driven method for generating unique names.

Feel free to explore both implementations and choose the one that best suits your needs. The detailed implementation and usage instructions can be found in the Bigram.ipynb notebook.

 **Note:** 
-  The weighs and biases are set at random in first, it is to avoid the uniform probability distribution of the logits on backpropagating
-  0.01*(W**2).mean() is a gravity, that is only to avoid the uniform distribution. It reduces the likelyhood of unwanted probabilties.

The regularization term 0.01*(W**2).mean() in the loss function helps prevent overfitting by penalizing large values in the weight tensor W. The learning rate -50 in the weight update determines the step size in the direction opposite to the gradient.

```python
  # gradient descent
for k in range(100):

  # forward pass
  xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
  logits = xenc @ W # predict log-counts
  counts = logits.exp() # counts, equivalent to N
  probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
  loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
  print(loss.item())

  # backward pass
  W.grad = None # set to zero the gradient
  loss.backward()

  # update
  W.data += -50 * W.grad
```


### Usage

The included `Names.txt` dataset, as an example, has the most common 32K names takes 'https://babynames.extraprepare.com/'. It looks like:

```
Aaban,
Aabharan
Aabhas
Aabhat
Aabheer
...
```
This will load the best model so far and print more samples on demand. Here are some unique baby names that get eventually generated from current default settings (test negative log likelyhood prob of ~2.1, though much lower logprobs are achievable with some hyperparameter tuning):
```
junidhananasan.
prusayaninan.
shinithashasathan.
sanniani.
mileviajedbininrwi.
thashiyanay.
arthanavaumarifottumj.
ponishinaruthi.
coraayaruganan.
jandin.
jamiki.
```
