# Multilayer Perceptrons  - Character and Word Level MLPs

- MLP, following [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

This project implements character-level  Multilayer Perceptrons (MLPs) inspired by the research paper by Bengio. The aim is to predict the next character or word in a given sequence.

### Character Level MLP

For character-level prediction, a 2D embedding approach is used .

### Word Level MLP

The feature vector represents various aspects of each word, placing them in a vector space. The number of features, denoted as `m` (e.g., 30, 60, or 100 in the experiments), is significantly smaller than the size of the vocabulary (e.g., 17,000). The probability function is expressed as a product of conditional probabilities, predicting the next word given the previous ones. This is achieved using a multilayer neural network with parameters that can be iteratively tuned to maximize the log-likelihood of the training data. A regularization criterion, such as weight decay penalty, can be applied.

```python
import re

with open("Mahabharat.txt", 'r') as file:
    lines = file.read().splitlines()

text = ' '.join(lines).lower()
cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)

words = cleaned_text.split()

len(words)
```
```python
block_size = 3  # context length: how many words do we take to predict the next one?
X, Y = [], []

for i in range(10):
    context = words[i:i + block_size]
    next_word = words[i + block_size]
    
    context_indices = [stoi.get(w, len(itos)) for w in context]  # Use len(itos) as the default index for unknown words
    next_word_index = stoi.get(next_word, len(itos))
    
    X.append(context_indices)
    Y.append(next_word_index)
    
    print(' '.join(context), '--->', next_word)
```
```
the complete mahabharata ---> in
complete mahabharata in ---> english
mahabharata in english ---> the
in english the ---> mahabharata
english the mahabharata ---> of
the mahabharata of ---> krishnadwaipayana
mahabharata of krishnadwaipayana ---> vyasa
of krishnadwaipayana vyasa ---> book
krishnadwaipayana vyasa book ---> adi
vyasa book adi ---> parva
```
## Neural architecture
![nn architecture](_extensions/img/NN_architecture.png)

## References

- Mention the Bengio research paper that inspired this project.
