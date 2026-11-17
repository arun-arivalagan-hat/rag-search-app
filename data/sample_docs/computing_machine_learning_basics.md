# Machine Learning Basics

Machine learning is the practice of building systems that improve at a task by learning patterns from data rather than following hand-written rules.

## Supervised learning

In supervised learning the training data includes the correct answer, called a label. A model learns a mapping from inputs to labels. Classification predicts a category, such as spam or not spam, while regression predicts a number, such as a house price. Training typically minimises a loss function using gradient descent, which repeatedly adjusts the model's parameters in the direction that reduces error.

## Unsupervised learning

Unsupervised learning finds structure in unlabelled data. Clustering groups similar examples together, and dimensionality reduction compresses many features into a few while preserving as much information as possible.

## Overfitting and evaluation

A model that memorises its training data but fails on new data is overfitting. To detect it, the data is split into training, validation and test sets, and performance is measured on data the model has never seen. Regularisation, more data and simpler models all reduce overfitting.

## Embeddings

Modern systems often represent text as embeddings: dense vectors in which semantically similar texts lie close together. Similarity between two embeddings is commonly measured with cosine similarity, which compares the angle between vectors and ignores their length.
