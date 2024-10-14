## Dataset

For this project, I am using the **CodeSearchNet dataset** to detect formatting styles in Java code. The dataset provides a large variety of real-world code snippets, sourced from open-source repositories, and includes metadata such as the repository name, file path, function name, original code, tokenized code, and associated docstrings.

The dataset is split into three parts:
- **train.jsonl**: The training set, which consists of 16 separate `.jsonl` files. These files contain a large volume of code snippets in Java, providing a comprehensive dataset for training the machine learning model.
- **valid.jsonl**: A validation set used to tune hyperparameters and monitor model performance during training.
- **test.jsonl**: A test set that will be used for evaluating the final performance of the model on unseen data.

The **Java language** was chosen for this project due to its many formatting options and preferences, such as:
- **Brace placement** (K&R style, Allman style, etc.)
- **Indentation styles** (spaces vs. tabs, different levels of indentation)
- **Spacing** (around operators, after keywords, etc.)
- **Line wrapping** (breaking lines for long statements or parameters)

By using the different dataset splits, I aim to ensure the model generalizes well to a wide variety of coding styles. The training set, with its 16 files, allows for experimentation with detecting formatting patterns over a large and diverse codebase.

The dataset can be found [here](https://huggingface.co/datasets/code-search-net/code_search_net), with the Java subset being used for this task.