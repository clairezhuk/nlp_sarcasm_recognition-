# Sarcasm Detection

This project leverages Large Language Models (LLM) and machine learning techniques to identify sarcastic tones in text. It allows users to train a model, save it for future use, and choose between different classification models.

## Project Structure

- **Input Data**: The training data is located at `data/sarcasm_detection.csv`.
  
- **Data Splitting**: To split the dataset into training and testing sets, run the script `scripts/train_test_split.py`. This will create `data/train.csv` and `data/test.csv`.

- **Model Selection and Customization**: In `labsnlp/model.py`, you can find the available algorithms for the classifiers. You can adjust their hyperparameters or add your own classifier. The current optimal model is implemented in the `train_neighbors_classifier` function.

## Training the Model

To train the model and test its performance, run the `train.py` script. Make sure to do this before using the model for the first time, as the trained model file `clf.joblib` is too large to be stored in GitHub and must be generated locally.

## Using the Trained Model

To use the trained model on your own data:
1. Replace the placeholder `SUBMISSION_DATA_TEST_PATH` in `use.py` with the path to your CSV file.
2. Run `use.py` to apply the model to your data.
