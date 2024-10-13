import pandas as pd
from labsnlp.CONFIG import *
from labsnlp.preprocessing import preprocess_text
from labsnlp.vectorize import get_tfidf_vectorizer
from labsnlp.model import *

from sklearn.metrics import roc_auc_score
from joblib import dump, load

def train():
    train_data = pd.read_csv(TRAIN_DATA_PATH, index_col=0)
    test_data = pd.read_csv(TEST_DATA_PATH, index_col=0)

    # Preprocess text data
    preprocessed_text_train_data = preprocess_text(train_data)
    preprocessed_text_test_data = preprocess_text(test_data)

    # Generate TF-IDF vectors
    vectorizer = get_tfidf_vectorizer(preprocessed_text_train_data['headline'].values.tolist(), **TFIDF_OPTIONS)

    # Transform text data into TF-IDF sparse matrices
    X_train = vectorizer.transform(preprocessed_text_train_data['headline'].values.tolist())
    X_test = vectorizer.transform(preprocessed_text_test_data['headline'].values.tolist())

    # Train classifier (K-Nearest Neighbors by default)
    clf = train_neighbors_classifier(X_train.toarray(), train_data[OUTPUT_KEY].values.tolist())

    # Make predictions on test set
    predictions = clf.predict_proba(X_test.toarray())[:, 1]

    # Evaluate the classifier using ROC-AUC score
    print(f'ROC-AUC: {roc_auc_score(y_true=test_data[OUTPUT_KEY].values.tolist(), y_score=predictions)}')

    # Save trained classifier and vectorizer
    dump(clf, 'clf.joblib')
    dump(vectorizer, 'vectorizer.joblib')

train.__doc__="""
    Trains a classifier using preprocessed text data and TF-IDF vectorization.

    Reads training and testing data from specified paths, preprocesses text data,
    generates TF-IDF vectors, trains a chosen classifier (K-Nearest Neighbors by default),
    evaluates the classifier using ROC-AUC score on the test set, and saves the trained
    classifier and vectorizer.

    Returns:
        None
    """


def prepare_submission(patch=SUBMISSION_DATA_TEST_PATH):
    # Load trained classifier and vectorizer
    clf = load('clf.joblib')
    vectorizer = load('vectorizer.joblib')

    # Read submission test data
    data = pd.read_csv(patch, index_col=0)
    # Preprocess text data
    preprocessed_data = preprocess_text(data)

    # Generate TF-IDF vectors
    X = vectorizer.transform(preprocessed_data['headline'].values.tolist())
    # Make predictions using the classifier
    predictions = clf.predict_proba(X.toarray())[:, 1].tolist()

    # Save predictions in a submission file
    data[OUTPUT_KEY] = predictions
    data[OUTPUT_KEY].to_csv('submission.csv', sep=';')

prepare_submission.__doc__="""
    Prepares a submission file using a trained classifier and TF-IDF vectorizer.

    Loads a trained classifier and TF-IDF vectorizer from disk, reads submission test data,
    preprocesses the text data, generates TF-IDF vectors, makes predictions using the classifier,
    and saves the predictions in a submission file.

    Returns:
        None
    """

if __name__ == "__main__":
    train()
    prepare_submission()