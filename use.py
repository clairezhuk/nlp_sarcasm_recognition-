from labsnlp.CONFIG import SUBMISSION_DATA_TEST_PATH
from train import prepare_submission

if __name__ == "__main__":
    # change patch to necessary
    patch = SUBMISSION_DATA_TEST_PATH
    prepare_submission(patch)