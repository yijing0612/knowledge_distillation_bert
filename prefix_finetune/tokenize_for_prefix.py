import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.proprocess_agnews import load_and_tokenize_dataset


if __name__ == "__main__":
    tokenized = load_and_tokenize_dataset()
    tokenized.save_to_disk("data/tokenized_agnews")