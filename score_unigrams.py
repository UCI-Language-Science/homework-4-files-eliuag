# Write a function called score_unigrams that takes three arguments:
#   - a path to a folder of training data 
#   - a path to a test file that has a sentence on each line
#   - a path to an output CSV file
#
# Your function should do the following:
#   - train a single unigram model on the combined contents of every .txt file
#     in the training folder
#   - for each sentence (line) in the test file, calculate the log unigram 
#     probability ysing the trained model (see the lab handout for details on log 
#     probabilities)
#   - write a single CSV file to the output path. The CSV file should have two
#     columns with headers, called "sentence" and "unigram_prob" respectively.
#     "sentence" should contain the original sentence and "unigram_prob" should
#     contain its unigram probabilities.
#
# Additional details:
#   - there is training data in the training_data folder consisting of the contents 
#     of three novels by Jane Austen: Emma, Sense and Sensibility, and Pride and Prejudice
#   - there is test data you can use in the test_data folder
#   - be sure that your code works properly for words that are not in the 
#     training data. One of the test sentences contains the words 'color' (American spelling)
#     and 'television', neither of which are in the Austen novels. You should record a log
#     probability of -inf (corresponding to probability 0) for this sentence.
#   - your code should be insensitive to case, both in the training and testing data
#   - both the training and testing files have already been tokenized. This means that
#     punctuation marks have been split off of words. All you need to do to use the
#     data is to split it on spaces, and you will have your list of unigram tokens.
#   - you should treat punctuation marks as though they are words.
#   - it's fine to reuse parts of your unigram implementation from HW3.

# You will need to use log and -inf here. 
# You can add any additional import statements you need here.
from math import log, inf

#######################
# YOUR CODE GOES HERE #
#######################
import os
import csv
import math
from pathlib import Path
from collections import Counter

def train_unigram_model(training_path):
    training_path = Path(training_path)
    word_counts = Counter()
    total_words = 0
    
    for file in training_path.glob("*.txt"):
        with file.open("r") as f:
            for line in f:
                words = line.strip().lower().split()
                word_counts.update(words)
                total_words += len(words)
    
    unigram_probs = {word: count / total_words for word, count in word_counts.items()}
    return unigram_probs, total_words

def calculate_log_probability(sentence, unigram_probs, total_words):
    words = sentence.strip().lower().split()
    log_prob = 0
    
    for word in words:
        if word in unigram_probs:
            log_prob += math.log(unigram_probs[word])
        else:
            return float('-inf')
    
    return log_prob

def score_unigrams(training_data_path, test_file_path, output_csv_path):
    unigram_probs, total_words = train_unigram_model(training_data_path)
    test_file_path = Path(test_file_path)
    output_csv_path = Path(output_csv_path)
    
    with test_file_path.open("r") as test_file, output_csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["sentence", "unigram_prob"])
        
        for line in test_file:
            log_prob = calculate_log_probability(line, unigram_probs, total_words)
            writer.writerow([line.strip(), log_prob])

# Do not modify the following line
if __name__ == "__main__":
    # You can write code to test your function here
    pass 
