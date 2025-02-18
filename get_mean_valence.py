# The file valence_data/winter_2016_senses_valence.csv contains data from an 
# experiment that asked people to provide valence ratings for words associated
# with each of the five senses (touch, taste, smell, sight, sound). The file has
# three columns: Word, Modality, and Val. Word contains the word, Modality the
# sensory modality, and Val contains the mean valence rating for that word,
# where higher valence corresponds to more positive emotional states.

# The question we'll try to answer is whether certain sensory modalities have 
# higher or lower mean valences than others.
# 
#  Write a function called get_mean_valence that takes a Path to a CSV file
#  as input. You can assume the file will be formatted as described above.
#  Your function should return a dictionary with keys corresponding to each
#  of the five modalities. The value for each key should be its mean valence
#  score across all of the words in the CSV file.

# The data are from the paper 
#
# Winter, B. (2016). Taste and smell words form an affectively loaded and emotionally
# flexible part of the English lexicon. Language, Cognition and Neuroscience, 31(8), 
# 975-988.

#######################
# YOUR CODE GOES HERE #
#######################
import csv
from collections import defaultdict

def get_mean_valence(file_path):
    valence_sums = defaultdict(float)
    valence_counts = defaultdict(int)
    
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        
        for row in reader:
            modality, valence = row[1], float(row[2])
            valence_sums[modality] += valence
            valence_counts[modality] += 1
    
    mean_valences = {modality: valence_sums[modality] / valence_counts[modality] for modality in valence_sums}
    
    return mean_valences

# Do not modify the following line
if __name__ == "__main__":
    # You can write code to test your function here
    pass 