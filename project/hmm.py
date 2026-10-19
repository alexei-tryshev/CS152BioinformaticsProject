import torch
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Categorical

# tensor = representing raw data of different things...
    # i.e. emission_matrices, starting probabilities, etc.

exon_probs = torch.tensor([0.20, 0.30, 0.30, 0.20])
intron_probs = torch.tensor([0.35, 0.15, 0.15, 0.35])
donor_probs = torch.tensor([0.05, 0.05, 0.85, 0.05])

# there's four entries per row bc there's four nucleotides: A, C, G, T
# first one = representing A, second one = representing C, etc (all the way to T)
# each entry = representing probability of nucleotide for each respective row (i.e. Exon, Intron, etc.)

distributions = [
    Categorical(exon_probs.unsqueeze(0)),
    Categorical(intron_probs.unsqueeze(0)),
    Categorical(donor_probs.unsqueeze(0))
]
# this = acting like observer for each hidden state (i.e. determing whether u go from Intron to Intron, Exon to Intron, etc.)
# this = calculating how likely a specific nucleotide will appear based on a specific DNA sequence

starts = torch.tensor([1.0, 0.0, 0.0])
# this is ur starting probabilities

# this = transition matrix
edges = torch.tensor([
    [0.95, 0.00, 0.05],
    [0.02, 0.98, 0.00],
    [0.00, 1.00, 0.00]
])
# row = from state; column = to state
# Rows = going from: Exon, Intron, Splice Site
# Cols = going from: Exon, Intron, Splice Site
# [0][0] = going from Exon to Exon, [0][1] = going from Exon to Intron, etc.

model = DenseHMM(distributions, edges=edges, starts=starts)

# test data
dna_to_int = {'A':0, 'C':1, 'G':2, 'T':3}
int_to_state = {0:"Exon", 1:"Intron", 2:"Splice_Site"}
# this = dictionary to convert into numbers for our tensors

test_sequence = "ACGCGTACGGTTTTTAAATTT"

formatted_sequence = torch.tensor([dna_to_int[base] for base in test_sequence])
# converts all the bases in the sequence w/ the base dictionary

input_tensor = formatted_sequence.unsqueeze(0).unsqueeze(-1)
# this = representing adding a new dimension so that Pomegranate can process things simultaneously
# 0 = representing the index of where you want to put the dimension..., so in this case it's the first index
    # this represents a new batch size with a sequence of how many bases there are...
# -1 = representing putting the dimension at the very end of the sequence

model_prediction = model.predict(input_tensor)[0]
# this = basically Viberti algorithm

probability_prediction = model.log_probability(input_tensor)[0]
# this = ur model's probability (tells you how likely that that it's the correct DNA sequence)
    # this means that if u let it run long enough, it would create this sequence of DNA...

print(f"Sequence Log Probability: {probability_prediction:.4f}\n")
print(f"{'Base':<6} | {'Predicted Hidden State':<22}")
print("-" * 33)
for base, state_idx in zip(test_sequence, model_prediction):
    print(f"{base:<6} | {int_to_state[state_idx.item()]:<22}")
