import random
import torch
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Categorical

from data_loader import load_training_data, training_path, encode_dna

TEST_FILE = "training_genes_new.tsv"
# this = ur testing file (i.e. what ur model = using 2 test)
# if not provided --> chooses random file & 20% of file = used 4 testing

TEST_FRACTION = 0.2
# IF TEST_FILE = empty --> 20% of genes = reserved 4 testing
# IF TEST_FILE != empty --> 20% of genes = used for validation genes
    # validation genes = used 4 intron-entry boost (encouraging exon-intron state)

MAX_GENE_LENGTH = 100000
# skip any genes LONGER than 100,000 bases (so training doesn't take forever)

CONTEXT_BASES = 2
# this = looking 2 bases BEFORE ur current base...
    # SO...ur model = looking 3 bases (current one + 2 bases beforehand)
    # this = boosting up the accuracy bc more bases = providing more context of what it stands for...
        # i.e. TAG, GC = meaning smn (like stop codon, exon site) --> can't rlly do that w/ single bases

BOOST_CHOICES = [1, 3, 10, 15, 20, 25, 30]
# basically --> intron-entry boost = helping shift probabilities MORE towards introns...
    # this = bc probabilities of going from exon --> intron & intron --> exon = rlly low
    # different values = meaning more likelihood 2 find intron, BUT more likelihood of false ones too

RUN_BAUM_WELCH = False
BW_MAX_ITER = 20
# Baum-Welch = ONLY addjusting the transitions NOT emissions
    # it = off by default (bc it = making things worse each test...)

SEED = 0
# this = randomizer (for training/testing gene)

label_letter = "EIS"   # labels in training_genes.tsv: 0 = E (exon), 1 = I (intron), 2 = S (splice site)
K = CONTEXT_BASES + 1  # length of how many bases it = reading...(3 bases in total)
N_CONTEXTS = 4 ** CONTEXT_BASES   # 16 combos for 2 bases (AA, AT, AG, AC, ..., TT)
N_SYMBOLS = 4 ** K                # 64 possible 3-base patterns (4 * 4 * 4)


#   donor  (exon -> intron):   C A G | G T A A G T      ("|" = where the intron starts)
#                             Donor_E1-3  Donor_I1-6   (Donor_I1 = the splice site)
#
#   acceptor (intron -> exon): Y Y Y Y N C A G | exon   (Y = C or T)
#                              Acceptor1 ... Acceptor8

# this = what splice sites = usually looking like...
    # our model NOW = recognizing that (bc it = looking @ 3 bases @ a time)
    # splice sites = having certain patterns... --> our model = able 2 recognize that...
N_DONOR_EXON = 3     # last 3 exon bases before an intron
N_DONOR_INTRON = 6   # first 6 intron bases
N_ACCEPTOR = 8       # last 8 intron bases

STATE_NAMES = (["Exon"]
               + [f"Donor_E{i}" for i in range(1, N_DONOR_EXON + 1)]
               + [f"Donor_I{i}" for i in range(1, N_DONOR_INTRON + 1)]
               + ["Intron"]
               + [f"Acceptor{i}" for i in range(1, N_ACCEPTOR + 1)])
# so this basically creates a state for each EXON, INTRON, ACCEPTOR state...
    # 3 states for each N_DONOR_EXON
    # 6 states for each N_DONOR_INTRON
    # 8 states for each N_ACCEPTOR
    # add all that w/ 2 states (exon & intron) --> u = getting 19 total states

N_STATES = len(STATE_NAMES)   # 19

EXON = 0
DONOR_E = list(range(1, 1 + N_DONOR_EXON))                                  # states 1-3
DONOR_I = list(range(DONOR_E[-1] + 1, DONOR_E[-1] + 1 + N_DONOR_INTRON))    # states 4-9
INTRON = DONOR_I[-1] + 1                                                     # state 10
ACCEPTOR = list(range(INTRON + 1, INTRON + 1 + N_ACCEPTOR))                  # states 11-18
# we = grouping up the indexes of each "type"
    # i.e. states 11-18 are ACCEPTORS & states 1-3 are DONOR_EXONS
    # this = helping us distinguish which states belong to which...

# THEN we = converting those (from previous step) 2 fit w/ E/I/S
# we = grouping those sections to be Intron, Exon, Splice Site
STATE_TO_LABEL = torch.zeros(N_STATES, dtype=torch.long)
STATE_TO_LABEL[DONOR_E] = 0           # last bases of an exon are still exon
STATE_TO_LABEL[DONOR_I] = 1           # first bases of an intron are intron...
STATE_TO_LABEL[DONOR_I[0]] = 2        # ...except the first one, which is the splice site
STATE_TO_LABEL[INTRON] = 1
STATE_TO_LABEL[ACCEPTOR] = 1

# this = governing WHICH transitions = allowed & whatnot...
    # i.e. some transitions are NOT allowed --> i.e. intron --> exon = NOT allowed...
        # we = needing some splice site in between those things...
ALLOWED = torch.zeros(N_STATES, N_STATES, dtype=torch.bool)
# this = creating 19x19 table of boolean values... --> ALL FALSE (to begin w/)
    # BUT some values = becoming true --> that = forming valid sequence

chain = [EXON] + DONOR_E + DONOR_I + [INTRON] + ACCEPTOR + [EXON]
for a, b in zip(chain[:-1], chain[1:]):
    ALLOWED[a, b] = True              # walk through each chain in order
# this = guaranteeing the valid chain (i.e. EXON --> DONOR_E & DONOR_I --> INTRON --> ACCEPTOR --> EXON)
    # this = basically like a MINI HMM --> different parts of the chain = different states...

ALLOWED[EXON, EXON] = True            # exons state = able 2 loop (they = repeating themselves)
ALLOWED[INTRON, INTRON] = True        # introns states = able 2 loop (so they = repeating themselves)
ALLOWED_STARTS = torch.zeros(N_STATES, dtype=torch.bool)
ALLOWED_STARTS[EXON] = True           # genes start in an exon
ALLOWED_ENDS = torch.zeros(N_STATES, dtype=torch.bool)
ALLOWED_ENDS[EXON] = True             # ...and end in an exon
# last 4 = ensuring u = starting w/ & ending off w/ an exon...

#------------------------------------------------------------------------------------------------------------

# this = turning DNA into 3-base pattern numbers (sorting them into 0-63)
    # this = helping us categorizing certain gene combinations
        # i.e. which gene sequence = more likely in introns vs. exons
def to_patterns(bases):
    padded = torch.cat([torch.zeros(CONTEXT_BASES, dtype=torch.long), bases])
    patterns = torch.zeros(len(bases), dtype=torch.long)
    for j in range(K):
        patterns = patterns * 4 + padded[j:j + len(bases)]
    return patterns.unsqueeze(-1)   # shape (length, 1), what pomegranate wants

# this = converting E/I/S states into the 19 states
def labels_to_states(y):
    states = torch.where(y == 0, EXON, INTRON)

    exon_starts = torch.where((y[1:] == 0) & (y[:-1] != 0))[0] + 1

    for site in torch.where(y == 2)[0].tolist():
        intron_end = exon_starts[torch.searchsorted(exon_starts, site)].item()  # next exon start
        too_short = (intron_end - site < N_DONOR_INTRON + 1 + N_ACCEPTOR
                     or site < N_DONOR_EXON + 1
                     or (y[site - N_DONOR_EXON - 1:site] != 0).any())
        if too_short:
            return None   # this gene has an intron or exon too short for the chains

        states[site - N_DONOR_EXON:site] = torch.tensor(DONOR_E)
        states[site:site + N_DONOR_INTRON] = torch.tensor(DONOR_I)
        states[intron_end - N_ACCEPTOR:intron_end] = torch.tensor(ACCEPTOR)
    return states

# learn probabilities from labeled genes
def count_probabilities(bases_list, Y, pseudocount=1.0):
    start_counts = torch.zeros(N_STATES) # emptying out counts (all the counts)
    end_counts = torch.zeros(N_STATES)
    edge_counts = torch.zeros(N_STATES, N_STATES)
    pattern_counts = torch.zeros(N_STATES, N_SYMBOLS)
    skipped = 0

    for bases, y in zip(bases_list, Y): # this = counting up the 
        path = labels_to_states(y)
        if path is None:
            skipped += 1
            continue
        start_counts[path[0]] += 1
        end_counts[path[-1]] += 1
        edge_counts += torch.bincount(path[:-1] * N_STATES + path[1:],
                                      minlength=N_STATES ** 2).reshape(N_STATES, N_STATES)
        patterns = to_patterns(bases)[:, 0]
        pattern_counts += torch.bincount(path * N_SYMBOLS + patterns,
                                         minlength=N_STATES * N_SYMBOLS).reshape(N_STATES, N_SYMBOLS)
        # remember that pattern (0 - 63) for the genes sequences?
            # it calculating which sequence = correlating to which number
            # then it = seeing how many times that numbers shows up in a sequence...
                # that = helping us w/ emission & transition probability counts

    if skipped:
        print(f"  (skipped {skipped} genes with an intron or exon too short for the chains)")

    start_counts = (start_counts + pseudocount) * ALLOWED_STARTS
    end_counts = (end_counts + pseudocount) * ALLOWED_ENDS
    edge_counts = (edge_counts + pseudocount) * ALLOWED
    out_totals = edge_counts.sum(dim=1) + end_counts
    starts = start_counts / start_counts.sum()
    edges = edge_counts / out_totals.unsqueeze(1)
    ends = end_counts / out_totals

    # this = getting counts of the starts, ends, & edges based on the allowed moves...
        # remember how genes = moving based on a SPECIFIC pattern?
        # this = getting the counts of those based on that...
    # then it = dividing by the total sum of those counts...
        # this = giving ur TRANSITION probabilities

    counts = pattern_counts.reshape(N_STATES, N_CONTEXTS, 4)
    with_context = counts + pseudocount
    with_context = with_context / with_context.sum(dim=2, keepdim=True)

    # this = calculating P(base | state, previous bases)
        # i.e. probability of nucleotide given the state & previous 2 bases

    no_context = counts.sum(dim=1) + pseudocount
    no_context = (no_context / no_context.sum(dim=1, keepdim=True)).unsqueeze(1).expand(-1, N_CONTEXTS, -1)

    # this = in the case where there = not enough data 2 lean towards a particular state...
        # so this = falling back to the general probabilities

    emissions = torch.where(torch.tensor([s in (EXON, INTRON) for s in range(N_STATES)]).reshape(-1, 1, 1),
                            with_context, no_context)
    emissions = (emissions / N_CONTEXTS).reshape(N_STATES, N_SYMBOLS)

    # this = calculating the emission probabilities

    return starts, edges, ends, emissions

# this = making introns easier to enter
def boost_intron_entry(probs, factor):
    starts, edges, ends, emissions = probs
    edges, ends = edges.clone(), ends.clone()
    for a, b in ((EXON, DONOR_E[0]), (INTRON, ACCEPTOR[0])):
        edges[a, b] *= factor
        total = edges[a].sum() + ends[a]
        edges[a] /= total
        ends[a] /= total
    return starts, edges, ends, emissions

# this = assembling pomegranate model
def build_model(probs, max_iter=BW_MAX_ITER, tol=0.1):
    model_starts, model_edges, model_ends, emissions = probs
    distributions = [Categorical(row.clone().unsqueeze(0), frozen=True) for row in emissions]
    model = DenseHMM(distributions, edges=model_edges.clone(), starts=model_starts.clone(),
                     ends=model_ends.clone(), max_iter=max_iter, tol=tol, verbose=True)
    return model.double()
    # .double() = 64-bit numbers; 32-bit isn't precise enough for genes this long

# this = showing what the model learned
def print_probabilities(title, model):
    trans = torch.exp(model.edges).detach()
    print(f"\n--- {title} ---")
    print(f"  Exon   -> Exon: {trans[EXON, EXON]:.4f}   Exon   -> Donor chain:    {trans[EXON, DONOR_E[0]]:.4f}")
    print(f"  Intron -> Intron: {trans[INTRON, INTRON]:.4f} Intron -> Acceptor chain: {trans[INTRON, ACCEPTOR[0]]:.4f}")
    print(f"  {'Emissions':<11} {'A':>6} {'C':>6} {'G':>6} {'T':>6}   (average over the previous 2 bases)")
    for i, dist in enumerate(model.distributions):
        probs = dist.probs[0].reshape(N_CONTEXTS, 4).sum(dim=0).tolist()
        best = "ACGT"[max(range(4), key=lambda b: probs[b])]
        print(f"  {STATE_NAMES[i]:<11} " + " ".join(f"{p:6.3f}" for p in probs) + f"   {best}")

    # I asked AI to do this --> make it more organizable/formatable

# this = labeling the genes w/ Viterbi
def score_with_viterbi(model, X, Y):
    correct = total = 0
    true_sites = found_sites = predicted_sites = 0

    for x, y in zip(X, Y):
        path = model.viterbi(x.unsqueeze(0))[0] # this = where it = doing calculations w/ transitions & probabilities & whatnot
        predicted = STATE_TO_LABEL[path]   # turn the 19 states back into E / I / S
        correct += (predicted == y).sum().item()
        total += len(y)

        true_pos = torch.where(y == 2)[0]
        pred_pos = torch.where(predicted == 2)[0]
        if len(true_pos) and len(pred_pos):
            dist = (true_pos.unsqueeze(1) - pred_pos.unsqueeze(0)).abs()
            found_sites += (dist.min(dim=1).values <= 2).sum().item()
        true_sites += len(true_pos)
        predicted_sites += len(pred_pos)

    return {"accuracy": correct / total, "found": found_sites,
            "true": true_sites, "predicted": predicted_sites}

# this = testing version w/ Viterbi model
def test_with_viterbi(model, X, Y):
    r = score_with_viterbi(model, X, Y)
    print(f"  bases labeled correctly: {r['accuracy']:.1%}")
    print(f"  real splice sites found: {r['found']} of {r['true']}")
    print(f"  splice sites predicted:  {r['predicted']}")

# this = giving us close-up view of the model...
def show_region(model, x, y, dna, width=70):
    site = torch.where(y == 2)[0][0].item()
    lo = max(0, site - width // 2)
    hi = min(len(dna), lo + width)
    predicted = STATE_TO_LABEL[model.viterbi(x.unsqueeze(0))[0]]
    print(f"\n  DNA        {dna[lo:hi]}")
    print(f"  True       {''.join(label_letter[s] for s in y[lo:hi])}")
    print(f"  Viterbi    {''.join(label_letter[s] for s in predicted[lo:hi])}")

if __name__ == "__main__":

    # Stage 1: load & filter the data
    def load(path):
        recs = load_training_data(path)
        if MAX_GENE_LENGTH is not None:
            recs = [r for r in recs if len(r.dna) <= MAX_GENE_LENGTH]
        return recs

    records = load(training_path())
    random.Random(SEED).shuffle(records)

    # Stage 2: choose training/testing genes & convert into tensors
    if TEST_FILE:
        train_records = records
        test_records = load(training_path().with_name(TEST_FILE))
    else:
        n_test = max(1, round(len(records) * TEST_FRACTION))
        test_records, train_records = records[:n_test], records[n_test:]

    B_train = [torch.tensor(r.dna_index()) for r in train_records]   # plain bases 0-3 (for counting)
    Y_train = [torch.tensor(r.label_index()) for r in train_records]
    X_train = [to_patterns(b) for b in B_train]                       # 3-base patterns (for the model)
    X_test = [to_patterns(torch.tensor(r.dna_index())) for r in test_records]
    Y_test = [torch.tensor(r.label_index()) for r in test_records]

    n_bases = sum(len(x) for x in X_train)
    print(f"Training on {len(train_records)} genes ({n_bases:,} bases), "
          f"testing on {len(test_records)} genes")
    intron_share = sum((y == 1).sum().item() for y in Y_test) / sum(len(y) for y in Y_test)
    print(f"(labeling every test base 'intron' would score {intron_share:.1%})")

    # Stage 3: choose the boost on the validation genes
    n_val = max(1, round(len(train_records) * TEST_FRACTION))
    print(f"\nChoosing the intron-entry boost ({len(train_records) - n_val} genes to count, "
          f"{n_val} to validate):")
    val_probs = count_probabilities(B_train[n_val:], Y_train[n_val:])
    best_boost, best_acc = None, -1
    for factor in BOOST_CHOICES:
        r = score_with_viterbi(build_model(boost_intron_entry(val_probs, factor)),
                               X_train[:n_val], Y_train[:n_val])
        print(f"  boost x{factor:<3} accuracy {r['accuracy']:.1%}, "
              f"sites found {r['found']} of {r['true']}, predicted {r['predicted']}")
        if r["accuracy"] > best_acc:
            best_boost, best_acc = factor, r["accuracy"]
    print(f"  -> using boost x{best_boost}")

    # Stage 4: build final model & test it
    probs = boost_intron_entry(count_probabilities(B_train, Y_train), best_boost)
    model = build_model(probs, tol=1e-5 * n_bases)
    print_probabilities("Counted from training labels", model)

    print("\nViterbi on the TEST genes")
    test_with_viterbi(model, X_test, Y_test)
    show_region(model, X_test[0], Y_test[0], test_records[0].dna)

    # Stage 5 (optional): Baum-Welch adjusts the transitions using the DNA only
    if RUN_BAUM_WELCH:
        print("\nBaum-Welch training (transitions only)")
        model.fit(X_train)
        print_probabilities("After Baum-Welch training", model)
        print("\nViterbi on the TEST genes after Baum-Welch")
        test_with_viterbi(model, X_test, Y_test)

    # Stage 6: save the probabilities & decode the original test sequence
    torch.save({
        "starts": torch.exp(model.starts).detach(),
        "edges": torch.exp(model.edges).detach(),
        "ends": torch.exp(model.ends).detach(),
        "emissions": torch.stack([d.probs[0].detach() for d in model.distributions]),
    }, "trained_model.pt")
    print("\nSaved trained probabilities to trained_model.pt")

    test_sequence = "ACGCGTACGGTTTTTAAATTT"
    pattern_tensor = to_patterns(torch.tensor(encode_dna(test_sequence))).unsqueeze(0)
    path = STATE_TO_LABEL[model.viterbi(pattern_tensor)[0]]
    print(f"\nOriginal test sequence: {test_sequence}")
    print(f"Viterbi labels:         {''.join(label_letter[s] for s in path)}")