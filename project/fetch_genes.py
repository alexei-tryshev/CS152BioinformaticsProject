import time
from Bio import Entrez, SeqIO
# you need "pip install biopython"
# allows us to access the NCBI database...

Entrez.email = "rayhannkwon@gmail.com"

N_GENES = 50

GENES = []

def wait():
    time.sleep(0.4) # NCBI allows only like ~3 requests/second w/o an API key

def search_gene_id(n): #search gene database for protein-coding human genes
    query = ('"Homo sapiens"[organism] AND alive[property] '
             'AND "genetype protein coding"[property]')
    with Entrez.esearch(db="gene", term=query, retmax=n) as handle:
        result = Entrez.read(handle)
    wait()
    return result["IdList"]

def symbol_to_gene_id(symbol): #converts name of gene into its corresponding gene ID
    term = f'{symbol}[gene] AND "Homo sapiens"[organism]'
    with Entrez.esearch(db="gene", term=term) as handle:
        ids = Entrez.read(handle)["IdList"]
    wait()
    return ids[0] if ids else None

# gets the chromosome location, where the gene is located on the chromosome & exon counts
# ex: ('BRCA1', {'ChrLoc': '17', 'ChrAccVer': 'NC_000017.11', 'ChrStart': '43170326', 'ChrStop': '43044294', 'ExonCount': '31'})
    # this = for the gene BRCA1 (just an example)
def gene_summary(gene_id):
    with Entrez.esummary(db="gene", id=gene_id) as handle:
        summary = Entrez.read(handle)["DocumentSummarySet"]["DocumentSummary"][0]
    wait()
    loc = summary["GenomicInfo"][0] if summary["GenomicInfo"] else None
    return summary["Name"], loc

# this gives u the genomic reference for a certain gene...
# i.e. other related genes / variants, their accession numbers, etc. 
def find_refseqgene(gene_id):
    with Entrez.elink(dbfrom="gene", db="nuccore", id=gene_id,
                      linkname="gene_nuccore_refseqgene") as handle:
        result = Entrez.read(handle)
    wait()
    for linkset in result:
        for link_db in linkset.get("LinkSetDb", []):
            linked_ids = [item['Id'] for item in link_db.get('Link', [])]
            return linked_ids
    return None 
# this CAN return MULTIPLE nuccore_ids

# this gets the genbank of the gene (from the nuccore_id) from find_refseqgene
    # i.e. it gets the description, sequence, accession #, etc.
    # we're trying to get the features out of it (esp exons, introns, splice sites)
def fetch_genbank(nuccore_id, seq_start=None, seq_stop=None):
    params = dict(db="nuccore", id=nuccore_id, rettype="gbwithparts", retmode="text")
    if seq_start is not None:
        params.update(seq_start=seq_start, seq_stop=seq_stop)
    with Entrez.efetch(**params) as handle: # unpacks the dictionary
        record = SeqIO.read(handle, "genbank")
    wait()
    return record

# this gives us the gene record (i.e. gene name, sequence, id, etc.)
# cool thing abt this is the transcript part (i.e. mRNA) = telling HOW to make the mRNA transcript
    # i.e. basically pulls parts of the DNA sequence & assembles those to make the mRNA sequence
# basically, it gets GenBank's records (i.e. summarized version of GenBank)
def fetch_gene_record(gene_id):
    symbol, loc = gene_summary(gene_id)

    refseqgene_id = find_refseqgene(gene_id)
    if refseqgene_id:
        return symbol, fetch_genbank(refseqgene_id)

    if loc is None: # fetches the GenBank data (the long way)
        return symbol, None
    start, stop = int(loc["ChrStart"]), int(loc["ChrStop"])
    lo, hi = min(start, stop) + 1, max(start, stop) + 1
    record = fetch_genbank(loc["ChrAccVer"], seq_start=lo, seq_stop=hi)
    if (start > stop):
        record = record.reverse_complement()
        # this = important because the reverse complement will result in something different
    return symbol, record

# this gives us the mRNA transcript of the record (from the fetch_genbank method)
# we use MANE Select since that's the standard main transcript...
    # this also gives us the exon boundaries, which we can use to label the data
def pick_transcript(record, symbol=None):
    candidates = [
        f for f in record.features
        if f.type == "mRNA" and f.location.strand == 1
        and (symbol is None or f.qualifiers.get("gene", [None][0] == symbol))
    ]
    for f in candidates:
        if any("MANE Select" in tag for tag in f.qualifiers.get("tag", [])):
            return f
    return candidates[0] if candidates else None

# this basically labels the DNA sequence based on exons, introns, & splice sites
# gets the transcript to determine the exon boundaries
    # basically convert the entire sequence as introns, THEN overwrite exon positions w/ E
    # any space in between are labeled as "S" (the splice sites between introns & exons)
def label_record(record, symbol=None):
    feature = pick_transcript(record, symbol)
    if feature is None:
        return None

    exons = sorted(feature.location.parts, key=lambda p: p.start)
    if len(exons) < 2:
        return None

    gene_start, gene_end = int(exons[0].start), int(exons[-1].end)
    dna = str(record.seq[gene_start:gene_end]).upper()
    labels = ["I"] * len(dna)
    for exon in exons:
        for pos in range(int(exon.start), int(exon.end)):
            labels[pos - gene_start] = "E"

    for exon in exons[:-1]:
        labels[int(exon.end) - gene_start] = "S"

    if set(dna) - set("ACGT"):
        return None
    return dna, "".join(labels)

if __name__ == "__main__":
    if GENES:
        gene_ids = [g if g.isDigit() else symbol_to_gene_id(g) for g in GENES]
        gene_ids = [g for g in gene_ids if g]
        # if you alr specified the genes above, it uses them
    else:
        gene_ids = search_gene_id(N_GENES * 3)
        # if not, it searches a bunch of genes (w/ more than the N_GENES amount)
            # we = looking @ more since some might not be usable...

    training_data = []
    for gene_id in gene_ids:
        try:
            symbol, record = fetch_gene_record(gene_id)
            # gets the summarized version of GenBank of that gene ID
            # symbol = Gene name; record = summarized GenBank record
                # remember - fetch_gene_record = returning two things

            result = label_record(record, symbol) if record else None
            # labels the DNA sequence (from symbol & record)
            # i.e. labels exons, introns, & splice sites...
        except Exception as e:
            print(f"Gene {gene_id}: error ({e}), skipping")
            continue

        if result is None:
            print(f"Gene {gene_id}: no usable transcript, skipping")
            continue

        dna, label = result
        training_data.append(result) # adds the result (i.e. ur labeled data) to the training data
        print(f"{symbol} ({record.id}): {len(dna):,} bases, {label.count('S')} introns")
        if len(training_data) == N_GENES:
            break
            # if it reaches the specified # of genes --> it stops...

    with open("training_genes.tsv", "w") as f:
        for dna, labels in training_data:
            f.write(f"{dna}\t{labels}\n")
    print(f"\nSaved {len(training_data)} genes to training_genes.tsv")
