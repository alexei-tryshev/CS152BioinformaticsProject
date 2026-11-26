import time
from pathlib import Path
from Bio import Entrez, SeqIO
# you need "pip install biopython"
# allows us to access the NCBI database...

Entrez.email = "rayhannkwon@gmail.com"

N_GENES = 200
# how many NEW genes to download

GENES = []
# optional: specific genes, e.g. ["BRCA1", "TP53", "672"]. Leave empty to search automatically

SEARCH_START = 150
# skip the first 150 search results (my first run already used those), so we get different genes

EXISTING_FILE = "training_genes.tsv"
# genes already in this file are skipped, so the new file has no duplicates

OUTPUT_FILE = "training_genes_new.tsv"
# where the new genes go. If this file exists, a number is added (training_genes_new_2.tsv, ...)
# so nothing is ever overwritten

def wait():
    time.sleep(0.4) # NCBI allows only like ~3 requests/second w/o an API key

def search_gene_id(n, start=0): #search gene database for protein-coding human genes
    query = ('"Homo sapiens"[organism] AND alive[property] '
             'AND "genetype protein coding"[property]')
    # retstart = which result to start from (0 = the very first one)
    with Entrez.esearch(db="gene", term=query, retmax=n, retstart=start) as handle:
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

# this finds the gene's curated RefSeqGene record (accession starts with NG_)
def find_refseqgene(gene_id):
    with Entrez.elink(dbfrom="gene", db="nuccore", id=gene_id,
                      linkname="gene_nuccore_refseqgene") as handle:
        result = Entrez.read(handle)
    wait()
    for linkset in result:
        for link_db in linkset.get("LinkSetDb", []):
            linked_ids = [item['Id'] for item in link_db.get('Link', [])]
            if linked_ids:
                return linked_ids[0]
                # FIX: return ONE id. fetch_genbank uses SeqIO.read, which errors
                # if NCBI sends back more than one record
    return None

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
        record = record.reverse_complement(features=True)
        # FIX: features=True flips the exon positions too
            # without it, the flipped record has NO features, so the gene gets skipped
    return symbol, record

# this gives us the mRNA transcript of the record (from the fetch_genbank method)
# we use MANE Select since that's the standard main transcript...
    # this also gives us the exon boundaries, which we can use to label the data
def pick_transcript(record, symbol=None):
    candidates = [
        f for f in record.features
        if f.type == "mRNA" and f.location.strand == 1
        and (symbol is None or f.qualifiers.get("gene", [None])[0] == symbol)
        # FIX: the bracket goes after [None], so we compare the gene NAME to the symbol
            # before, the comparison happened inside .get(), so it never filtered anything
    ]
    for f in candidates:
        if any("MANE Select" in tag for tag in f.qualifiers.get("tag", [])):
            return f
    return candidates[0] if candidates else None

# this basically labels the DNA sequence based on exons, introns, & splice sites
    # start w/ everything as intron, THEN overwrite exon positions w/ E,
    # THEN mark the first base of every intron as S (the splice site)
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

    labels = "".join(labels)
    if set(dna) - set("ACGT"):
        return None   # skip genes w/ N or other unknown bases
    if "SE" in labels or "EE" not in labels:
        return None   # skip genes w/ a 1-base "intron" (annotation glitch) or no real exons
    return dna, labels

# reads the DNA of every gene already in the existing file (so we can skip them)
def load_existing_dna(path):
    path = Path(path)
    if not path.is_file():
        return set()
    with path.open() as f:
        return {line.split("\t")[0] for line in f if line.strip()}

# picks a file name that doesn't exist yet, so we never overwrite anything
def new_file_path(name):
    path = Path(name)
    number = 2
    while path.exists():
        path = Path(f"{Path(name).stem}_{number}{Path(name).suffix}")
        number += 1
    return path

if __name__ == "__main__":
    if GENES:
        gene_ids = [g if g.isdigit() else symbol_to_gene_id(g) for g in GENES]
        # FIX: isdigit (lowercase d)
        gene_ids = [g for g in gene_ids if g]
    else:
        gene_ids = search_gene_id(N_GENES * 3, start=SEARCH_START)
        # ask for 3x as many, since some won't be usable

    existing = load_existing_dna(EXISTING_FILE)
    print(f"{len(existing)} genes already in {EXISTING_FILE} will be skipped")

    out_path = new_file_path(OUTPUT_FILE)
    list_path = out_path.with_name(out_path.stem + "_list.txt")
    # the .tsv keeps the same 2 columns (DNA, labels) so data_loader.py can read it.
    # the _list.txt file records WHICH gene each line is, in the same order

    saved = 0
    with open(out_path, "w") as data_file, open(list_path, "w") as list_file:
        for gene_id in gene_ids:
            try:
                symbol, record = fetch_gene_record(gene_id)
                result = label_record(record, symbol) if record else None
            except Exception as e:
                print(f"Gene {gene_id}: error ({e}), skipping")
                continue

            if result is None:
                print(f"Gene {gene_id}: no usable transcript, skipping")
                continue

            dna, labels = result
            if dna in existing:
                print(f"{symbol}: already have this gene, skipping")
                continue
            existing.add(dna)   # also stops the same gene being saved twice in this run

            # write each gene right away, so a crash halfway doesn't lose everything
            data_file.write(f"{dna}\t{labels}\n")
            list_file.write(f"{symbol}\t{gene_id}\t{record.id}\n")
            data_file.flush()
            list_file.flush()
            saved += 1
            print(f"[{saved}/{N_GENES}] {symbol} ({record.id}): {len(dna):,} bases, {labels.count('S')} introns")
            if saved == N_GENES:
                break

    print(f"\nSaved {saved} new genes to {out_path}")
    print(f"Gene names are in {list_path}")