from pathlib import Path

DNA_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}
LABEL_TO_INT = {"E": 0, "I": 1, "S": 2}

class TrainingRecord:
    def __init__(self, dna, labels):
        dna = dna.strip().upper()
        labels = labels.strip().upper()

        if not dna:
            raise ValueError("DNA sequence is empty")
        if not labels:
            raise ValueError("label sequence is empty")

        if len(dna) != len(labels):
            raise ValueError("DNA and Label sequence length are not equal")

        invalid_dna = sorted(set(dna) - set(DNA_TO_INT))
        if invalid_dna:
            raise ValueError("DNA has unrecognized symbols")

        invalid_labels = sorted(set(labels) - set(LABEL_TO_INT))
        if invalid_labels:
            raise ValueError("Label has unrecognized symbols")
        self.dna = dna
        self.labels = labels

    def dna_index(self):
        return encode_dna(self.dna)

    def label_index(self):
        return encode_labels(self.labels)

def encode_dna(dna):
    dna = dna.strip().upper()

    invalid = sorted(set(dna) - set(DNA_TO_INT))
    if invalid:
        raise ValueError("DNA contains unrecognized symbols")

    return [DNA_TO_INT[base] for base in dna]

def encode_labels(labels):
    labels = labels.strip().upper()

    invalid = sorted(set(labels) - set(LABEL_TO_INT))
    if invalid:
        raise ValueError("labels contain unrecognized symbols")

    return [LABEL_TO_INT[label] for label in labels]

def training_path():
    return Path(__file__).resolve().parents[1] / "training_genes.tsv"

def load_training_data(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError("Training data file not found")

    records = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")

            if not line.strip():
                continue

            fields = line.split("\t")

            if len(fields) != 2:
                raise ValueError("improper number of fields")

            try:
                record = TrainingRecord(fields[0], fields[1])
                records.append(record)
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: {error}"
                ) from error

    if not records:
        raise ValueError("Training data file is empty")

    return records

def main():
    records = load_training_data(training_path())

    total_bases = sum(len(record.dna) for record in records)

    print("Loaded", len(records), "training records")
    print("Loaded", total_bases,  "DNA bases")
    print("First record length:", len(records[0].dna), "bases")
    print("First record labels", len(records[0].labels))
    print("DNA encoding: A=0, C=1, G=2, T=3")
    print("Label encoding: E=0, I=1, S=2")

if __name__ == "__main__":
    main()