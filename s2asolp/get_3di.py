# https://github.com/westlake-repl/SaProt/blob/main/utils/foldseek_util.py
import numpy as np
import pandas as pd
import os, glob
import Bio.SeqIO as SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq


def get_3di(folder_path, save_name, chains=None):
    """
    get 3di sequence from the tsv files in the folder
    folder_path: the protein's 3di file folder
    save_name: the name of the file to save
    """
    tsv_files = glob.glob(os.path.join(folder_path, "*.tsv"))
    struct_list = []
    for file_path in tsv_files:

        struct_dict = {}
        name = os.path.basename(file_path)
        seq_name = name.split(".")[0]
        with open(file_path, "r") as r:
            seq_dict = {}
            for i, line in enumerate(r):
                desc, seq, struc_seq = line.split("\t")[:3]

                name_chain = desc.split(" ")[0]
                chain = name_chain.replace(name, "").split("_")[-1]
                seq_dict[chain] = (seq, struc_seq)

            struct_dict[seq_name] = seq_dict
        struct_list.append(struct_dict)

    seqs = []
    structs = []
    for items_dict in struct_list:
        for name, item in items_dict.items():
            for k, v in item.items():
                seqs.append(v[0])
                structs.append(v[1])
    df = pd.DataFrame({"sequence": seqs, "3di": structs})
    df.to_csv(f"{save_name}.csv", index=False)
    print(struct_list[:3])
    return struct_list


def merge_data(file_name, labels_file, save_name):
    """
    merge the sequence and 3di file as input file
    file_name: the file to merge
    save_name: the name of the file to save
    """
    df = pd.read_csv(file_name)
    seq = df["sequence"].values.tolist()
    struc_seq = df["3di"].values.tolist()
    combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
    label_df = pd.read_csv(labels_file)
    labels = label_df["label"].values.tolist()

    df = pd.DataFrame({"label": labels, "comb_seq": combined_seq})
    df.to_csv(f"{save_name}.csv", index=False)


if __name__ == "__main__":
    get_3di(folder_path="", save_name="", chains=None)
    merge_data(file_name="", labels_file="", save_name="")
