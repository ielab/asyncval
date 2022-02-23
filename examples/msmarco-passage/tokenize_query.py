from argparse import ArgumentParser
from transformers import AutoTokenizer
import datasets
from tqdm import tqdm
import json
import os


class Processor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def process_pair(self, did, text, max_length):
        text_encoded = self.tokenizer.encode(
            text,
            max_length=max_length,
            truncation=True
        )
        encoded = {
            'text_id': did,
            'text': text_encoded
        }
        return json.dumps(encoded)


def read_queries(query_file):
    qmap = {}
    with open(query_file) as f:
        for l in f:
            qid, qry = l.strip().split('\t')
            qmap[qid] = qry
    return qmap


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path,
                                              cache_dir=args.cache_dir,
                                              use_fast=True)
    processor = Processor(tokenizer=tokenizer)
    with open(args.query_file, 'r') as f:
        lines = f.readlines()
        with open(os.path.join(args.save_to, f'queries.jsonl'), 'w') as qf:
            for line in tqdm(lines, desc="Tokenizing queries"):
                qid, text = line.strip().split('\t')
                q = processor.process_pair(qid, text, 32)
                qf.write(q + '\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--tokenizer_name_or_path', required=True)
    parser.add_argument('--query_file', required=True)
    parser.add_argument('--save_to', required=True)
    parser.add_argument('--cache_dir', default='cache')

    args = parser.parse_args()
    os.makedirs(args.save_to, exist_ok=True)
    main(args)
