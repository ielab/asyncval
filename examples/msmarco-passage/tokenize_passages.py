from argparse import ArgumentParser
from transformers import AutoTokenizer, PreTrainedTokenizer
import os
from tqdm import tqdm
from multiprocessing import Pool
import json
from dataclasses import dataclass


@dataclass
class SimpleCollectionPreProcessor:
    tokenizer: PreTrainedTokenizer
    separator: str = '\t'
    max_length: int = 128

    def process_line(self, line: str):
        xx = line.strip().split(self.separator)
        text_id, text = xx[0], xx[1:]
        text_encoded = self.tokenizer.encode(
            ' '.join(text),
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True
        )
        encoded = {
            'text_id': text_id,
            'text': text_encoded
        }
        return json.dumps(encoded)


parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--truncate', type=int, default=128)
parser.add_argument('--file', required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--n_splits', type=int, default=10)

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
processor = SimpleCollectionPreProcessor(tokenizer=tokenizer, max_length=args.truncate)

with open(args.file, 'r') as f:
    lines = f.readlines()

n_lines = len(lines)
if n_lines % args.n_splits == 0:
    split_size = int(n_lines / args.n_splits)
else:
    split_size = int(n_lines / args.n_splits) + 1


os.makedirs(args.save_to, exist_ok=True)
with Pool() as p:
    for i in range(args.n_splits):
        with open(os.path.join(args.save_to, f'split{i:02d}.json'), 'w') as f:
            pbar = tqdm(lines[i*split_size: (i+1)*split_size])
            pbar.set_description(f'split - {i:02d}')
            for jitem in p.imap(processor.process_line, pbar, chunksize=500):
                f.write(jitem + '\n')


