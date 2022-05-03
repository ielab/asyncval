from argparse import ArgumentParser
from transformers import AutoTokenizer, PreTrainedTokenizer
import os
from tqdm import tqdm
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
parser.add_argument('--truncate', type=int, default=32)
parser.add_argument('--query_file', required=True)
parser.add_argument('--save_to', required=True)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
processor = SimpleCollectionPreProcessor(tokenizer=tokenizer, max_length=args.truncate)

with open(args.query_file, 'r') as f:
    lines = f.readlines()

os.makedirs(os.path.split(args.save_to)[0], exist_ok=True)
with open(args.save_to, 'w') as jfile:
    for x in tqdm(lines):
        q = processor.process_line(x)
        jfile.write(q + '\n')
