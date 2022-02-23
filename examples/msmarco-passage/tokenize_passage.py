from argparse import ArgumentParser
from transformers import AutoTokenizer
import datasets
from multiprocessing import Pool
from tqdm import tqdm
import json
import os


class Processor:
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer

        columns = ['text_id', 'title', 'text']
        self.collection = datasets.load_dataset(
            'csv',
            data_files=args.collection_file,
            column_names=columns,
            delimiter='\t',
            cache_dir=args.cache_dir
        )['train']

    def process_doc(self, did):
        title = self.collection[int(did)]['title']
        text = self.collection[int(did)]['text']
        text = f'{title} {self.tokenizer.sep_token} {text}'
        text_encoded = self.tokenizer.encode(
            text,
            max_length=128,  # hard code for now
            truncation=True
        )
        encoded = {
            'text_id': str(did),
            'text': text_encoded
        }

        return json.dumps(encoded)


def read_run_file(run_file, qrel_file, cutoff=100):
    uniq_dids = set()
    with open(run_file) as rf:
        lines = rf.readlines()
        rank = 0
        temp_qid = None
        for line in tqdm(lines, desc='Loading run file'):
            qid, did, _ = line.strip().split('\t')
            if qid != temp_qid:
                temp_qid = qid
                rank = 0
            if rank < cutoff:
                uniq_dids.add(did)
            rank += 1

    with open(qrel_file) as qf:
        lines = qf.readlines()
        for line in tqdm(lines, desc='Loading qrel file'):
            qid, _, did, _ = line.strip().split('\t')
            uniq_dids.add(did)

    return uniq_dids


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path,
                                              cache_dir=args.cache_dir,
                                              use_fast=True)
    processor = Processor(tokenizer=tokenizer, args=args)
    if args.process_all:
        uniq_ids = processor.collection['text_id']
    else:
        uniq_ids = list(read_run_file(args.run_file, args.qrel_file, args.cutoff))
    print(f"Total number of documents: {len(uniq_ids)}")
    ids = tqdm(uniq_ids, desc="Tokenizing candidate documents")
    with Pool() as p:
        with open(os.path.join(args.save_to, f'candidate_docs.jsonl'), 'w') as df:
            for d in p.imap(processor.process_doc, ids, chunksize=500):
                df.write(d + '\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--tokenizer_name_or_path', required=True)
    parser.add_argument('--run_file', required=True)
    parser.add_argument('--qrel_file', required=True)
    parser.add_argument('--collection_file', required=True)
    parser.add_argument('--save_to', required=True)
    parser.add_argument('--cutoff', type=int, default=100)
    parser.add_argument('--process_all', action="store_true", help='Whether to process the whole collection, if set, will ignore run and qrel file.')
    parser.add_argument('--cache_dir', default='cache')

    args = parser.parse_args()
    os.makedirs(args.save_to, exist_ok=True)
    main(args)
