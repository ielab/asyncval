from argparse import ArgumentParser
import logging
from tqdm import tqdm
import datasets
import json
import os
from multiprocessing import Pool
logger = logging.getLogger(__name__)


class Processor:
    def __init__(self, uniq_ids):
        self.uniq_ids = uniq_ids
        self.input_keys = ['text_id', 'text']

    def process_doc(self, data):
        text_id, text = (data[k] for k in self.input_keys)
        if str(text_id) in self.uniq_ids:
            encoded = {
                'text_id': text_id,
                'text': text
            }
            return json.dumps(encoded)
        return None


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
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    uniq_ids = list(read_run_file(args.run_file, args.qrel_file, args.cutoff))
    processor = Processor(uniq_ids)
    logger.info(f"Total number of documents: {len(uniq_ids)}")
    candidate_data = datasets.load_dataset(
        'json',
        data_files=args.candidate_file,
        cache_dir=args.cache_dir
    )['train']

    data = tqdm(candidate_data, desc="Filtering candidate documents")
    with Pool() as p:
        with open(os.path.join(args.output_dir, f'candidate_docs.jsonl'), 'w') as df:
            for d in p.imap(processor.process_doc, data, chunksize=500):
                if d is not None:
                    df.write(d + '\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--candidate_file', required=True)
    parser.add_argument('--run_file', required=True)
    parser.add_argument('--qrel_file', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--cache_dir', default=None)
    parser.add_argument('--cutoff', type=int, default=100)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
