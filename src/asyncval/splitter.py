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


def read_run_file(run_file, qrel_file, run_type, depth=100):
    uniq_dids = set()
    with open(run_file) as rf:
        lines = rf.readlines()
        rank = 0
        temp_qid = None
        for line in tqdm(lines, desc='Loading run file'):
            if run_type == 'msmarco':
                qid, did, _ = line.strip().split('\t')
            elif run_type == 'trec':
                qid, _, did, _, _, _ = line.strip().split(' ')
            else:
                raise KeyError(f"{run_type} is unknown run type.")

            if qid != temp_qid:
                temp_qid = qid
                rank = 0
            if rank < depth:
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

    uniq_ids = list(read_run_file(args.run_file, args.qrel_file, args.run_type, args.depth))
    processor = Processor(uniq_ids)
    logger.info(f"Total number of documents: {len(uniq_ids)}")
    files = os.listdir(args.candidate_dir)
    candidate_files = [
        os.path.join(args.candidate_dir, f) for f in files if f.endswith('json')
    ]
    candidate_data = datasets.load_dataset(
        'json',
        data_files=candidate_files,
        cache_dir=args.cache_dir
    )['train']

    n_lines = len(uniq_ids)
    if n_lines % args.num_splits == 0:
        split_size = int(n_lines / args.num_splits)
    else:
        split_size = int(n_lines / args.num_splits) + 1
    print(split_size)
    os.makedirs(args.output_dir, exist_ok=True)
    current_num = 0
    current_split_num = 0
    f = open(os.path.join(args.output_dir, f'split{current_split_num:02d}.json'), 'w')

    data = tqdm(candidate_data, desc="Sampling corpus subset")
    with Pool() as p:
        for d in p.imap(processor.process_doc, data, chunksize=500):
            if d is not None:
                f.write(d + '\n')
                current_num += 1
                if current_num % split_size == 0:
                    f.close()
                    current_split_num += 1
                    current_num = 0
                    f = open(os.path.join(args.output_dir, f'split{current_split_num:02d}.json'), 'w')
    f.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--candidate_dir', required=True)
    parser.add_argument('--run_file', required=True)
    parser.add_argument('--qrel_file', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--cache_dir', default=None)
    parser.add_argument('--depth', type=int, default=100)
    parser.add_argument('--num_splits', type=int, default=1)
    parser.add_argument('--run_type', type=str, default='msmarco')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
