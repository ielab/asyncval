from dataclasses import dataclass, field
from typing import List


@dataclass
class AsyncvalArguments:
    ckpts_dir: str = field(
        default=None, metadata={"help": "Path to the folder where model ckpts are saved."}
    )
    query_files: List[str] = field(
        default=None, metadata={"help": "Path to query files."}
    )

    qrel_file: str = field(
        default=None, metadata={"help": "Path to qrel file."}
    )
    candidate_file: str = field(
        default=None, metadata={"help": "Path to candidate file."}
    )
    cache_dir: str = field(
        default=None, metadata={"help": "Cache folder."}
    )
    tokenizer_name_or_path: str = field(
        default=None, metadata={"help": "Tokenizer"}
    )

    q_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    p_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_num_valid: int = field(
        default=-1, metadata={"help": "Max number of checkpoints to validate."}
    )

    retrieve_batch_size: int = field(default=64, metadata={"help": "Faiss retrieval batch size."})
    depth: int = field(default=100, metadata={"help": "Ranking depth."})

    metrics: List[str] = field(
        default_factory=lambda: ['mrr'], metadata={"help": "The list of metrics for validation."}
    )

    write_run: bool = field(
        default=False, metadata={"help": "Write the run file to disk or not"}
    )