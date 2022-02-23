# Asyncval
Asyncval: A toolkit for asynchronously validating dense retriever checkpoints during training.

## Installation
For customized dense retriever encoders, clone this repo and install as editable,

```
git clone https://github.com/ArvinZhuang/asyncval.git
cd asyncval
pip install --editable .
```
> Note: The current code base has been tested with, `torch==1.10.0`, `transformers==4.5.1`, `datasets==1.16.1`, `faiss-cpu==1.7.1`, `python==3.7`