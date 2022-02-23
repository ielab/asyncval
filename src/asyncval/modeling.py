import torch
from typing import Dict
from torch import Tensor
from tevatron.modeling import DenseModelForInference
from asyncval.arguments import AsyncvalArguments


class Encoder(torch.nn.Module):
    def __init__(self, ckpt_path: str, async_args: AsyncvalArguments):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.async_args = async_args

    @torch.no_grad()
    def encode_passage(self, psg: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def encode_query(self, qry: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError


class DenseModel(Encoder):
    def __init__(self, ckpt_path, async_args):
        super(DenseModel, self).__init__(ckpt_path, async_args)
        self.model = DenseModelForInference.build(model_name_or_path=self.ckpt_path,
                                                  cache_dir=self.async_args.cache_dir)

    def encode_passage(self, psg):
        return self.model.encode_passage(psg)[1]

    def encode_query(self, qry):
        return self.model.encode_query(qry)[1]
