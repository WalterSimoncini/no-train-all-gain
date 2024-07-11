import torch.nn as nn

from src.enums import ModelType

from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, T5EncoderModel


def get_model(type_: ModelType):
    tokenizer = None

    if type_ == ModelType.BERT_BASE_UNCASED:
        model = BertModel.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        model.pooler = nn.Identity()
    elif type_ == ModelType.T5_SMALL:
        model = T5EncoderModel.from_pretrained("google-t5/t5-small")
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    else:
        raise ValueError(f"invalid model type {type_}")

    return model, tokenizer
