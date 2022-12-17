import os
import torch.nn as nn
from transformers import BertForSeauenceClassification

BASE_DIR = os.path.abspath(os.getcwd())
HIDDEN_DIM = 768

class Model(nn.Module):
    """Classification model."""

    def __init__(self, roberta_path: str, num_classes: int):
        """Set model."""
        super(Model, self).__init__()
        self.calssify_model = BertForSeauenceClassification.from_pretrained(
            roberta_path,
            num_labels=num_classes,
            output_attention=False,
            output_hidden_states=False,
        )

    def forword(self, inputs:dict):
        """Get result.

        :param inputs:input paramters
        :return: probability
        """
        outputs = self.calssify_model(**inputs)
        return outputs

