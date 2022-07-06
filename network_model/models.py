from torch import nn
from transformers import BertTokenizer, BertModel


class BertClassifier(nn.Module):
    """BERT model for classification of text."""
    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()
