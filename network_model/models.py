import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import BertTokenizer, BertModel, HubertModel


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(dropout)
        # 10 categories
        self.linear = nn.Linear(768, 10)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


# Models
class HubertClassifier(nn.Module):
    def __init__(self, dropout=0.5):

        super(HubertClassifier, self).__init__()

        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.dropout = nn.Dropout(dropout)
        # 10 categories
        self.linear = nn.Linear(768, 10)
        self.relu = nn.ReLU()

    def forward(self, input_values):

        output_tuple = self.hubert(input_values=input_values, return_dict=False)
        (pooled_output,) = output_tuple
        print("pooled", pooled_output.size())
        output_reduced = torch.mean(pooled_output, -2)
        print("reduced", output_reduced.size())
        dropout_output = self.dropout(output_reduced)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer
