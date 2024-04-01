"""Models for training features, text and audio for confidence classification.

----------------------------------
Class BertClassifier: Model for text with Bert tokenized tensors classifier.
Class HubertClassifier: Model for audio with HuBert tokenized tensors classifier.
Class SelectFeaturesClassifier: Model for features as input tensors.
Class CustomBERTSimpleModel: Model for text with Bert.
Class CustomHUBERTModel: Model for audio with Hubert and Bi-LSTM.
Class CustomHUBERTSimpleModel: Model for audio with Hubert.
Class CustomMultiModel: Model for text and audio with Bi-LSTM.
Class CustomMultiModel: Model for text and audio.
"""

import torch
from torch import nn
from transformers import BertModel, HubertModel


# Final version of file used
class CustomMultiModelSimplePooled(nn.Module):
    def __init__(self, dropout=0.5):
        super(CustomMultiModelSimplePooled, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.relu = nn.ReLU()
        self.layernorm1 = nn.LayerNorm([4, 768 * 2])
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768 * 2, 32)
        self.layernorm2 = nn.LayerNorm([4, 32])
        self.linear2 = nn.Linear(32, 1)
        self.tanh = nn.Tanh()

    def forward(self, input_values, input_id, mask):
        ## Bert transform
        sequence_output_bert, pooled_output_bert = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )

        ## Hubert transform
        output_tuple_hubert = self.hubert(input_values=input_values, return_dict=False)
        (pooled_output_hubert,) = output_tuple_hubert
        pooled_hubert = torch.mean(pooled_output_hubert, dim=1)

        # Concat the two models
        concat = torch.cat((pooled_output_bert, pooled_hubert), dim=1)
        concat_norm = self.layernorm1(concat)
        dropout1 = self.dropout(concat_norm)

        linear1 = self.linear1(dropout1)
        relu1 = self.relu(linear1)
        dropout2 = self.dropout(relu1)
        dropout2_norm = self.layernorm2(dropout2)
        linear2 = self.linear2(dropout2_norm)

        tanh = self.tanh(linear2)
        # Scale to match input
        prediction = tanh * 2.5

        return prediction


# Final version of file used
class CustomMultiModelSimplePooledAudio(nn.Module):
    def __init__(self, dropout=0.5):
        super(CustomMultiModelSimplePooledAudio, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.relu = nn.ReLU()
        self.layernorm1 = nn.LayerNorm([4, 768 * 2])
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768 * 2, 32)
        self.layernorm2 = nn.LayerNorm([4, 32])
        self.linear2 = nn.Linear(32, 1)
        self.tanh = nn.Tanh()

    def forward(self, input_values, input_id, mask):
        ## Bert transform

        sequence_output_bert, pooled_output_bert = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )

        ## Hubert transform

        output_tuple_hubert = self.hubert(input_values=input_values, return_dict=False)
        (pooled_output_hubert,) = output_tuple_hubert

        pooled_hubert = torch.mean(pooled_output_hubert, dim=1)
        # Set audio representation to zero
        pooled_hubert_zeroed = torch.zeros(pooled_hubert.size()).cuda()

        # Concat the two models
        concat = torch.cat((pooled_output_bert, pooled_hubert_zeroed), dim=1)
        concat_norm = self.layernorm1(concat)
        dropout1 = self.dropout(concat_norm)

        linear1 = self.linear1(dropout1)
        relu1 = self.relu(linear1)
        dropout2 = self.dropout(relu1)
        dropout2_norm = self.layernorm2(dropout2)
        linear2 = self.linear2(dropout2_norm)
        tanh = self.tanh(linear2)
        # Scale to match input
        prediction = tanh * 2.5

        return prediction


# Final version of file used
class CustomMultiModelSimplePooledText(nn.Module):
    def __init__(self, dropout=0.5):
        super(CustomMultiModelSimplePooledText, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.relu = nn.ReLU()
        self.layernorm1 = nn.LayerNorm([4, 768 * 2])
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768 * 2, 32)
        self.layernorm2 = nn.LayerNorm([4, 32])
        self.linear2 = nn.Linear(32, 1)
        self.tanh = nn.Tanh()

    def forward(self, input_values, input_id, mask):
        ## Bert transform

        sequence_output_bert, pooled_output_bert = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        # Set text representation to zero
        pooled_bert_zeroed = torch.zeros(pooled_output_bert.size()).cuda()

        ## Hubert transform

        output_tuple_hubert = self.hubert(input_values=input_values, return_dict=False)
        (pooled_output_hubert,) = output_tuple_hubert

        pooled_hubert = torch.mean(pooled_output_hubert, dim=1)

        # Concat the two models
        concat = torch.cat((pooled_bert_zeroed, pooled_hubert), dim=1)
        concat_norm = self.layernorm1(concat)
        dropout1 = self.dropout(concat_norm)

        linear1 = self.linear1(dropout1)
        relu1 = self.relu(linear1)
        dropout2 = self.dropout(relu1)
        dropout2_norm = self.layernorm2(dropout2)

        linear2 = self.linear2(dropout2_norm)

        tanh = self.tanh(linear2)
        # Scale to match input
        prediction = tanh * 2.5

        return prediction
