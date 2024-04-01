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
        # print("bert input size", input_id.size())
        sequence_output_bert, pooled_output_bert = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )

        ## Hubert transform
        # print("hubert input size", input_values.size())
        output_tuple_hubert = self.hubert(input_values=input_values, return_dict=False)
        (pooled_output_hubert,) = output_tuple_hubert
        # print("pooled hubert", pooled_output_hubert.size())
        pooled_hubert = torch.mean(pooled_output_hubert, dim=1)
        # print("pooled hubert", pooled_hubert.size())

        # Concat the two models
        concat = torch.cat((pooled_output_bert, pooled_hubert), dim=1)
        concat_norm = self.layernorm1(concat)
        dropout1 = self.dropout(concat_norm)
        # print("concat size", dropout1.size())

        linear1 = self.linear1(dropout1)
        relu1 = self.relu(linear1)
        dropout2 = self.dropout(relu1)
        dropout2_norm = self.layernorm2(dropout2)
        # print("dropout2 norm", dropout2_norm.size())
        linear2 = self.linear2(dropout2_norm)
        # print("linear 2 size", linear2.size())
        tanh = self.tanh(linear2)
        # Scale to match input
        prediction = tanh * 2.5
        # print("prediction size", prediction.size())

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
        # print("bert input size", input_id.size())
        sequence_output_bert, pooled_output_bert = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )

        ## Hubert transform
        # print("hubert input size", input_values.size())
        output_tuple_hubert = self.hubert(input_values=input_values, return_dict=False)
        (pooled_output_hubert,) = output_tuple_hubert
        # print("pooled hubert", pooled_output_hubert.size())
        pooled_hubert = torch.mean(pooled_output_hubert, dim=1)
        # Set audio representation to zero
        pooled_hubert_zeroed = torch.zeros(pooled_hubert.size()).cuda()
        # print("pooled hubert", pooled_hubert.size())

        # Concat the two models
        concat = torch.cat((pooled_output_bert, pooled_hubert_zeroed), dim=1)
        concat_norm = self.layernorm1(concat)
        dropout1 = self.dropout(concat_norm)
        # print("concat size", dropout1.size())

        linear1 = self.linear1(dropout1)
        relu1 = self.relu(linear1)
        dropout2 = self.dropout(relu1)
        dropout2_norm = self.layernorm2(dropout2)
        # print("dropout2 norm", dropout2_norm.size())
        linear2 = self.linear2(dropout2_norm)
        # print("linear 2 size", linear2.size())
        tanh = self.tanh(linear2)
        # Scale to match input
        prediction = tanh * 2.5
        # print("prediction size", prediction.size())

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
        # print("bert input size", input_id.size())
        sequence_output_bert, pooled_output_bert = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        # Set text representation to zero
        pooled_bert_zeroed = torch.zeros(pooled_output_bert.size()).cuda()

        ## Hubert transform
        # print("hubert input size", input_values.size())
        output_tuple_hubert = self.hubert(input_values=input_values, return_dict=False)
        (pooled_output_hubert,) = output_tuple_hubert
        # print("pooled hubert", pooled_output_hubert.size())
        pooled_hubert = torch.mean(pooled_output_hubert, dim=1)
        # print("pooled hubert", pooled_hubert.size())

        # Concat the two models
        concat = torch.cat((pooled_bert_zeroed, pooled_hubert), dim=1)
        concat_norm = self.layernorm1(concat)
        dropout1 = self.dropout(concat_norm)
        # print("concat size", dropout1.size())

        linear1 = self.linear1(dropout1)
        relu1 = self.relu(linear1)
        dropout2 = self.dropout(relu1)
        dropout2_norm = self.layernorm2(dropout2)
        # print("dropout2 norm", dropout2_norm.size())
        linear2 = self.linear2(dropout2_norm)
        # print("linear 2 size", linear2.size())
        tanh = self.tanh(linear2)
        # Scale to match input
        prediction = tanh * 2.5
        # print("prediction size", prediction.size())

        return prediction
