"""Models for training features, text and audio for confidence classification.

----------------------------------
Class BertClassifier: Model for text with Bert tokenized tensors classifier.
Class HubertClassifier: Model for audio with HuBert tokenized tensors classifier.
Class SelectFeaturesClassifier: Model for features as input tensors.
Class ResidualBlock: Residual block for ResNet.
Class CustomBERTModel: Model for text with Bert and Bi-LSTM.
Class CustomBERTSimpleModel: Model for text with Bert.
Class CustomHUBERTModel: Model for audio with Hubert and Bi-LSTM.
Class CustomHUBERTSimpleModel: Model for audio with Hubert.
Class CustomMultiModel: Model for text and audio with Bi-LSTM.
Class CustomMultiModel: Model for text and audio.
"""

import torch
from torch import nn
from transformers import BertModel, HubertModel


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(dropout)
        # Regression
        self.linear = nn.Linear(768, 1)
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
        self.linear = nn.Linear(768, 1)
        self.relu = nn.ReLU()

    def forward(self, input_values):
        output_tuple = self.hubert(input_values=input_values, return_dict=False)
        (pooled_output,) = output_tuple
        # print("pooled", pooled_output.size())
        output_reduced = torch.mean(pooled_output, dim=1)
        # print("reduced", output_reduced.size())
        dropout_output = self.dropout(output_reduced)
        # print("dropout", dropout_output.size())
        linear_output = self.linear(dropout_output)
        # print("linear", linear_output.size())
        final_layer = self.relu(linear_output)
        # print("final", final_layer.size())
        return final_layer


class SelectFeaturesClassifier(nn.Module):
    def __init__(self, num_rows, num_columns):
        super().__init__()
        # store the different layers
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(num_rows * num_columns, 256)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(256, 1)
        self.softmax = nn.Softmax(dim=1)

    # in what sequence do we input the data
    def forward(self, input_data):
        flattened_data = self.flatten(input_data)
        # print("flattened size", flattened_data.size())
        linear1 = self.linear1(flattened_data)
        # print("linear1 size", linear1.size())
        relu = self.relu(linear1)
        # print("relu size", relu.size())
        linear2 = self.linear2(relu)
        # print("linear2 size", linear2.size())
        predictions = self.softmax(linear2)
        return predictions


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.activate = nn.LeakyReLU()
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class CustomBERTModel(nn.Module):
    def __init__(self, dropout=0.5):
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(256 * 2, 32)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(32, 1)
        self.tanh = nn.Tanh()

    def forward(self, input_id, mask):
        self.lstm.flatten_parameters()
        sequence_output, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )

        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        ## extract the 1st token's embeddings
        lstm_output, (h, c) = self.lstm(sequence_output)
        hidden = torch.cat((lstm_output[:, -1, :256], lstm_output[:, 0, 256:]), dim=-1)
        ### assuming only using the output of the last LSTM cell to perform classification
        linear1 = self.linear1(hidden.view(-1, 256 * 2))
        dropout = self.dropout(linear1)
        linear2 = self.linear2(dropout)
        tanh = self.tanh(linear2)
        # Scale to match input
        prediction = tanh * 2.5

        return prediction


class CustomBERTSimpleModel(nn.Module):
    def __init__(self, dropout=0.5):
        super(CustomBERTSimpleModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 32)
        self.linear2 = nn.Linear(32, 1)
        self.tanh = nn.Tanh()

    def forward(self, input_id, mask):
        sequence_output, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        dropout = self.dropout(sequence_output)
        linear1 = self.linear1(dropout)
        dropout2 = self.dropout(linear1)
        linear2 = self.linear2(dropout2)
        output_reduced = torch.mean(linear2, dim=1)
        tanh = self.tanh(output_reduced)
        # Scale to match input
        prediction = tanh * 2.5

        return prediction


class CustomHUBERTModel(nn.Module):
    def __init__(self, dropout=0.5):
        super(CustomHUBERTModel, self).__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(256 * 2, 32)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(32, 1)
        self.tanh = nn.Tanh()

    def forward(self, input_values):
        self.lstm.flatten_parameters()
        output_tuple = self.hubert(input_values=input_values, return_dict=False)
        (pooled_output,) = output_tuple

        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        ## extract the 1st token's embeddings
        lstm_output, (h, c) = self.lstm(pooled_output)
        hidden = torch.cat((lstm_output[:, -1, :256], lstm_output[:, 0, 256:]), dim=-1)
        # print("hidden size", hidden.size())
        ### assuming only using the output of the last LSTM cell to perform classification
        linear1 = self.linear1(hidden.view(-1, 256 * 2))
        dropout = self.dropout(linear1)
        linear2 = self.linear2(dropout)
        tanh = self.tanh(linear2)
        # Scale to match input
        prediction = tanh * 2.5

        return prediction


class CustomHUBERTSimpleModel(nn.Module):
    def __init__(self, dropout=0.5):
        super(CustomHUBERTSimpleModel, self).__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.linear1 = nn.Linear(768, 32)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(32, 1)
        self.tanh = nn.Tanh()

    def forward(self, input_values):
        output_tuple = self.hubert(input_values=input_values, return_dict=False)
        (pooled_output,) = output_tuple

        dropout = self.dropout(pooled_output)
        # print("pooled size", pooled_output.size())
        linear1 = self.linear1(dropout)
        # print("linear 1", linear1.size())
        dropout2 = self.dropout(linear1)
        linear2 = self.linear2(dropout2)
        # print("linear 2", linear2.size())
        output_reduced = torch.mean(linear2, dim=1)
        # print("reduced", output_reduced.size())
        tanh = self.tanh(output_reduced)
        # Scale to match input
        prediction = tanh * 2.5

        return prediction


class CustomMultiModel(nn.Module):
    def __init__(self, dropout=0.5):
        super(CustomMultiModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.layernorm1 = nn.LayerNorm([1, 32])
        self.linear1 = nn.Linear(256 * 2, 32)
        self.layernorm2 = nn.LayerNorm([1, 64])
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(64, 1)
        self.tanh = nn.Tanh()

    def forward(self, input_values, input_id, mask):
        self.lstm.flatten_parameters()
        ## Bert transform
        # print("bert input size", input_id.size())
        sequence_output_bert, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        # print("sequence bert", sequence_output_bert.size())

        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        # extract the 1st token's embeddings
        lstm_output_bert, (h, c) = self.lstm(sequence_output_bert)
        hidden_bert = torch.cat(
            (lstm_output_bert[:, -1, :256], lstm_output_bert[:, 0, 256:]), dim=-1
        )
        # print("lstm bert", hidden_bert.size())
        ### assuming only using the output of the last LSTM cell to perform classification
        linear1_bert = self.linear1(hidden_bert.view(-1, 256 * 2))
        # print("linear 1 bert", linear1_bert.size())
        linear1_bert_norm = self.layernorm1(linear1_bert)
        dropout1_bert = self.dropout(linear1_bert_norm)
        # print("dropout 1 bert size:", dropout1_bert.size())

        ## Hubert transform
        # print("hubert input size", input_values.size())
        output_tuple_hubert = self.hubert(input_values=input_values, return_dict=False)
        (pooled_output_hubert,) = output_tuple_hubert
        # print("sequence hubert", pooled_output_hubert.size())
        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        ## extract the 1st token's embeddings
        lstm_output_hubert, (h, c) = self.lstm(pooled_output_hubert)
        hidden_hubert = torch.cat(
            (lstm_output_hubert[:, -1, :256], lstm_output_hubert[:, 0, 256:]), dim=-1
        )
        # print("lstm hubert", hidden_hubert.size())
        ### assuming only using the output of the last LSTM cell to perform classification
        linear1_hubert = self.linear1(hidden_hubert.view(-1, 256 * 2))
        # print("linear 1 hubert", linear1_hubert.size())
        linear1_hubert_norm = self.layernorm1(linear1_hubert)
        dropout1_hubert = self.dropout(linear1_hubert_norm)
        # print("dropout 1 hubert size:", dropout1_hubert.size())

        # Concat the two models
        concat = torch.cat((dropout1_bert, dropout1_hubert), dim=1)
        # print("concat size", concat.size())
        concat_norm = self.layernorm2(concat)
        dropout2 = self.dropout(concat_norm)
        # print("concat size", concat.size())
        linear2 = self.linear2(dropout2)
        # print("linear 2 size", linear2.size())
        tanh = self.tanh(linear2)
        # Scale to match input
        prediction = tanh * 2.5
        # print("prediction size", prediction.size())

        return prediction


class CustomMultiModelSimple(nn.Module):
    def __init__(self, dropout=0.5):
        super(CustomMultiModelSimple, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.layernorm1 = nn.LayerNorm([8, 1136, 768])
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 32)
        self.layernorm2 = nn.LayerNorm([8, 32])
        self.linear2 = nn.Linear(32, 1)
        self.tanh = nn.Tanh()

    def forward(self, input_values, input_id, mask):
        ## Bert transform
        # print("bert input size", input_id.size())
        sequence_output_bert, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        # print("sequence bert", sequence_output_bert.size())

        ## Hubert transform
        # print("hubert input size", input_values.size())
        output_tuple_hubert = self.hubert(input_values=input_values, return_dict=False)
        (pooled_output_hubert,) = output_tuple_hubert
        # print("sequence hubert", pooled_output_hubert.size())

        # Concat the two models
        concat = torch.cat((sequence_output_bert, pooled_output_hubert), dim=1)
        # print("concat size", concat.size())
        concat_norm = self.layernorm1(concat)
        dropout1 = self.dropout(concat_norm)
        # print("concat size", dropout2.size())

        # Reduce dimension
        output_reduced = torch.mean(dropout1, dim=1)
        # print("reduced size", output_reduced.size())
        linear1 = self.linear1(output_reduced)
        dropout2 = self.dropout(linear1)
        dropout2_norm = self.layernorm2(dropout2)
        # print("dropout2 norm", dropout2_norm.size())
        linear2 = self.linear2(dropout2_norm)
        # print("linear 2 size", linear2.size())
        tanh = self.tanh(linear2)
        # Scale to match input
        prediction = tanh * 2.5
        # print("prediction size", prediction.size())

        return prediction


class CustomMultiModelSimplePooled(nn.Module):
    def __init__(self, dropout=0.5):
        super(CustomMultiModelSimplePooled, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.layernorm1 = nn.LayerNorm([8, 768 * 2])
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768 * 2, 32)
        self.layernorm2 = nn.LayerNorm([1, 32])
        self.linear2 = nn.Linear(32, 1)
        self.tanh = nn.Tanh()

    def forward(self, input_values, input_id, mask):
        ## Bert transform
        # print("bert input size", input_id.size())
        sequence_output_bert, pooled_output_bert = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        # print("pooled bert", pooled_output_bert.size())

        ## Hubert transform
        # print("hubert input size", input_values.size())
        output_tuple_hubert = self.hubert(input_values=input_values, return_dict=False)
        (pooled_output_hubert,) = output_tuple_hubert
        # print("pooled hubert", pooled_output_hubert.size())
        pooled_hubert = torch.mean(pooled_output_hubert, dim=1)
        # print("pooled hubert", pooled_hubert.size())

        # Concat the two models
        concat = torch.cat((pooled_output_bert, pooled_hubert), dim=1)
        # print("concat size", concat.size())
        concat_norm = self.layernorm1(concat)
        dropout1 = self.dropout(concat_norm)
        # print("concat size", dropout1.size())

        linear1 = self.linear1(dropout1)
        dropout2 = self.dropout(linear1)
        dropout2_norm = self.layernorm2(dropout2)
        # print("dropout2 norm", dropout2_norm.size())
        linear2 = self.linear2(dropout2_norm)
        # print("linear 2 size", linear2.size())
        tanh = self.tanh(linear2)
        # Scale to match input
        prediction = tanh * 2.5
        # print("prediction size", prediction.size())

        return prediction
