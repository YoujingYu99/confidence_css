import torch
from torch import nn
from transformers import BertModel, HubertModel


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(dropout)
        # 5 categories
        self.linear = nn.Linear(768, 5)
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
        # 5 categories
        self.linear = nn.Linear(768, 5)
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
        self.linear2 = nn.Linear(256, 5)
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


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim1,
        hidden_dim2,
        output_dim,
        n_layers,
        bidirectional,
        dropout,
        pad_index,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim1,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_dim1 * 2, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(embedded, text_lengths, batch_first=True)

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        rel = self.relu(cat)
        dense1 = self.fc1(rel)
        drop = self.dropout(dense1)
        preds = self.fc2(drop)
        return preds


# class CNNNetwork(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         # 4 conv blocks / flatten / linear / softmax
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=1,
#                 out_channels=16,
#                 kernel_size=3,
#                 stride=1,
#                 padding=2
#             ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=16,
#                 out_channels=32,
#                 kernel_size=3,
#                 stride=1,
#                 padding=2
#             ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=32,
#                 out_channels=64,
#                 kernel_size=3,
#                 stride=1,
#                 padding=2
#             ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=64,
#                 out_channels=128,
#                 kernel_size=3,
#                 stride=1,
#                 padding=2
#             ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.flatten = nn.Flatten()
#         self.linear = nn.Linear(332, 5)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, input_data):
#         print("input size", input_data.size())
#         x = self.conv1(input_data)
#         print("conv1", x.size())
#         x = self.conv2(x)
#         print("conv2", x.size())
#         x = self.conv3(x)
#         print("conv3", x.size())
#         x = self.conv4(x)
#         print("conv4", x.size())
#         x = self.flatten(x)
#         print("flatten", x.size())
#         logits = self.linear(x)
#         print("logits", logits.size())
#         predictions = self.softmax(logits)
#         print("predict", predictions.size())
#         return predictions
