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


class CustomBERTModel(nn.Module):
    def __init__(self):
        super(CustomBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(256 * 2, 5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_id, mask):
        sequence_output, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )

        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        ## extract the 1st token's embeddings
        lstm_output, (h, c) = self.lstm(sequence_output)
        hidden = torch.cat((lstm_output[:, -1, :256], lstm_output[:, 0, 256:]), dim=-1)
        ### assuming only using the output of the last LSTM cell to perform classification
        linear_output = self.linear(hidden.view(-1, 256 * 2))
        relu = self.relu(linear_output)
        predictions = self.softmax(relu)

        return predictions


class CustomHUBERTModel(nn.Module):
    def __init__(self):
        super(CustomHUBERTModel, self).__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(256 * 2, 5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_values):
        output_tuple = self.hubert(input_values=input_values, return_dict=False)
        (pooled_output,) = output_tuple

        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        ## extract the 1st token's embeddings
        lstm_output, (h, c) = self.lstm(pooled_output)
        hidden = torch.cat((lstm_output[:, -1, :256], lstm_output[:, 0, 256:]), dim=-1)
        ### assuming only using the output of the last LSTM cell to perform classification
        linear_output = self.linear(hidden.view(-1, 256 * 2))
        relu = self.relu(linear_output)
        predictions = self.softmax(relu)

        return predictions



