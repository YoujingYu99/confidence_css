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


class AllFeaturesClassifier(nn.Module):
    def __init__(self, num_rows):
        super().__init__()
        # store the different layers
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(num_rows * 39, 256)
        # self.linear1 = nn.Linear(38, 256)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(256, 5)
        self.softmax = nn.Softmax(dim=1)

    # in what sequence do we input the data
    def forward(self, input_data):
        # input_data = input_data[1]
        print("input type in model", type(input_data))
        if isinstance(input_data, list):
            print("This input is a list")
            print("list size", len(input_data))
            # print(input_data)
            input_data = input_data[1]
            print(input_data)
            print("size of this tensor", input_data.size())
        print("final input size", input_data.size())
        flattened_data = self.flatten(input_data)
        # print("flattened size", flattened_data.size())
        # logits = self.dense_layers(flattened_data)
        linear1 = self.linear1(flattened_data)
        # print("linear1 size", linear1.size())
        relu = self.relu(linear1)
        # print("relu size", relu.size())
        linear2 = self.linear2(relu)
        # print("linear2 size", linear2.size())
        predictions = self.softmax(linear2)
        return predictions


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
