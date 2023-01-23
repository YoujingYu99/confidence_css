from models import *
from prettytable import PrettyTable


def count_parameters(model):
    """
    Print parameter names and layers and count the total number of
    parameters.
    :param model: The name of the deep learning model.
    :return: The total number of parameters as integer.
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param

    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


model = CustomMultiModelSimplePooled()

for layer in model.bert.encoder.layer[:11]:
    for param in layer.parameters():
        param.requires_grad = False

for layer in model.hubert.encoder.layers[:11]:
    for param in layer.parameters():
        param.requires_grad = False
print("new")
print(count_parameters(model))
