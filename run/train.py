import csv

import torch
from itertools import repeat

def train(model, train_loader, test_loader, epochs=500, model_name=None):
    model_path = '../saved_models/' + model_name + '.pth' if model_name is not None else None
    results_path = '../results/' + model_name + '.csv' if model_name is not None else None
    if results_path is not None:
        with open(results_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['predicted_value', 'ground_truth', 'epoch'])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    min_err = float('inf')
    for epoch in range(epochs):

        model.train()
        train_data, train_value = next(train_loader)
        optimizer.zero_grad()
        train_output = model(train_data)
        loss_value = criterion(train_output, train_value)
        loss_value.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_data, test_value = next(test_loader)
            test_output = model(test_data)
            error = torch.abs(test_output - test_value).mean()
            if error < min_err:
                min_err = error
                if model_path is not None:
                    torch.save(model.state_dict(), model_path)
            print(epoch,
                  'MADE: {:.4f}, '.format(error),
                  'Test Loss: {:.4f}, '.format(loss_value.item()),
                  'Min Error: {:.4f}'.format(min_err))

        if results_path is not None:
            with open(results_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(zip(test_output.tolist(), test_value.tolist(), repeat(epoch)))
