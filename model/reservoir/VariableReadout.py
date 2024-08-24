from torch import nn
import torch


class VariableReadout(nn.Module):
    def __init__(self, sequential_model):
        super(VariableReadout, self).__init__()
        self.sequential_model = sequential_model

    def forward(self, x):
        y = self.sequential_model(x)

        return y

    def predict(self, x):
        return self.forward(x)

    def fit(self, x, y, x_val=None, y_val=None, class_weights=None, epochs=100, batch_size=100):
        # Make sure the data is on the GPU
        x = x.to('cuda')
        y = y.to('cuda')

        if x_val is not None and y_val is not None:
            x_val = x_val.to('cuda')
            y_val = y_val.to('cuda')

        # Define the loss function
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Define the optimiser
        optimiser = torch.optim.AdamW(self.parameters(), lr=0.001)

        # Define the number of batches
        n_batches = x.shape[0] // batch_size

        # Define an early stopping criterion
        early_stopping = False
        patience = 10
        counter = 0
        best_loss = float('inf')

        losses = []
        val_losses = []

        with torch.cuda.device('cuda'):
            for epoch in range(epochs):
                for i in range(n_batches):
                    # Get the batch
                    x_batch = x[i * batch_size:(i + 1) * batch_size]
                    y_batch = y[i * batch_size:(i + 1) * batch_size]

                    # print(f"Device of x_batch: {x_batch.device}")
                    # Zero the gradients
                    optimiser.zero_grad()

                    # Forward pass
                    y_pred = self.forward(x_batch)

                    # Compute the loss
                    loss = criterion(y_pred, y_batch)

                    # Backward pass
                    loss.backward()

                    # Optimise
                    optimiser.step()

                    # Append the loss
                    losses.append(loss.item())

                # Compute the validation loss
                with torch.no_grad():
                    if x_val is not None and y_val is not None:
                        y_val_pred = self.forward(x_val)
                        val_loss = criterion(y_val_pred, y_val)
                    else:
                        y_pred = self.forward(x)
                        val_loss = criterion(y_pred, y)

                    val_losses.append(val_loss.item())

                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

                    if val_loss < best_loss:
                        best_loss = val_loss
                        counter = 0
                    else:
                        counter += 1

                if counter >= patience:
                    print(f"Early stopping after {epoch} epochs as loss has not improved in the last {patience} epochs.")
                    early_stopping = True
                    break

                if early_stopping:
                    break

        return losses, val_losses
