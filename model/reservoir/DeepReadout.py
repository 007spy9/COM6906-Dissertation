import torch
import torch.nn as nn

class DeepReadout(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepReadout, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Our readout will be a dense layer, which then feeds into a softmax layer
        self.fc1 = nn.Linear(input_size, hidden_size, device='cuda')
        self.fc2 = nn.Linear(hidden_size, output_size, device='cuda')

    def forward(self, x):
        # Our readout will be a dense layer, which then feeds into a softmax layer
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)

        x = torch.softmax(x, dim=1)

        return x

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

        with torch.cuda.device('cuda'):
            for epoch in range(epochs):
                for i in range(n_batches):
                    # Get the batch
                    x_batch = x[i*batch_size:(i+1)*batch_size]
                    y_batch = y[i*batch_size:(i+1)*batch_size]

                    #print(f"Device of x_batch: {x_batch.device}")

                    # Zero the gradients
                    optimiser.zero_grad()

                    # Forward pass
                    outputs = self.forward(x_batch)

                    # Calculate the loss
                    loss = criterion(outputs, torch.argmax(y_batch, dim=1))

                    # Backward pass
                    loss.backward()

                    # Optimise
                    optimiser.step()

                with torch.no_grad():
                    # Calculate the loss
                    if x_val is None or y_val is None:
                        outputs = self.forward(x_val)
                        loss = criterion(outputs, torch.argmax(y_val, dim=1))
                    else:
                        outputs = self.forward(x)
                        loss = criterion(outputs, torch.argmax(y, dim=1))
                    losses.append(loss.item())

                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

                    if loss < best_loss:
                        best_loss = loss
                        counter = 0
                    else:
                        counter += 1

                if counter >= patience:
                    early_stopping = True
                    break

            if early_stopping:
                print("Early stopping as loss has not improved in the last 10 epochs.")

        return losses
