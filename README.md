# LSTM from Scratch

Airline-passenger forecast with an LSTM cell written in PyTorch tensors and autograd. The notebook does not call `torch.nn.LSTM` or `torch.nn.LSTMCell`.

Data is the seaborn `flights` series, scaled with `MinMaxScaler` to [0, 1]. Each example is 3 past months, and the target is the next month. The cell has 16 units.

The cell computes a forget gate, an input gate, a candidate cell, the cell update, an output gate, and the hidden state. Training uses Adam at learning rate 0.01 and mean squared error for 200 epochs. The saved log goes from loss 0.009010 at epoch 0 to 0.002948 at epoch 180. Predictions are inverse-transformed for `prediction.png`.

## Run

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn
```

Open `scratch_lstm_airline_prediction.ipynb` and run it from the top. The optimizer cell must run before the 200-epoch loop.
