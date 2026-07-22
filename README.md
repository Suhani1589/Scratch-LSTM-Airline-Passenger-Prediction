# Scratch LSTM for Airline Passenger Prediction

## Overview

This project implements an **LSTM (Long Short-Term Memory)** network completely from scratch using **PyTorch tensors and autograd**, without using `torch.nn.LSTM` or `torch.nn.LSTMCell`.

The model is trained on the Airline Passenger dataset to predict future passenger counts.

---

## Features

- Manual Forget Gate
- Manual Input Gate
- Manual Candidate Cell State
- Manual Cell State Update
- Manual Output Gate
- Manual Hidden State
- Backpropagation using PyTorch Autograd
- Airline Passenger Forecasting

---

## LSTM Equations

Forget Gate

fₜ = σ(Wf xₜ + Uf hₜ₋₁ + bf)

Input Gate

iₜ = σ(Wi xₜ + Ui hₜ₋₁ + bi)

Candidate Cell State

c̃ₜ = tanh(Wc xₜ + Uc hₜ₋₁ + bc)

Cell State

cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ

Output Gate

oₜ = σ(Wo xₜ + Uo hₜ₋₁ + bo)

Hidden State

hₜ = oₜ ⊙ tanh(cₜ)

---

## Technologies

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Results

The manually implemented LSTM successfully learns the trend and seasonality of the Airline Passenger dataset.
