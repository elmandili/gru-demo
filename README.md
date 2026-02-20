# Reuters Text Classification (GRU)

This repository contains a Jupyter notebook (`train.ipynb`) that trains and evaluates a neural network model using TensorFlow/Keras.

## Notebook
- **File:** `train.ipynb`

## What the code does
- Loads data and creates train/test splits.
- Uses the Reuters dataset (Keras built-in) for multi-class text classification (46 classes).
- Converts text to integer sequences (tokenization + padding) and feeds them to an Embedding layer.

- Trains an RNN-family model (SimpleRNN / LSTM / GRU) and evaluates accuracy.
- Uses time-series windows (generator/dataset) for sequence forecasting.

## How to run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start Jupyter and open the notebook:
   ```bash
   jupyter notebook train.ipynb
   ```

## Saving the trained model
```python
model.save("model.keras")
```

Load it later:
```python
from tensorflow.keras.models import load_model
model = load_model("model.keras")
```

## Notes
- If labels are one-hot encoded (`to_categorical`), use `categorical_crossentropy`.
- If labels are integers, use `sparse_categorical_crossentropy`.

## License
Educational / research use.
