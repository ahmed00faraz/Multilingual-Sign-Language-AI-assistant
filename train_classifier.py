import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = './data.pickle'  # Use a variable for the data path

with open(DATA_PATH, 'rb') as f:
    data_dict = pickle.load(f)

data = data_dict['data']
labels = data_dict['labels']

# No need to convert to NumPy arrays, they are already arrays

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42) # Add random_state for reproducibility

model = RandomForestClassifier(n_estimators=100, random_state=42) # Increase n_estimators, add random_state

model.fit(x_train, y_train)

y_pred = model.predict(x_test) # Use y_pred for consistency

score = accuracy_score(y_pred, y_test)

print(f'{score * 100:.2f}% of samples were classified correctly!') # Use f-string for formatting

MODEL_PATH = './model.p' # Use a variable for the model path
with open(MODEL_PATH, 'wb') as f:
    pickle.dump({'model': model}, f)