import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from os import system
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import datetime

system('clear')

# --------------------------- VISUALIZATION CATEGORY ---------------------------
# Visualize the history data to improve model
def history_visualizer(history):
    pd.DataFrame(history.history).plot()
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.show()

# Visualize the ideal learning rate
def learning_rate_visualizer(learning_rate, epoch, history):
    lrs = learning_rate * (10 ** (tf.range(epoch)/20))

    plt.figure(figsize = (10, 7))
    plt.semilogx(lrs, history.history['loss'])
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate vs Loss')
    plt.show() 

# --------------------------- ML CATEGORY ---------------------------
# Acquire data to process
medical_data = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv')

# Preprocess data (MinMax (To get values btwn 1 and 0) and Onehot (To turn yes/no to 1 or 0))
ct = make_column_transformer(
    (MinMaxScaler(), ['age', 'bmi', 'children']),
    (OneHotEncoder(handle_unknown='ignore'), ['sex', 'smoker', 'region'])
)

# Create feature (X) and label (y)
X = medical_data.drop('charges', axis = 1)
y = medical_data['charges']

# Get training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the column transformer to our training data before transforming the data itself
ct.fit(X_train)

# Transform training and test data w/ normalization and OneHotEncoder
X_train_normalized = ct.transform(X_train)
X_test_normalized = ct.transform(X_test)

# Set seed for reproducibility
tf.random.set_seed(42)

# Create Compile and Fit Model
EPOCHS = 100
LEARNING_RATE = 0.01


model = tf.keras.Sequential([ 
    tf.keras.layers.Dense(11, activation = 'relu'),
    tf.keras.layers.Dense(1_000),
    tf.keras.layers.Dense(1, activation = 'linear')
])

model.compile(
    loss = 'mae',
    optimizer = tf.keras.optimizers.Adam(lr = LEARNING_RATE),
    metrics = ['mae']
)

#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: LEARNING_RATE * 10**(epoch/20))

#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(
    X_train_normalized, 
    y_train, 
    epochs = EPOCHS, 
    verbose = 0
    #callbacks = [lr_scheduler],
)

# Evaluate the model
print('\nEVALUATION\n')
model.evaluate(X_test_normalized, y_test)

# Prediction
y_pred = model.predict(X_test_normalized)

history_visualizer(history)
#learning_rate_visualizer(LEARNING_RATE, EPOCHS, history)

# To view in tensorboard, write in command:
# tensorboard --logdir logs/fit