import tensorflow as tf
import numpy as np

# Define the model
model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Training data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0], dtype=float)
model.fit(xs, ys, epochs=500, verbose=0)

# Predict with NumPy array input
input_data = np.array([10.0])
print(model.predict(input_data))
# Output: ~[[18.999998]]
print(model.predict(np.array([20.0])))
# Output: ~[[38.999996]]
print(model.predict(np.array([0.0])))