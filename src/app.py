import tensorflow as tf
import numpy as np

model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0], dtype=float)
model.fit(xs, ys, epochs=500, verbose=0)

print(model.predict([10.0]))
# Output: [[18.999998]]
print(model.predict([20.0]))
# Output: [[38.999996]]
print(model.predict([0.0]))

