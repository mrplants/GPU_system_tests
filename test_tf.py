import unittest
import torch
import tensorflow as tf
import sys
import numpy as np

class GPUTensorflowTest(unittest.TestCase):

    def util_test_tensor_operation(self, gpu_id):
        with tf.device(f'/GPU:{gpu_id}'):
            a = tf.constant([1.0, 2.0, 3.0])
            b = tf.constant([4.0, 5.0, 6.0])
            c = a + b
            self.assertTrue(isinstance(c, tf.Tensor))
            self.assertEqual(c.numpy().tolist(), [5.0, 7.0, 9.0], f'GPU {gpu_id} failed basic tensor operation test')
            
    
    def util_test_gradient(self, gpu_id):
        def compute_gradient(x):
            with tf.GradientTape() as tape:
                y = tf.square(x)
            return tape.gradient(y, x)

        with tf.device(f'/GPU:{gpu_id}'):
            x = tf.Variable([2.0, 3.0, 4.0])
            gradient = compute_gradient(x)
            expected_gradient = tf.constant([4.0, 6.0, 8.0])
            self.assertTrue(isinstance(gradient, tf.Tensor), f'GPU {gpu_id} failed gradient test')
            tf.debugging.assert_near(expected_gradient, gradient, atol=1e-6)
    
    def util_test_model_fit(self, gpu_id):
        with tf.device(f'/GPU:{gpu_id}'):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(512, input_shape=(784,), activation='relu'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
            (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
            x_train = x_train.reshape((-1, 784)).astype('float32') / 255.0
            y_train = y_train.astype('int32')

            # Train model and capture training history
            history = model.fit(x_train, y_train, epochs=4, batch_size=32, verbose=0)

            # Check that loss has decreased and is not NaN
            loss_values = history.history['loss']
            self.assertGreater(loss_values[0], loss_values[-1], f'GPU {gpu_id} failed to decrease loss during model training test')
            self.assertFalse(np.isnan(loss_values[-1]).any(), f'GPU {gpu_id} loss is NaN during model training test')

            # Check that accuracy has increased
            acc_values = history.history['accuracy']
            self.assertLess(acc_values[0], acc_values[-1], f'GPU {gpu_id} failed to increase accuracy during model training test')

    def setUp(self):
        # Set up any common resources or test dependencies
        self.gpus = tf.config.list_physical_devices('GPU')

    def tearDown(self):
        # Clean up any resources after each test method
        pass

    def test_tensorflow_cuda_available(self):
        self.assertTrue(len(tf.config.list_physical_devices('GPU')) > 0)

    def test_tensorflow_tensor_operation(self):
        for GPU_index in range(len(self.gpus)):
            self.util_test_tensor_operation(GPU_index)

    def test_tensorflow_gradient(self):
        for GPU_index in range(len(self.gpus)):
            self.util_test_gradient(GPU_index)

    def test_tensorflow_model_training(self):
        for GPU_index in range(len(self.gpus)):
            self.util_test_model_fit(GPU_index)

if __name__ == '__main__':
    unittest.main()