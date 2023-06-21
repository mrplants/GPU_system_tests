import unittest
import torch
import tensorflow as tf
import torchvision
import sys

class GPUTorchTest(unittest.TestCase):

    def util_test_nan(self, gpu_id):
        device = torch.device(f'cuda:{gpu_id}')

        torch.manual_seed(0)  # fix the random seed
        torch.cuda.empty_cache()  # clear the GPU memory

        # Create some random tensors
        a = torch.randn(1000, 1000).to(device)
        b = torch.randn(1000, 1000).to(device)

        # Perform computations
        c = a * b
        d = torch.exp(c)

        # check if there is any NaN in the numpy array
        has_nan = torch.isnan(d).any().item()
        self.assertFalse(has_nan, "NaN values found in the calculations on GPU")

    def util_test_tensor_operation(self, gpu_id):
        device = torch.device(f'cuda:{gpu_id}')
        a = torch.tensor([1.0, 2.0, 3.0], device=device)
        b = torch.tensor([4.0, 5.0, 6.0], device=device)
        c = a + b
        self.assertTrue(isinstance(c, torch.Tensor))
        self.assertEqual(c.cpu().numpy().tolist(), [5.0, 7.0, 9.0])

    def util_test_gradient(self, gpu_id):
        device = torch.device(f'cuda:{gpu_id}')
        x = torch.tensor([2.0, 3.0, 4.0], device=device, requires_grad=True)
        y = x ** 2
        y.sum().backward()
        expected_gradient = torch.tensor([4.0, 6.0, 8.0], device=device)
        self.assertTrue(isinstance(x.grad, torch.Tensor))
        self.assertTrue(torch.allclose(expected_gradient, x.grad, atol=1e-6))

    def util_test_model_fit(self, gpu_id):
        device = torch.device(f'cuda:{gpu_id}')
        model = torch.nn.Sequential(
            torch.nn.Linear(784, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10),
            torch.nn.Softmax(dim=1)
        ).to(device)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        for epoch in range(1):  # loop over the dataset once
            for inputs, labels in trainloader:
                inputs, labels = inputs.view(-1, 784).to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def setUp(self):
        # Set up any common resources or test dependencies
        self.gpus = torch.cuda.device_count()

    def tearDown(self):
        # Clean up any resources after each test method
        pass

    def test_torch_cuda_available(self):
        self.assertTrue(torch.cuda.is_available())

    def test_torch_tensor_operation_on_gpu_0(self):
        self.util_test_tensor_operation(0)
    def test_torch_tensor_operation_on_gpu_1(self):
        self.util_test_tensor_operation(1)

    def test_torch_gradient_on_gpu_0(self):
        self.util_test_gradient(0)
    def test_torch_gradient_on_gpu_1(self):
        self.util_test_gradient(1)

    def test_torch_model_training_on_gpu_0(self):
        self.util_test_model_fit(0)
    def test_torch_model_training_on_gpu_1(self):
        self.util_test_model_fit(1)

    def test_torch_nan_on_gpu_0(self):
        self.util_test_nan(0)
    def test_torch_nan_on_gpu_1(self):
        self.util_test_nan(1)

if __name__ == '__main__':
    unittest.main()
