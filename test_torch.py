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
        self.assertFalse(has_nan, "NaN values found in the calculations on GPU {gpu_id}.")

    def util_test_tensor_operation(self, gpu_id):
        device = torch.device(f'cuda:{gpu_id}')
        a = torch.tensor([1.0, 2.0, 3.0], device=device)
        b = torch.tensor([4.0, 5.0, 6.0], device=device)
        c = a + b
        self.assertTrue(isinstance(c, torch.Tensor), f'GPU {gpu_id} failed basic tensor operation test because c is not a tensor')
        self.assertEqual(c.cpu().numpy().tolist(), [5.0, 7.0, 9.0], f'GPU {gpu_id} failed basic tensor operation test because c is not correct')

    def util_test_gradient(self, gpu_id):
        device = torch.device(f'cuda:{gpu_id}')
        x = torch.tensor([2.0, 3.0, 4.0], device=device, requires_grad=True)
        y = x ** 2
        y.sum().backward()
        expected_gradient = torch.tensor([4.0, 6.0, 8.0], device=device)
        self.assertTrue(isinstance(x.grad, torch.Tensor), f'GPU {gpu_id} failed gradient test because x.grad is not a tensor')
        self.assertTrue(torch.allclose(expected_gradient, x.grad, atol=1e-6), f'GPU {gpu_id} failed gradient test because x.grad is not correct')

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
        loss_values = []
        acc_values = []
        for epoch in range(2):  # loop over the dataset more than once
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in trainloader:
                inputs, labels = inputs.view(-1, 784).to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            loss_values.append(running_loss / len(trainloader))
            acc_values.append(100. * correct / total)
        
        # Check that loss has decreased and accuracy has increased over epochs
        self.assertGreater(loss_values[0], loss_values[-1], f'GPU {gpu_id} failed to decrease loss during model training test')
        self.assertLess(acc_values[0], acc_values[-1], f'GPU {gpu_id} failed to increase accuracy during model training test')

    def setUp(self):
        # Set up any common resources or test dependencies
        self.gpus = torch.cuda.device_count()

    def tearDown(self):
        # Clean up any resources after each test method
        pass

    def test_torch_cuda_available(self):
        self.assertTrue(torch.cuda.is_available())

    def test_torch_tensor_operation(self):
        for GPU_id in range(self.gpus):
            self.util_test_tensor_operation(GPU_id)

    def test_torch_gradient(self):
        for GPU_id in range(self.gpus):
            self.util_test_gradient(GPU_id)

    def test_torch_model_training(self):
        for GPU_id in range(self.gpus):
            self.util_test_model_fit(GPU_id)

    def test_torch_nan(self):
        for GPU_id in range(self.gpus):
            self.util_test_nan(GPU_id)

if __name__ == '__main__':
    unittest.main()
