import torch
import time

def test_item_bottleneck():
    x = torch.tensor([1.0])
    iterations = 100000
    
    start = time.time()
    for i in range(iterations):
        val = x.item()
    end = time.time()
    
    print(f'Time for {iterations} item() calls: {(end - start) * 1000:.2f} ms')
    print(f'Average time per call: {(end - start) * 1000 / iterations:.4f} ms')

if __name__ == "__main__":
    test_item_bottleneck()