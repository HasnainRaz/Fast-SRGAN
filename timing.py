from time import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch

from model import Generator


def benchmark(args):
    model = Generator(args['num_filters'], args['num_blocks'])
    model.cuda()
    model.eval()

    duration, count = 0, 0
    for i in range(50):
        start = time()
        _ = model(torch.rand((1, 3, 256, 256)).cuda())
        if i < 10:
            continue
        duration += time() - start
        count += 1

    avg_duration = duration / count

    print("Avg Time Taken: ", avg_duration)


if __name__ == '__main__':
    args = {'num_filters': 32, 'num_blocks': 8}
    benchmark(args)
