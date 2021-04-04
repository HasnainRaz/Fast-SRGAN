import torch
from model import Generator
import os
import cv2


@torch.no_grad()
def main(args):
    state_dict = torch.load('weights/epoch=999-step=66999-v1.ckpt')['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'generator' in k:
            new_state_dict[k.replace('generator.', '')] = v
    model = Generator(32, 6)
    model.eval()
    #model.load_state_dict(new_state_dict)

    image_paths = [os.path.join(args['image_dir'], x) for x in os.listdir(args['image_dir'])]

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        image = cv2.resize(image, (w // 4, h // 4), interpolation=cv2.INTER_LINEAR)
        image = (image / 127.5) - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        y = (model(image)[0].permute(1, 2, 0) + 1.0) / 2.0
        y = y.numpy()
        image = image[0].permute(1, 2, 0).numpy()
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
        image = (image + 1.0) / 2.0
        cv2.imwrite(f'{i}.png', y * 255)
        cv2.imwrite(f'{i}_lr.png', image * 255)


if __name__ == '__main__':
    args = {'image_dir': '/home/hasnain/Downloads/DIV2K'}
    main(args)
