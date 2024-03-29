import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import argparse
import matplotlib.pyplot as plt
import numpy as np

from simple_model import simple_model
from FGSM import FGSM


def main(args):
    epsilons = np.arange(0, 1, 0.01)
    accs = np.zeros((100))

    batch_size = args.batch_size
    ckpt_path = args.ckpt_path
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Data loading code
    print("Loading test data")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    dataset_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    # Data loader
    print("Creating data loaders")
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, sampler=test_sampler,
                                                   num_workers=16)

    # model load
    print("Creating and loading model")
    model = simple_model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()

    for i in range(100):
        attacker = FGSM(model, criterion, epsilons[i])
        accs[i] = _accFGSM(attacker, data_loader_test, device, model)
        print(accs[i])
    plt.plot(epsilons, accs)
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.title("Accuracy according to epsilon")
    plt.savefig("repo_images/Acc_graph.png")
    plt.show()



def _accFGSM(attacker, dataLoader, device, model):
    correct = 0
    total = 0
    for data in dataLoader:
        images, labels = data
        images = images.float().to(device)
        labels = labels.to(device)
        # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
        images = images.view(-1, 28 * 28)

        attacked_images = attacker(images, labels)

        outputs = model(attacked_images)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(attacked_images)):
            if predicted[i] == labels[i]:
                correct += 1
            total += 1
        #print(total, ":", 100 * correct / total)
    return 100 * correct / total



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
    ap.add_argument('--ckpt_path', type=str, default='modelsave/clean_simple_model.pth', help='checkpoint file path')
    args = ap.parse_args()
    main(args)
