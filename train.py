import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from YODADataset import YODADataset
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.3656, 0.3844, 0.3725], [0.4194, 0.4075, 0.4239])
])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classification")
    parser.add_argument('-e', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('-lr', type=int, default=1e-3,
                        help='learning rate')
    parser.add_argument('-b', type=int, default=10,
                        help='batch size')
    parser.add_argument('-model_select', type=int, default=0,
                        help='choose 0 for classifier and 1 for YODA')
    parser.add_argument('-s', type=str, default='decoder.pth',
                        help='save decoder')
    parser.add_argument('-p', type=str, default='decoder.png',
                        help='Value for p')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device being used: {device}")

    if args.model_select == 0:
        train_dataset = YODADataset('data/Kitti8_ROIs', training=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.b, shuffle=False, num_workers=2)
        test_dataset = YODADataset('data/Kitti8_ROIs', training=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.b, shuffle=False, num_workers=2)

        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512, 1)

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.7)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)

        model = model.to(device)

        train_losses = []
        val_losses = []
        best_test_acc = 10000
        for epoch in range(args.e):
            print('Epoch: ', epoch)
            start = time.time()
            train_loss = 0.0
            test_loss = 0.0

            model.train()
            for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training: ',
                                                  unit='batch', leave=False):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target.unsqueeze(1).float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)

            train_loss = train_loss / len(train_loader.dataset)
            train_losses.append(train_loss)
            print('Train Loss: ', train_loss)

            model.eval()
            with torch.no_grad():
                for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing: ',
                                                      unit='batch', leave=False):
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target.unsqueeze(1).float())
                    test_loss += loss.item() * data.size(0)

            end = time.time()

            test_loss = test_loss / len(test_loader.dataset)
            val_losses.append(test_loss)
            print('Test Loss: ', test_loss)
            scheduler.step()
            print('Time: ', end - start, 's')

            if test_loss < best_test_acc:
                best_test_acc = test_loss
                torch.save(model.state_dict(), args.s)
                print('Saved model')
        print("Training Complete")

        import matplotlib.pyplot as plt

        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(args.p)
        plt.show()

