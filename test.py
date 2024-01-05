import argparse
import torchvision.transforms.functional as F
import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader
from torchvision import transforms, models
from YODADataset import YODADataset
import warnings
from tqdm import tqdm
from YODAModel import YODAModel
from KittiDataset import KittiDataset, batch_ROIs

warnings.filterwarnings("ignore")

# Define data transforms for testing
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.3656, 0.3844, 0.3725], [0.4194, 0.4075, 0.4239])
])

car_count = 0
nocar_count = 0
max_examples_per_class = 10000


def test_yoda_model(model, dataset, device):
    # Iterating through batches
    total_iou_scores = []  # To store IoU scores of all 'Car' ROIs
    display_count = 0  # Counter to keep track of the number of images displayed

    for i, (image, label) in enumerate(dataset):
        idx = dataset.class_label['Car']
        car_ROIs = dataset.strip_ROIs(class_ID=idx, label_list=label)

        anchor_centers = model.anchors.calc_anchor_centers(image.shape, model.anchors.grid)
        ROIs, boxes = model.anchors.get_anchor_ROIs(image, anchor_centers, model.anchors.shapes)
        predictions = []
        for roi in ROIs:
            # Apply preprocessing
            pil_image = Image.fromarray(roi)

            # Apply any necessary preprocessing (resize, normalize, etc.)
            pil_image = transform(pil_image).unsqueeze(0)

            # Move to device if using GPU
            pil_image = pil_image.to(device)

            # Forward pass
            with torch.no_grad():
                output = model(pil_image)
                pred = torch.sigmoid(output)
                pred = torch.round(pred)
                predictions.append(pred)

        for k, pred in enumerate(predictions):
            is_car = pred
            # Assuming binary classification: [0, 1] where 1 is 'Car'
            if is_car:
                iou_score = model.anchors.calc_max_IoU(boxes[k], car_ROIs)
                total_iou_scores.append(iou_score)
                box = boxes[k]
                pt1 = (int(box[0][1]), int(box[0][0]))
                pt2 = (int(box[1][1]), int(box[1][0]))
                cv2.rectangle(image, pt1, pt2, color=(0, 255, 0), thickness=2)

        # Display the image with bounding boxes for the first two images only
        if display_count < 5:
            cv2.imshow(f'Image {i}', image)
            key = cv2.waitKey(0)  # Wait for a key press to move to the next image
            if key == ord('q'):  # Press 'q' to quit the display loop early
                break
            display_count += 1

        print(f"Processed image {i}")

    mean_iou = sum(total_iou_scores) / len(total_iou_scores) if total_iou_scores else 0
    print(f"Mean IoU for 'Car' ROIs: {mean_iou}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classification")
    parser.add_argument('-b', type=int, default=1,
                        help='batch size')
    parser.add_argument('-model_select', type=int, default=0,
                        help='choose 0 for classifier and 1 for YODA')
    parser.add_argument('-l', type=str, default='decoder.pth',
                        help='load decoder')
    args = parser.parse_args()

    mem_pin = False
    if torch.cuda.is_available():
        device = torch.device('cuda')
        mem_pin = True
    else:
        device = torch.device('cpu')
    print('Found device ', device)

    if args.model_select == 0:
        test_dataset = YODADataset('data/Kitti8_ROIs', training=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=False, num_workers=2, pin_memory=mem_pin)

        # Load the trained model
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(512, 1)
        model.load_state_dict(torch.load(args.l))

        model = model.to(device)
        model.eval()
        # Testing loop
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = torch.sigmoid(output)
                pred = torch.round(pred)
                y_true.append(target.item())
                y_pred.append(pred.item())

        print('Accuracy: ', accuracy_score(y_true, y_pred))
        print('F1 score: ', f1_score(y_true, y_pred))
        print('Confusion matrix: ')
        print(confusion_matrix(y_true, y_pred))
        print('Classification report: ')
        print(classification_report(y_true, y_pred))

    elif args.model_select == 1:
        yoda_classifier = models.resnet18(pretrained=False)
        yoda_classifier.fc = torch.nn.Linear(512, 1)
        yoda_classifier.load_state_dict(torch.load(args.l))

        IoU_threshold = 0.02

        dataset = KittiDataset('data/Kitti8', training=False)

        model = YODAModel(yoda_classifier)
        model = model.to(device)
        test_yoda_model(model, dataset, device)
