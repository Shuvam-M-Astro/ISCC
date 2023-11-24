import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms
import os

# Ensure everything runs on CPU
device = torch.device('cpu')

# Define the transform
transform = transforms.Compose([
    transforms.ToTensor(), 
    # Add other necessary transformations as per your requirement
])

# Paths for the dataset
data_dir = './data'
train_dir = os.path.join(data_dir, 'train2017')
val_dir = os.path.join(data_dir, 'val2017')
train_annotation = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
val_annotation = os.path.join(data_dir, 'annotations', 'instances_val2017.json')

# Load the COCO dataset
train_dataset = CocoDetection(root=train_dir, annFile=train_annotation, transform=transform, download=True)
test_dataset = CocoDetection(root=val_dir, annFile=val_annotation, transform=transform, download=True)

# Define the data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

# Load a pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)

# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f'Epoch {epoch}/{num_epochs}, Loss: {losses.item()}')

# Evaluate the model
model.eval()
with torch.no_grad():
    for images, targets in test_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        predictions = model(images)

        # Here, you would implement how you want to calculate the accuracy
        # using the `predictions` and `targets`
        # ...

print("Evaluation complete")
