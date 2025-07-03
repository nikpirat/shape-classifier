import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ShapeDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir # path to the root folder containing class subfolders

        # Set up a default transform pipeline if none is provided
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(),  # Convert image to grayscale (1 channel)
            transforms.Resize((48, 48)),  # Resize all images to 48x48
            transforms.ToTensor()  # Convert image to PyTorch tensor and normalize to [0, 1]
        ])
        self.data = []  # List to store full file paths of all images
        self.labels = []  # Corresponding class labels (integers)

        # Map each class folder name to a unique integer label
        self.label_map = {
            "circle": 0,
            "square": 1,
            "triangle": 2,
            "unknown": 3
        }

        # Loop through each class directory
        for label in os.listdir(image_dir):
            class_dir = os.path.join(image_dir, label)
            if os.path.isdir(class_dir):  # Ensure it's a folder
                for file in os.listdir(class_dir):
                    if file.endswith(".png") or file.endswith(".jpg"):  # Only consider image files
                        self.data.append(os.path.join(class_dir, file))  # Add full file path
                        self.labels.append(self.label_map[label])  # Add corresponding label

    def __len__(self):
        return len(self.data) # Return the total number of images

    def __getitem__(self, idx):
        # Load image using PIL
        img = Image.open(self.data[idx]).convert("L")  # Convert again to grayscale just in case

        # Apply transform (resize, tensor conversion, etc.)
        img = self.transform(img)

        # Retrieve the corresponding label
        label = self.labels[idx]

        # Return both image and label
        return img, label
