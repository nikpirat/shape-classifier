import torch
from torch.utils.data import DataLoader
from src.model import ShapeClassifier
from src.dataset import ShapeDataset
from sklearn.metrics import classification_report

def evaluate_model(data_path, model_path):
    dataset = ShapeDataset(data_path)
    loader = DataLoader(dataset, batch_size=32)

    model = ShapeClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            output = model(x)
            preds = output.argmax(1)
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    print(classification_report(all_labels, all_preds, target_names=["circle", "square", "triangle", "unknown"]))