import random
from pathlib import Path

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from helper_functions import device, train, make_predictions
from model import TinyVGG

data_dir = Path("data")

if data_dir.exists():
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=False,
        transform=ToTensor(),
        target_transform=None
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,
        transform=ToTensor(),
        target_transform=None
    )
else:
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=None
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
        target_transform=None
    )

class_name = train_data.classes

BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                              batch_size=BATCH_SIZE,
                              shuffle=False)


model = TinyVGG(input_shape=1, hidden_units=10, output_shape=len(class_name)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=0.001)

NUM_EPOCHS = 20

model_results = train(model=model,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS)



test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
  test_samples.append(sample)
  test_labels.append(label)

test_samples[0].shape

pred_probs = make_predictions(model=model,
                              data=test_samples)


pred_classes = pred_probs.argmax(dim=1)


plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
  plt.subplot(nrows, ncols, i+1)

  plt.imshow(sample.squeeze(), cmap='gray')

  pred_label = train_data.classes[pred_classes[i]]

  truth_label = train_data.classes[test_labels[i]]

  title_text = f'pred: {pred_label} | truth: {truth_label}'

  if pred_label == truth_label:
    plt.title(title_text, fontsize=10, c='g')
  else:
    plt.title(title_text, fontsize=10, c='r')

  plt.axis(False)
plt.show()


y_preds = []
model.eval()
with torch.inference_mode():
  for X, y in tqdm(test_dataloader, desc='making preds..'):
    X, y = X.to(device), y.to(device)

    y_logit = model(X)

    y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)

    y_preds.append(y_pred.cpu())


y_pred_tensor = torch.cat(y_preds)


confmat = ConfusionMatrix(num_classes=len(train_data.classes), task="multiclass")
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)

fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=train_data.classes,
    figsize=(10, 7)
)
plt.show()


MODEL_PATH = Path('models')
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)


MODEL_NAME = 'clothes-recognition.pth'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

torch.save(obj=model.state_dict(),
           f=MODEL_SAVE_PATH)


loaded_model = TinyVGG(input_shape=1,
                                     hidden_units=10,
                                     output_shape=len(class_name))

loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

loaded_model.to(device)
