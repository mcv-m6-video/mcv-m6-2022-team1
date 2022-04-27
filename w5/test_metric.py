import torch
import numpy as np

from pytorch_metric_learning import testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from models import CarIdResnet
from datasets import CarIdDataset

from scipy.spatial.distance import cdist


#%%
device = torch.device("cuda")

model_weights = "/home/pau/Documents/master/M6/project/repo/w5/results/margin02/weights/weights_5.pth"
model = CarIdResnet([512, 256])
weights_dict = torch.load(model_weights)
model.load_state_dict(weights_dict)
model = model.to(device)

dataset = CarIdDataset(
    "/home/pau/Documents/datasets/aicity",
    ["S03"],
    "test"
)

acc_calculator = AccuracyCalculator(
    k='max_bin_count',
    device=device
)

tester = testers.BaseTester()
embeddings = tester.get_all_embeddings(dataset, model)

labels = embeddings[1]
embeddings = embeddings[0]
labels = labels.flatten()

#%%

metrics = acc_calculator.get_accuracy(
    embeddings,
    embeddings,
    labels,
    labels,
    embeddings_come_from_same_source=True
)

#%%
labels = labels.detach().cpu().numpy()
embeddings = embeddings.detach().cpu().numpy()

distances = cdist(embeddings, embeddings, metric="euclidean")
distances = np.triu(distances, k=1)
valid = np.ones(distances.shape, dtype=bool)
valid = np.triu(valid, k=1)

#%%
margin = 0.2

positives = distances < margin
negatives = distances > margin

same_label = labels[:, None] == labels[None, :]

true_positives = np.count_nonzero(positives & same_label & valid)
true_negatives = np.count_nonzero(negatives & np.logical_not(same_label) & valid)

false_positives = np.count_nonzero(positives & np.logical_not(same_label) & valid)
false_negatives = np.count_nonzero(negatives & same_label & valid)

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

f1score = (2 * precision * recall) / (precision + recall)

print(
    f"TP: {true_positives} \t  TN: {true_negatives} \n"
    f"FP: {false_positives} \t FN: {false_negatives} \n"
    f"--------------------------------------------------------------------------------\n"
    f"Precision: {precision}\n"
    f"Recall: {recall}\n"
    f"F1 Score: {f1score}\n"
    f"--------------------------------------------------------------------------------\n"
    f"Total Sum: {sum([true_positives, true_negatives, false_positives, false_negatives])} \n"
    f"Valid: {np.count_nonzero(valid)}"
)

