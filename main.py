"""
Prototypical Network for NIH Chest X-ray (FSL + WANDB SWEEP VERSION)
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# WANDB SWEEP SETTINGS (MANUAL GRID)
# =========================
k_shots_list = [1, 5]
lrs = [1e-4, 1e-3]

# =========================
# LABELS
# =========================
ALL_LABELS = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration",
    "Mass","Nodule","Pneumonia","Pneumothorax",
    "Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia"
]

label_to_idx = {l:i for i,l in enumerate(ALL_LABELS)}

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# DATASET
# =========================
class NIHSingleLabelDataset(Dataset):
    def __init__(self, img_dir, csv_path=None, samples=None):
        self.img_dir = img_dir
        self.samples = [] if samples is None else list(samples)

        if samples is None:
            df = pd.read_csv(csv_path)

            for _, row in df.iterrows():
                img = row["Image Index"]
                labels = row["Finding Labels"].split("|")

                for l in labels:
                    if l in label_to_idx:
                        self.samples.append((img, label_to_idx[l]))
                        break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        img = transform(img)

        return img, label

# =========================
# ENCODER
# =========================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, 128)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# =========================
# EPISODE SAMPLER (UPDATED WITH k-shot)
# =========================
def sample_episode(dataset, n_way=5, k_shot=5, q_query=3):
    eligible_classes = []
    required_samples = k_shot + q_query

    for cls in range(len(ALL_LABELS)):
        cls_count = sum(1 for _, label in dataset.samples if label == cls)
        if cls_count >= required_samples:
            eligible_classes.append(cls)

    if len(eligible_classes) < n_way:
        raise ValueError(
            f"Need at least {n_way} classes with {required_samples}+ samples, "
            f"but found {len(eligible_classes)}."
        )

    classes = random.sample(eligible_classes, n_way)

    support_x, support_y = [], []
    query_x, query_y = [], []

    for cls in classes:
        cls_samples = [(x,y) for x,y in dataset.samples if y == cls]
        selected = random.sample(cls_samples, k_shot + q_query)

        support = selected[:k_shot]
        query = selected[k_shot:]

        for img_name, label in support:
            img = Image.open(os.path.join(dataset.img_dir, img_name)).convert("RGB")
            support_x.append(transform(img))
            support_y.append(label)

        for img_name, label in query:
            img = Image.open(os.path.join(dataset.img_dir, img_name)).convert("RGB")
            query_x.append(transform(img))
            query_y.append(label)

    return (
        torch.stack(support_x),
        torch.tensor(support_y),
        torch.stack(query_x),
        torch.tensor(query_y)
    )


def split_dataset(dataset, test_ratio=0.2, seed=42):
    rng = random.Random(seed)
    grouped_samples = {label: [] for label in range(len(ALL_LABELS))}

    for sample in dataset.samples:
        grouped_samples[sample[1]].append(sample)

    train_samples = []
    test_samples = []

    for label_samples in grouped_samples.values():
        if not label_samples:
            continue

        shuffled = label_samples[:]
        rng.shuffle(shuffled)

        split_idx = max(1, int(len(shuffled) * (1 - test_ratio)))
        if split_idx >= len(shuffled):
            split_idx = len(shuffled) - 1

        train_samples.extend(shuffled[:split_idx])
        test_samples.extend(shuffled[split_idx:])

    train_dataset = NIHSingleLabelDataset(dataset.img_dir, samples=train_samples)
    test_dataset = NIHSingleLabelDataset(dataset.img_dir, samples=test_samples)
    return train_dataset, test_dataset

# =========================
# PROTOTYPICAL LOSS
# =========================
def prototypical_loss(embeddings, labels):
    prototypes = []

    for c in torch.unique(labels):
        prototypes.append(embeddings[labels == c].mean(0))

    prototypes = torch.stack(prototypes)

    dists = torch.cdist(embeddings, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1)

    target_inds = torch.zeros(len(labels), dtype=torch.long).to(device)

    for i, label in enumerate(torch.unique(labels)):
        target_inds[labels == label] = i

    loss = F.nll_loss(log_p_y, target_inds)
    acc = (log_p_y.argmax(dim=1) == target_inds).float().mean()

    return loss, acc

# =========================
# TRAIN FUNCTION (UPDATED)
# =========================
def evaluate_fsl(model, dataset, k_shot, eval_episodes=20):
    model.eval()
    losses = []
    accuracies = []

    with torch.no_grad():
        for _ in range(eval_episodes):
            support_x, support_y, query_x, query_y = sample_episode(
                dataset,
                n_way=5,
                k_shot=k_shot
            )

            support_x = support_x.to(device)
            query_x = query_x.to(device)

            emb_support = model(support_x)
            emb_query = model(query_x)

            embeddings = torch.cat([emb_support, emb_query], dim=0)
            labels = torch.cat([support_y, query_y], dim=0).to(device)

            loss, acc = prototypical_loss(embeddings, labels)
            losses.append(loss.item())
            accuracies.append(acc.item())

    model.train()
    return sum(losses) / len(losses), sum(accuracies) / len(accuracies)


def train_fsl(train_dataset, test_dataset, episodes, k_shot, lr):
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is required for training but is not installed.") from exc

    # NEW WANDB RUN PER COMBINATION
    wandb.init(
        project="NIH-FSL-Prototypical",
        name=f"k{k_shot}_lr{lr}",
        config={
            "k_shot": k_shot,
            "lr": lr,
            "episodes": episodes
        },
        reinit=True
    )
    wandb.define_metric("episode")
    wandb.define_metric("train_loss", step_metric="episode")
    wandb.define_metric("train_accuracy", step_metric="episode")
    wandb.define_metric("test_loss", step_metric="episode")
    wandb.define_metric("test_accuracy", step_metric="episode")

    model = Encoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(episodes):

        support_x, support_y, query_x, query_y = sample_episode(
            train_dataset,
            n_way=5,
            k_shot=k_shot
        )

        support_x = support_x.to(device)
        query_x = query_x.to(device)

        emb_support = model(support_x)
        emb_query = model(query_x)

        embeddings = torch.cat([emb_support, emb_query], dim=0)
        labels = torch.cat([support_y, query_y], dim=0).to(device)

        loss, acc = prototypical_loss(embeddings, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # =========================
        # WANDB LOGGING
        # =========================
        test_loss, test_acc = evaluate_fsl(model, test_dataset, k_shot)

        wandb.log({
            "episode": ep,
            "train_loss": loss.item(),
            "train_accuracy": acc.item(),
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        })

        if ep % 20 == 0:
            print(
                f"[k={k_shot}, lr={lr}] Ep {ep} | "
                f"Train Loss {loss:.4f} | Train Acc {acc:.4f} | "
                f"Test Loss {test_loss:.4f} | Test Acc {test_acc:.4f}"
            )
        # =========================
    # SAVE MODEL
    # =========================
    save_path = f"proto_model_k{k_shot}_lr{lr}.pth"
    torch.save(model.state_dict(), save_path)

    print(f"Model saved to {save_path}")

    wandb.finish()
    return model


def load_trained_model(model_path):
    model = Encoder().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_image(image_source):
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")
    return Image.open(image_source).convert("RGB")


def image_to_embedding(model, image_source):
    image = load_image(image_source)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(tensor).squeeze(0)

    return embedding


def build_prototypes(model, labeled_images):
    grouped_embeddings = {}

    for label, image_source in labeled_images:
        embedding = image_to_embedding(model, image_source)
        grouped_embeddings.setdefault(label, []).append(embedding)

    prototypes = {}
    for label, embeddings in grouped_embeddings.items():
        prototypes[label] = torch.stack(embeddings).mean(dim=0)

    return prototypes


def predict_with_prototypes(model, image_source, prototypes, top_k=5):
    if not prototypes:
        raise ValueError("At least one support prototype is required for prediction.")

    query_embedding = image_to_embedding(model, image_source)

    scored = []
    for label, prototype in prototypes.items():
        distance = torch.norm(query_embedding - prototype, p=2).item()
        scored.append((label, distance))

    scored.sort(key=lambda item: item[1])
    distance_tensor = torch.tensor([-distance for _, distance in scored], device=device)
    confidence_scores = torch.softmax(distance_tensor, dim=0).cpu().tolist()

    ranked_predictions = [
        {
            "label": label,
            "distance": float(distance),
            "score": float(score),
        }
        for (label, distance), score in zip(scored[:top_k], confidence_scores[:top_k])
    ]

    top_prediction = ranked_predictions[0]
    return ranked_predictions, top_prediction
    

# =========================
# MAIN SWEEP
# =========================
if __name__ == "__main__":

    img_dir = r"C:\Users\sarva\Downloads\NIHIHIH_extracted\images-224\images-224"
    csv_path = r"C:\Users\sarva\Downloads\Data_Entry_2017.csv"

    dataset = NIHSingleLabelDataset(img_dir, csv_path)
    train_dataset, test_dataset = split_dataset(dataset)

    EPOCHS = 300

    for k in k_shots_list:
        for lr in lrs:

            print(f"\n================ RUN k={k} lr={lr} ================\n")

            train_fsl(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                episodes=EPOCHS,
                k_shot=k,
                lr=lr
            )
