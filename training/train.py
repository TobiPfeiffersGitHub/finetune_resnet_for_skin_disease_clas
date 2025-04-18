import torch
import torch.nn.functional as F
from torch import optim

def soft_f1_loss(logits, labels, epsilon=1e-7):
    num_classes = logits.shape[1]
    labels_one_hot = F.one_hot(labels, num_classes).float()
    probs = torch.softmax(logits, dim=1)

    tp = probs * labels_one_hot
    fp = probs * (1 - labels_one_hot)
    fn = (1 - probs) * labels_one_hot

    tp_sum = tp.sum(dim=0)
    fp_sum = fp.sum(dim=0)
    fn_sum = fn.sum(dim=0)

    soft_f1 = 2 * tp_sum / (2 * tp_sum + fp_sum + fn_sum + epsilon)
    return 1 - soft_f1.mean()

def train(model, train_loader, val_loader, device, lr=1e-3, epochs=10, save_path="resnet50.pth"):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = soft_f1_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += soft_f1_loss(outputs, labels).item()
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
