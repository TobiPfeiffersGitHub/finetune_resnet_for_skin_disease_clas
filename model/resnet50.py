import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output
            self.activations.requires_grad = True

        def backward_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_index=None):
        self.model.eval()
        logits = self.model(input_tensor)
        if class_index is None:
            class_index = torch.argmax(logits, dim=1).item()

        self.model.zero_grad()
        loss = logits[:, class_index].sum()
        loss.backward()

        grads_mean = self.gradients.mean(dim=[0, 2, 3], keepdim=True)
        gradcam_map = (self.activations * grads_mean).sum(dim=1, keepdim=True)
        gradcam_map = F.relu(gradcam_map)
        gradcam_map = F.interpolate(gradcam_map, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)

        gradcam_map -= gradcam_map.min()
        gradcam_map /= gradcam_map.max() + 1e-8
        return gradcam_map.detach()

    @staticmethod
    def overlay(image_tensor, heatmap, alpha=0.5):
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        heatmap_np = heatmap.squeeze().cpu().numpy()
        color_map = plt.cm.jet(heatmap_np)[:, :, :3]
        overlayed = np.clip(alpha * color_map + (1 - alpha) * image_np, 0, 1)
        return overlayed
