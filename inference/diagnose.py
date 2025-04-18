import torch
from PIL import Image
from torchvision import transforms
from model.resnet50 import ResNet50Classifier

def classify_image(image_path, model_path, class_names, device="cpu"):
    model = ResNet50Classifier(len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        predicted = torch.argmax(outputs, dim=1).item()

    return class_names[predicted]
