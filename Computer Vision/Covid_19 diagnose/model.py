import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=2):
    resnet50v2 = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
    
    # Freeze all layers except the last one
    for param in resnet50v2.parameters():
        param.requires_grad = False

    # Modify the model to extract 1024 features and add a num_classes-neuron output layer
    resnet50v2.fc = nn.Sequential(
        nn.Linear(resnet50v2.fc.in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes)
    )

    # Unfreeze the last layer
    for param in resnet50v2.fc.parameters():
        param.requires_grad = True

    return resnet50v2

def load_model(model, weights_path):
    model.load_state_dict(torch.load(weights_path))
    return model

def load_feature_extractor(model, weights_path):
    model.fc[:-1].load_state_dict(torch.load(weights_path))
    return model

def extract_deep_features(model, images):
    with torch.no_grad():
        features = model.fc[:-1](model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(images))))))))))
    return features.squeeze()