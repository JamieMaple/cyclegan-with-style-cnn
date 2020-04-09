import torch
import torchvision

vgg = torchvision.models.vgg19(pretrained=True).features
vgg.to("cuda" if torch.cuda.is_available() else "cpu")

for param in vgg.parameters():
    param.requires_grad = False

def get_features(image):
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',  # content representation
        '28': 'conv5_1',
    }

    features = {}
    x = image
    for name, layer in vgg._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features

def get_content_loss(output_features, target_features):
    return torch.mean((output_features['conv4_2'] - target_features['conv4_2']) ** 2)

feature_layers_w = {
    'conv1_1': 10.0,
    'conv2_1': 1.0,
    'conv3_1': 1.0,
    'conv4_1': 1.0,
    'conv5_1': 1.0
}
def get_style_loss(output_features, target_features):
    loss = 0
    for layer in feature_layers_w:
        out_gram = gram_matrix(output_features[layer])
        target_gram = gram_matrix(target_features[layer])
        loss += feature_layers_w[layer] * torch.mean((out_gram - target_gram) ** 2)
    return loss


def gram_matrix(output):
    batch_size, d, h, w = output.shape
    features = output.view(batch_size * d, h * w)

    gram = torch.mm(features, features.t())

    return gram.div(batch_size * d * h * w)

