import torch
import torch.nn as nn

class DualTowerModel(nn.Module):
    def __init__(self, num_classes, vocab_size=30522, embed_dim=768):
        super(DualTowerModel, self).__init__()
        self.image_tower = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.text_tower = nn.Sequential(
            nn.Embedding(vocab_size, embed_dim),
            nn.Linear(embed_dim, 256),
            nn.ReLU()
        )
        self.classifier = nn.Linear(256 + 256, num_classes)

    def forward(self, images, input_ids, attention_mask):
        image_features = self.image_tower(images)
        text_features = self.text_tower(input_ids).mean(dim=1)
        combined_features = torch.cat((image_features, text_features), dim=1)
        output = self.classifier(combined_features)
        return output
