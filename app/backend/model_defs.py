import torch
import torch.nn as nn
import torchvision.models as models
from transformers import GPT2LMHeadModel

# ==========================================
# MODEL A: CNN + LSTM
# ==========================================
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights='DEFAULT')
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(CNNtoRNN, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []
        with torch.no_grad():
            x = self.encoder(image.unsqueeze(0)).unsqueeze(1)
            states = None
            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoder.embed(predicted).unsqueeze(1)
                if vocabulary.itos[predicted.item()] == "<EOS>": break
        return [vocabulary.itos[idx] for idx in result_caption]

# ==========================================
# MODEL B: CLIP + GPT
# ==========================================
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x): return self.net(x)

class ClipGPT2Model(nn.Module):
    def __init__(self, prefix_length=10, prefix_size=512):
        super().__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embed_size = self.gpt.transformer.wte.weight.shape[1]
        self.clip_project = MLP(prefix_size, (self.gpt_embed_size * prefix_length) // 2, self.gpt_embed_size * prefix_length)

    def forward(self, features, tokens, mask=None):
        prefix_embeds = self.clip_project(features).view(-1, self.prefix_length, self.gpt_embed_size)
        text_embeds = self.gpt.transformer.wte(tokens)
        inputs_embeds = torch.cat((prefix_embeds, text_embeds), dim=1)
        return self.gpt(inputs_embeds=inputs_embeds).logits