import torch
import torch.nn as nn
import torchvision.models as models

# --------------------------
# 1. Encoder (CNN)
# --------------------------
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        # Load pre-trained ResNet50
        # 'weights="DEFAULT"' loads the best available pre-trained weights
        resnet = models.resnet50(weights='DEFAULT')
        
        # Freezing layers (we generally don't retrain the CNN for simple projects)
        for param in resnet.parameters():
            param.requires_grad = False
        
        # Replace the last fully connected layer for classification
        # with a Linear layer to map to our specific embedding size
        modules = list(resnet.children())[:-1] # Remove the last FC layer
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        
        # Optional: Fine-tune CNN? (Default: No)
        if train_CNN:
            for param in self.resnet.parameters():
                param.requires_grad = True

    def forward(self, images):
        # images shape: (batch_size, 3, 224, 224)
        features = self.resnet(images) 
        # features shape: (batch_size, 2048, 1, 1) -> flatten to (batch, 2048)
        features = features.view(features.size(0), -1)
        # map to embedding size
        features = self.embed(features)
        return features

# --------------------------
# 2. Decoder (LSTM)
# --------------------------
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # features shape: (batch_size, embed_size)
        # captions shape: (batch_size, seq_length)
        
        # 1. Create embeddings for the captions
        # We drop the last token of the caption from the input because 
        # the model shouldn't see the <EOS> when predicting the next step.
        embeddings = self.embed(captions[:, :-1]) 
        
        # 2. Concatenate Image Features and Caption Embeddings
        # The image acts as the "first word" or context starter.
        # features need unsqueeze to match seq_len dim: (batch, 1, embed_size)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        # 3. Pass through LSTM
        hiddens, _ = self.lstm(embeddings)
        
        # 4. Map to vocabulary size
        outputs = self.linear(hiddens)
        return outputs

# --------------------------
# 3. Full Model Wrapper
# --------------------------
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(CNNtoRNN, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        """
        Generates a caption for a single image during inference.
        Does NOT use teacher forcing (uses its own previous prediction).
        """
        result_caption = []

        with torch.no_grad():
            x = self.encoder(image.unsqueeze(0)).unsqueeze(1)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                
                result_caption.append(predicted.item())
                
                # Input for next step is the embedding of the predicted word
                x = self.decoder.embed(predicted).unsqueeze(1)

                # Stop if <EOS> is predicted
                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]

# --------------------------
# 4. Test Block
# --------------------------
if __name__ == "__main__":
    embed_size = 256
    hidden_size = 256
    vocab_size = 3000 # Example size
    num_layers = 1
    
    # Initialize model
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers)
    
    # Create dummy data
    images = torch.randn(4, 3, 224, 224)    # Batch of 4 images
    captions = torch.randint(0, 3000, (4, 20)) # Batch of 4 captions, length 20
    
    # Forward pass
    outputs = model(images, captions)
    
    print(f"Output Shape: {outputs.shape}") 
    # Expected: [4, 20, 3000] -> (Batch, Seq_Len, Vocab_Size)