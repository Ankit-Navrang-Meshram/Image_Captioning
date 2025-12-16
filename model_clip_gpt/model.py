import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.net(x)

class ClipGPT2Model(nn.Module):
    def __init__(self, prefix_length=10, prefix_size=512):
        super().__init__()
        self.prefix_length = prefix_length
        
        # Load GPT-2
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embed_size = self.gpt.transformer.wte.weight.shape[1] # 768 for GPT-2 small
        
        # Mapping Network (CLIP 512 -> GPT 768 * prefix_length)
        # We project the image into 'prefix_length' number of fake words
        self.clip_project = MLP(
            prefix_size, 
            (self.gpt_embed_size * prefix_length) // 2, 
            self.gpt_embed_size * prefix_length
        )

    def forward(self, features, tokens, mask=None):
        # 1. Embed the Image Features
        # features: (batch, 512) -> (batch, prefix_length * 768)
        prefix_embeds = self.clip_project(features)
        
        # Reshape to (batch, prefix_length, 768)
        prefix_embeds = prefix_embeds.view(-1, self.prefix_length, self.gpt_embed_size)
        
        # 2. Embed the Text
        # tokens: (batch, seq_len) -> (batch, seq_len, 768)
        text_embeds = self.gpt.transformer.wte(tokens)
        
        # 3. Concatenate: [Image_Prefix, Caption_Words]
        inputs_embeds = torch.cat((prefix_embeds, text_embeds), dim=1)
        
        # 4. Pass to GPT-2
        # We don't need to pass labels here if we calculate loss manually
        outputs = self.gpt(inputs_embeds=inputs_embeds)
        
        return outputs.logits

if __name__ == "__main__":
    model = ClipGPT2Model()
    dummy_feat = torch.randn(2, 512)
    dummy_toks = torch.randint(0, 1000, (2, 40))
    out = model(dummy_feat, dummy_toks)
    print(f"Output shape: {out.shape}") 
    # Expected: [2, 10 + 40, 50257] (Batch, Prefix+Seq, Vocab)