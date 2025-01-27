import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

if __name__=='__main__':
    texts = ["cubesmall inspect", "cubesmall lift"]
    text_inputs = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    print(text_features.shape)
    pass