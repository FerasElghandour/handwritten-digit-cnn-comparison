from models import BaselineCNN,DeepCNN
from training import main

def build_model(c):
    if c["classes"]!=10:raise ValueError("Digit architectures expose10 class logits")
    if c["architecture"] not in ("baseline","deep"):raise ValueError("Unknown architecture")
    return DeepCNN() if c["architecture"]=="deep" else BaselineCNN()

if __name__=="__main__":main(build_model,{"classes":10,"architecture":"deep"})
