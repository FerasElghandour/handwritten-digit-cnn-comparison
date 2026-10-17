"""Offline example using the public implementation; see README for evidence scope."""
import sys,json,pathlib,os
repo=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'src'));os.chdir(repo)
import ast,torch,numpy as np
from torch import nn,optim
from torch.utils.data import DataLoader,TensorDataset
from sklearn.datasets import load_digits
torch.set_num_threads(2);torch.manual_seed(42)
scope={'torch':torch,'nn':nn,'optim':optim,'DataLoader':DataLoader,'TensorDataset':TensorDataset,'device':torch.device('cpu')}
nb=json.loads((repo/'analysis.ipynb').read_text())
for cell in nb['cells']:
 if cell['cell_type']!='code':continue
 tree=ast.parse(''.join(cell['source']))
 for node in tree.body:
  if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in ['BaselineCNN','DeepCNN','make_loader','train_model']:exec(compile(ast.Module(body=[node],type_ignores=[]),'analysis.ipynb','exec'),scope)
digits=load_digits();X=(digits.images[:32,None,:,:]/16).astype('float32');y=digits.target[:32].astype('int64')
loader=scope['make_loader'](X,y,batch_size=16,shuffle=False)
model=scope['BaselineCNN']();history=scope['train_model'](model,loader,loader,epochs=1)
print(json.dumps({"fixture":"32 digits, same training/validation samples, demonstration only","logits_shape":list(model(torch.tensor(X[:2])).shape),"history":history}))
