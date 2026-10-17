"""Prepare the bundled real sklearn digits dataset with disjoint stratified splits."""
from pathlib import Path
import argparse,json,numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('Use a new output file')
 d=load_digits();x=d.images[:,None].astype(np.float32)/16;y=d.target
 train,test=train_test_split(np.arange(len(y)),test_size=.2,random_state=42,stratify=y)
 train,val=train_test_split(train,test_size=.15,random_state=42,stratify=y[train])
 a.output.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(a.output,**{k:v for name,idx in [('train',train),('val',val),('test',test)] for k,v in [('x_'+name,x[idx]),('y_'+name,y[idx])]})
 a.output.with_suffix('.splits.json').write_text(json.dumps({'source':'sklearn.datasets.load_digits: real 1797 handwritten 8x8 images','seed':42,'train':train.tolist(),'validation':val.tolist(),'test':test.tolist()},indent=2)+'\n')
if __name__=='__main__':main()
