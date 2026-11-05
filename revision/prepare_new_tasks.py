"""Prepare six new public-data tasks, with train-only preprocessing and hashes."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
from scipy import sparse
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE=Path(__file__).resolve().parent
DATA=HERE/'newtask_data'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(url,name):
    path=DATA/'raw'/name;path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url,timeout=90) as f: content=f.read()
        tmp=path.with_suffix('.tmp');tmp.write_bytes(content);tmp.replace(path)
    return path


def split(y):
    idx=np.arange(len(y));a,c=train_test_split(idx,test_size=.15,stratify=y,random_state=42)
    a,b=train_test_split(a,test_size=.15/.85,stratify=np.asarray(y)[a],random_state=42)
    return a,b,c


def save(name,xs,ys,meta,ids=None):
    folder=DATA/name;folder.mkdir(parents=True,exist_ok=True)
    files=[]
    for label,x,y in zip(['train','val','test'],xs,ys):
        if sparse.issparse(x):
            p=folder/f'x_{label}.npz';sparse.save_npz(p,x.astype('float32').tocsr())
        else:
            p=folder/f'x_{label}.npy';np.save(p,np.asarray(x,dtype='float32'))
        yp=folder/f'y_{label}.npy';np.save(yp,np.asarray(y,dtype='int64' if meta['family']!='inverse_imaging' else 'float32'))
        files.extend([p,yp])
    if ids is not None:
        for label,idx in zip(['train','val','test'],ids):
            p=folder/f'indices_{label}.npy';np.save(p,idx);files.append(p)
        assert not set(ids[0])&set(ids[1]) and not set(ids[0])&set(ids[2]) and not set(ids[1])&set(ids[2])
    meta=dict(meta,name=name,seed=42,shape=[list(x.shape) for x in xs],
              preprocessing_fit_split='training only',
              file_sha256={p.name:sha(p) for p in files},
              preparation_source_sha256=sha(__file__))
    (folder/'meta.json').write_text(json.dumps(meta,indent=2))
    print(name,meta['shape'],flush=True)


def imaging():
    raw=download('https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip','optdigits.zip')
    with zipfile.ZipFile(raw) as z:
        tr=np.loadtxt(io.BytesIO(z.read('optdigits.tra')),delimiter=',')
        te=np.loadtxt(io.BytesIO(z.read('optdigits.tes')),delimiter=',')
    a,b=train_test_split(np.arange(len(tr)),test_size=.2,stratify=tr[:,-1],random_state=42)
    xs=[tr[a,:64]/16,tr[b,:64]/16,te[:,:64]/16]
    for name in ['image_denoise','image_inpaint']:
        save(name,xs,xs,dict(family='inverse_imaging',source='https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits',
            source_sha256=sha(raw),original_training=len(tr),original_test=len(te),
            target='clean 8x8 pixel values',corruption='Gaussian sigma .3, clip to [0,1]' if name.endswith('denoise') else '50% independent pixel masking; concatenate visibility mask',
            evaluation='MSE on fixed independent validation/test corruptions',
            metric_definition='all-pixel MSE' if name.endswith('denoise') else 'missing-pixel MSE; observed pixels copied for full-image PSNR',
            corruption_training='fresh corruption every full epoch',metric='mse',higher=False,
            source_test_split='official writer-disjoint split',pixel_normalization='divide by 16'),
            [a,b,np.arange(len(tr),len(tr)+len(te))])


def tabular():
    raw=download('https://archive.ics.uci.edu/static/public/94/spambase.zip','spambase.zip')
    with zipfile.ZipFile(raw) as z:a=np.loadtxt(io.BytesIO(z.read('spambase.data')),delimiter=',')
    idx=split(a[:,-1]);scaler=StandardScaler().fit(a[idx[0],:-1]);xs=[scaler.transform(a[i,:-1]) for i in idx]
    save('tab_spambase',xs,[a[i,-1] for i in idx],dict(family='tabular',metric='auroc',higher=True,
         source='https://archive.ics.uci.edu/dataset/94/spambase',source_sha256=sha(raw),
         split='stratified 70/15/15',n_raw=len(a),preprocessing='training mean and variance; no test clipping'),idx)
    import pandas as pd
    raw=download('https://archive.ics.uci.edu/static/public/2/adult.zip','adult.zip')
    with zipfile.ZipFile(raw) as z:
        tr=pd.read_csv(io.BytesIO(z.read('adult.data')),header=None,skipinitialspace=True,na_values='?')
        te=pd.read_csv(io.BytesIO(z.read('adult.test')),header=None,skipinitialspace=True,na_values='?',comment='|')
    y=(tr[14].str.strip()=='>50K').to_numpy().astype(int)
    yt=(te[14].str.strip().str.rstrip('.')=='>50K').to_numpy().astype(int)
    a,b=train_test_split(np.arange(len(tr)),test_size=.2,stratify=y,random_state=42)
    numeric=[0,2,4,10,11,12];categorical=[i for i in range(14) if i not in numeric]
    scaler=StandardScaler().fit(tr.iloc[a,numeric])
    try:encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    except TypeError:encoder=OneHotEncoder(handle_unknown='ignore',sparse=False)
    encoder.fit(tr.iloc[a,categorical].fillna('__MISSING__').astype(str))
    def transform(df):
        return np.concatenate([scaler.transform(df.iloc[:,numeric]),
            encoder.transform(df.iloc[:,categorical].fillna('__MISSING__').astype(str))],axis=1)
    xs=[transform(tr.iloc[a]),transform(tr.iloc[b]),transform(te)]
    save('tab_adult',xs,[y[a],y[b],yt],dict(family='tabular',metric='auroc',higher=True,
         source='https://archive.ics.uci.edu/dataset/2/adult',source_sha256=sha(raw),
         split='20% stratified validation from official train; official test unchanged',
         n_raw=len(tr)+len(te),preprocessing='numeric train-standardization; train-fitted one-hot categories; missing category explicit'),
         [a,b,np.arange(len(tr),len(tr)+len(te))])


def text():
    raw=download('https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip','sms.zip')
    with zipfile.ZipFile(raw) as z:lines=z.read('SMSSpamCollection').decode('utf-8-sig').splitlines()
    messages={};conflict=set();n_raw=0
    for line in lines:
        if not line.strip():continue
        label,body=line.split('\t',1);assert label in ('ham','spam');body=body.strip();n_raw+=1
        target=int(label=='spam')
        if body in messages and messages[body]!=target:conflict.add(body)
        messages.setdefault(body,target)
    items=[(s,y) for s,y in messages.items() if s not in conflict]
    docs=np.array([s for s,_ in items]);y=np.array([y for _,y in items]);idx=split(y)
    vec=TfidfVectorizer(max_features=2048,min_df=2,max_df=.95,sublinear_tf=True)
    vec.fit(docs[idx[0]]);xs=[vec.transform(docs[i]) for i in idx]
    save('text_sms',xs,[y[i] for i in idx],dict(family='text',metric='macro_f1',higher=True,
         source='https://archive.ics.uci.edu/dataset/228/sms+spam+collection',source_sha256=sha(raw),
         n_raw=n_raw,n_unique=len(items),conflicting_messages_removed=len(conflict),
         duplicate_policy='exact stripped message deduplication before stratified 70/15/15 split',
         preprocessing='train-only TF-IDF; max_features=2048; min_df=2; max_df=.95; sublinear_tf; L2 norm'),idx)
    tr=fetch_20newsgroups(data_home=str(DATA/'raw/sklearn'),subset='train',remove=('headers','footers','quotes'),random_state=42)
    te=fetch_20newsgroups(data_home=str(DATA/'raw/sklearn'),subset='test',remove=('headers','footers','quotes'),random_state=42)
    a,b=train_test_split(np.arange(len(tr.target)),test_size=.2,stratify=tr.target,random_state=42)
    vec=TfidfVectorizer(max_features=2048,min_df=2,max_df=.95,sublinear_tf=True)
    docs=np.array(tr.data);vec.fit(docs[a]);xs=[vec.transform(docs[a]),vec.transform(docs[b]),vec.transform(te.data)]
    training_texts=set(docs[a])
    save('text_20news',xs,[tr.target[a],tr.target[b],te.target],dict(family='text',metric='macro_f1',higher=True,
         source='https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html',
         n_raw=len(tr.target)+len(te.target),split='20% stratified validation from by-date training; official by-date test',
         preprocessing='remove headers/footers/quotes; train-only TF-IDF; vocabulary <=2048',
         empty_training_documents=int(sum(not s.strip() for s in docs[a])),
         test_text_overlap_train=int(sum(s in training_texts for s in te.data)),
         overlap_note='official corpus retained; no claim of exact-text independence across its predefined split'),
         [a,b,np.arange(len(tr.target),len(tr.target)+len(te.target))])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--family',choices=['all','image','tabular','text'],default='all')
    arg=ap.parse_args().family;DATA.mkdir(exist_ok=True)
    if arg in ('all','image'):imaging()
    if arg in ('all','tabular'):tabular()
    if arg in ('all','text'):text()
