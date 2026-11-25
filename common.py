"""Small deterministic experimental utilities, copied into each standalone project."""
import hashlib
import json
import re
import time
import numpy as np

MODEL_PROVENANCE=[]

def rng(seed=42): return np.random.default_rng(seed)
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-30,30)))
def fit(x,y,steps=350,rate=.15,weights=None):
    x=np.c_[np.ones(len(x)),x]; w=np.zeros(x.shape[1]);weights=np.ones(len(y)) if weights is None else weights
    for _ in range(steps): w-=rate*x.T@((sigmoid(x@w)-y)*weights)/len(y)
    return w
def probability(x,w):return sigmoid(np.c_[np.ones(len(x)),x]@w)
def measures(y,p,t=.5):
    pred=p>=t;truth=np.asarray(y,dtype=bool)
    tp=int(sum(pred&truth));fp=int(sum(pred&~truth));fn=int(sum(~pred&truth))
    precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn)
    order=np.argsort(-np.asarray(p),kind='stable');sorted_y=truth[order];pr=np.cumsum(sorted_y)/np.arange(1,len(y)+1)
    return {'accuracy':float(np.mean(pred==truth)),'precision':precision,'recall':recall,
            'f1':2*precision*recall/max(1e-12,precision+recall),'average_precision':float(sum(pr*sorted_y)/max(1,sum(truth))),
            'fp':fp,'fn':fn,'cases':len(y)}
def classification(seed=7,n=900,d=8):
    r=rng(seed);x=r.normal(size=(n,d));y=(x[:,0]+.7*x[:,1]-.4*x[:,2]+r.normal(0,.7,n)>0).astype(float)
    return x[:500],y[:500],x[500:700],y[500:700],x[700:],y[700:]
def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def softmax(x):
    z=x-np.max(x,axis=-1,keepdims=True);e=np.exp(z);return e/e.sum(axis=-1,keepdims=True)
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def bow_fit(texts):return {w:i for i,w in enumerate(sorted(set(re.findall(r'\w+',' '.join(texts).lower()))))}
def bow(texts,vocab):
    a=np.zeros((len(texts),len(vocab)))
    for row,text in enumerate(texts):
        for w in re.findall(r'\w+',text.lower()):
            if w in vocab:a[row,vocab[w]]+=1
    return a
def timed(fn,repeats=5):
    fn();dur=[]
    for _ in range(repeats):
        start=time.perf_counter();fn();dur.append(time.perf_counter()-start)
    return float(np.median(dur))
def check(condition,message):
    if not condition:raise AssertionError(message)
    return message

def model_path(kind):
    import os
    from pathlib import Path
    locations={
        'encoder':[os.environ.get('AI40_ENCODER',''),r'D:\AI-Projects\cache\ai40-models\encoder',r'D:\AI-Projects\outputs\ServeBench\private\base'],
        'causal':[os.environ.get('AI40_CAUSAL',''),r'D:\AI-Projects\cache\ai40-models\causal',r'D:\AI-Projects\outputs\AdapterForge\private\base-12fd25f77366'],
        'whisper':[os.environ.get('AI40_WHISPER',''),r'D:\AI-Projects\cache\ai40-models\whisper']}
    for folder in locations[kind]:
        if folder and (Path(folder)/'config.json').is_file():
            config=Path(folder)/'config.json';meta={'kind':kind,'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest()}
            for name in ['ai40-provenance.json','source-manifest.json','manifest.json']:
                path=Path(folder)/name
                if path.is_file():
                    raw=json.loads(path.read_text(encoding='utf-8'))
                    meta['upstream_model']=raw.get('repo',raw.get('model_id',raw.get('model','not recorded')))
                    meta['revision']=raw.get('revision','not recorded');break
            weight=Path(folder)/'model.safetensors'
            if weight.is_file():
                h=hashlib.sha256()
                with weight.open('rb') as stream:
                    while chunk:=stream.read(2**20):h.update(chunk)
                meta['weights_sha256']=h.hexdigest()
            if meta not in MODEL_PROVENANCE:MODEL_PROVENANCE.append(meta)
            return folder
    raise RuntimeError('Missing local '+kind+' model. Run manager.py prepare-models --kind '+kind+' or set AI40_'+kind.upper())

def encode(texts):
    import torch
    from transformers import AutoTokenizer,AutoModel
    torch.set_num_threads(2);path=model_path('encoder');tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModel.from_pretrained(path,local_files_only=True,use_safetensors=True).eval()
    batch=tok(texts,padding=True,truncation=True,max_length=128,return_tensors='pt')
    with torch.no_grad():
        values=model(**batch).last_hidden_state;mask=batch['attention_mask'].unsqueeze(-1)
        result=(values*mask).sum(1)/mask.sum(1)
    return normalize(result.numpy())

def causal_model():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.set_num_threads(2);path=model_path('causal')
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,use_safetensors=True).eval()
    return model,tok

def retrieval_fixture():
    topics=[('backup','restore the database snapshot'),('password','reset an account password'),('queue','drain queued worker jobs'),
            ('cache','invalidate stale cached records'),('email','inspect mail delivery failures'),('network','trace dropped network packets'),
            ('disk','free disk storage space'),('certificate','renew an expired TLS certificate'),('index','rebuild the search index'),
            ('metrics','inspect service latency measurements'),('permissions','revoke a service account role'),('deployment','roll back a failed release')]
    docs=[f'The {name} operations guide explains how to {action}.' for name,action in topics]
    queries=[f'How can I {action} safely?' for _,action in topics]
    return docs,queries
