"""Restricted NumPy reader for the project's own saved dense float32 cVAE.

Only explicit numeric tensor rebuilds and OrderedDict are allowed; arbitrary
pickle globals are rejected. This avoids installing PyTorch for inference.
"""
import collections,io,pickle,zipfile
import numpy as np


def load_checkpoint(path):
    with zipfile.ZipFile(path) as archive:
        key=next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        root=key.rsplit('/',1)[0]
        class Reader(pickle.Unpickler):
            def find_class(self,module,name):
                if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
                if module=='torch' and name=='FloatStorage':return np.dtype('<f4')
                if module=='torch._utils' and name=='_rebuild_tensor_v2':
                    def rebuild(storage,offset,size,stride,*rest):
                        return np.ndarray(tuple(size),dtype=storage.dtype,buffer=storage,
                            offset=offset*storage.itemsize,strides=tuple(s*storage.itemsize for s in stride)).copy()
                    return rebuild
                raise pickle.UnpicklingError('Unsupported checkpoint global '+module+'.'+name)
            def persistent_load(self,pid):
                kind,dtype,storage,location,count=pid
                if kind!='storage' or str(location)!='cpu':raise ValueError('Expected CPU numeric checkpoint')
                result=np.frombuffer(archive.read(root+'/data/'+str(storage)),dtype=dtype)
                if len(result)!=count:raise ValueError('Storage size mismatch')
                return result
        result=Reader(io.BytesIO(archive.read(key))).load()
    return result


class NumpyCVAE:
    def __init__(self,path):
        saved=load_checkpoint(path); self.w=saved['state_dict']; self.meta={k:v for k,v in saved.items() if k!='state_dict'}
    def linear(self,a,key):return a@self.w[key+'.weight'].T+self.w[key+'.bias']
    def encode(self,z,c):
        a=np.concatenate([z,c],axis=1).astype(np.float32)
        a=np.maximum(self.linear(a,'enc.0'),0);a=np.maximum(self.linear(a,'enc.2'),0)
        return self.linear(a,'mu'),np.clip(self.linear(a,'lv'),-10,10)
    def decode(self,z,c):
        a=np.concatenate([z,c],axis=1).astype(np.float32)
        a=np.maximum(self.linear(a,'dec.0'),0);a=np.maximum(self.linear(a,'dec.2'),0)
        return self.linear(a,'dec.4')
