import pickle
import numpy as np
from safetensors.numpy import load_file, load
from safetensors.numpy import save_file
import copy
import mmap
import io
from  collections import OrderedDict
import json
MAX_HEADER_SIZE = 100 * 1000 * 1000
import time

dtype_size = {
    "BOOL" : 1,
    "U8" : 1,
    "I8" : 1,
    "F8_E5M2" : 1,
    "F8_E4M3" : 1,
    "I16" : 2,
    "U16" : 2,
    "I32" : 4,
    "U32" : 4,
    "I64" : 8,
    "U64" : 8,
    "F16" : 2,
    "BF16" : 2,
    "F32" : 4,
    "F64" : 8,
    }

numpy_dtype = {
    "BOOL" : np.bool_,
    "U8" : np.uint8,
    "I8" : np.int8,
    "F8_E5M2" : 1, # no fp8
    "F8_E4M3" : 1, # no fp8
    "I16" : np.int16,
    "U16" : np.uint16,
    "I32" : np.int32,
    "U32" : np.uint32,
    "I64" : np.int64,
    "U64" : np.uint64,
    "F16" : np.float16,
    "BF16" : 2, # no bf16
    "F32" : np.float32,
    "F64" : np.float64,
    }


def getSize(fileobject):
    fileobject.seek(0,2) # move the cursor to the end of the file
    size = fileobject.tell()
    fileobject.seek(0) # move the cursor to the start of the file
    return size

MFILE = "/root/paddlejob/workspace/env_run/zhonghui03/PaddleNLP/unified_checkpoint/checkpoints/llama_pretrain_ckpts/checkpoint-12/model-00001-of-00008.safetensors"
def metadata_validate(metadata):
    start = 0;
    for key, info in metadata.items():
        print(key, info)
        s, e = info["data_offsets"];
        if s != start or e < s:
            raise ValueError(f"SafeTensorError::InvalidOffset({key})")
        start = e;
        nelements  = np.prod(info["shape"]) 
        nbytes = nelements * dtype_size[info["dtype"]] 
        if (e - s) != nbytes:
            raise ValueError("SafeTensorError::TensorInvalidInfo");
    return start

def read_metadata(buffer):
    buffer_len = getSize(buffer);
    if buffer_len < 8 :
        raise ValueError("SafeTensorError::HeaderTooSmall") 
    
    n =  np.frombuffer(buffer.read(8),dtype=np.uint64).item() 
    if n > MAX_HEADER_SIZE:
        raise ValueError("SafeTensorError::HeaderTooLarge")

    stop = n + 8
    if stop > buffer_len:
        raise ValueError("SafeTensorError::InvalidHeaderLength")
    
    tensors = json.loads(buffer.read(n), object_pairs_hook=OrderedDict)
    metadata = tensors.pop("__metadata__", None)
    # print(metadata)
    buffer_end = metadata_validate(tensors);

    if buffer_end + 8 + n != buffer_len:
        raise ValueError("SafeTensorError::MetadataIncompleteBuffer");
    
    return stop, tensors, metadata

def readinto_numpy(meta, buffer, base_ptr):
    def create_empty(info):
        return np.empty(shape=info["shape"], dtype=numpy_dtype[info["dtype"]]) 
    ret = {}
    for k,v in meta.items():
        t = create_empty(v)
        buffer.seek(base_ptr + v["data_offsets"][0])
        buffer.readinto(memoryview(t))
        ret[k] = t
    return ret


class PySafeSlice:
    def __init__(self, info, bufferfile, base_ptr):
        self.info = info 
        self.bufferfile = bufferfile
        self.base_ptr = base_ptr

        self.start = [0 for dim in self.shape]
        self.stop = [dim for dim in self.shape]
        self.step = [1 for dim in self.shape]

    @property
    def ndim(self):
        return len(self.shape)

    def __getitem__(self, index):        
        # https://github.com/numpy/numpy/blob/4d652465cea38e9504f954ac708d91e4954bd13a/numpy/lib/_arrayterator_impl.py#L96-L126
        # Fix index, handling ellipsis and incomplete slices.
        if not isinstance(index, tuple):
            index = (index,)
        fixed = []
        length, dims = len(index), self.ndim
        for slice_ in index:
            if slice_ is Ellipsis:
                fixed.extend([slice(None)] * (dims-length+1))
                length = len(fixed)
            elif isinstance(slice_, int):
                fixed.append(slice(slice_, slice_+1, 1))
            else:
                fixed.append(slice_)
        index = tuple(fixed)
        if len(index) < dims:
            index += (slice(None),) * (dims-len(index))

        out_start, out_stop, out_step = copy.deepcopy((self.start, self.stop, self.step))
        for i, (start, stop, step, slice_) in enumerate(
                zip(self.start, self.stop, self.step, index)):
            out_start[i] = slice_.start or 0
            out_step[i] = slice_.step or 1
            out_stop[i] = slice_.stop or stop-start
            out_stop[i] = min(stop, out_stop[i])

        target_shape = []
        for x,y,z in zip(out_start, out_stop, out_step):
            print(x,y,z)
            assert z == 1, "only support step = 1"
            if y-x>1:
                target_shape.append(y-x)

        # https://github.com/huggingface/safetensors/blob/b947b59079a6197d7930dfb535818ac4896113e8/safetensors/src/slice.rs#L297-L315
        indices = []
        span = self.bits 
        for i, (start, stop, step) in  enumerate(zip(out_start[::-1], out_stop[::-1], out_step[::-1])):
            if len(indices) == 0:
                if start == 0 and stop == self.shape[i]:
                    pass
                    #  We haven't started to slice yet, just increase the span
                else:
                    offset = start * span
                    small_span = stop * span - offset
                    indices.append((offset, offset + small_span))
                
            else:
                capacity = (stop - start) * len(indices);
                newindices = [];
                for n in range(start, stop):
                    offset = n * span
                    for (old_start, old_stop) in indices:
                        newindices.append((old_start + offset, old_stop + offset))
                indices = newindices
                assert len(indices) == capacity, f"error {capacity} {len(indices)}"
            span *= self.shape[-(i+1)]          
        
        # print(indices)
        merge_indices = []
        last_end = -1
        last_start = -1
        for start, end in indices:
            if start == last_end:
                last_end = end
                continue
            else:
                if last_start != -1:
                    merge_indices.append((last_start, last_end))
                last_start = start
                last_end = end
        if last_start != -1:  
            merge_indices.append((last_start, last_end))
        print(merge_indices)
        print(len(indices),len(merge_indices))
        
        tensor = np.empty(shape=target_shape, dtype=self.dtype)

        curr_data_ptr = 0      
        for start, end in indices:
            data_len = end - start
            self.bufferfile.seek(self.start_offset + start)
            self.bufferfile.readinto(memoryview(tensor)[curr_data_ptr:curr_data_ptr + data_len])
            print(tensor.reshpe(-1)[20])
            curr_data_ptr += data_len
        print(tensor, tensor.shape)
        return tensor

    def get(self, *args, **kwargs):        
        tensor = np.empty(shape=self.shape, dtype=self.dtype) 
        self.bufferfile.seek(self.start_offset)
        self.bufferfile.readinto(memoryview(tensor))
        return tensor
    
    @property
    def start_offset(self):
        return self.base_ptr + self.info["data_offsets"][0]
    
    @property
    def shape(self):
        return self.info["shape"]

    @property
    def dtype(self):
        return numpy_dtype[self.info["dtype"]]

    @property
    def nelements(self): 
        return  np.prod(self.info["shape"])

    @property
    def bits(self):
        return dtype_size[self.info["dtype"]]
    
    @property
    def nbytes(slef): 
        return  nelements * dtype_size[self.info["dtype"]]

# a simple file writer object

class fast_safe_open:
    def __init__(self, filename, framework=None):
        self.filename = filename
        self.framework = framework
        self.file = open(self.filename, 'rb')
        self.base, self.tensors_decs, self.metadata = read_metadata(self.file)
        self.tensors = OrderedDict()
        for key,info in self.tensors_decs.items():
            self.tensors[key] = PySafeSlice(info, self.file, self.base)
             
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.file.close()
    
    def metadata(self):
        return self.metadata

    def keys(self):
        return list(self.tensors.keys())

    def get_tensor(self, name):
        return self.tensors[name].get()
    
    def get_slice(self, name):
        return self.tensors[name]
        

def fast_load_file(filename):
    result = {}
    with fast_safe_open(filename, framework="np") as f:
        for k in f.keys():
            result[k] = f.get_tensor(k)
            t = f.get_slice(k)
            np.testing.assert_equal(result[k][2], t[2])
            a = result[k][...,0:10]
            b = t[..., 0:10]
            res = np.where(result[k] == b[1][1])
            print(res)
            np.testing.assert_equal(result[k][...,0:10], t[..., 0:10])

    return result








t1 = time.time()
fp = "test_file.safetensors"
fast_load_file(MFILE)
print("my sf load:", time.time()-t1)

w = np.empty([256,1024,1024])

print(w.dtype)

state_dict = {"weight": w}

import time
fp = "test_file.safetensors"
# t1 = time.time()
# save_file(state_dict,fp)
# print("sf save:",  time.time()-t1)

t1 = time.time()
state = load_file(fp) 
print("sf load:", time.time()-t1)

t1 = time.time()
# buffers = bytearray(1000*1024*1024)

a = np.empty(1000*1024*1024, dtype=np.uint8)
buffers = memoryview(a)


with open(fp, "rb") as f:
    # mm = mmap.mmap(f.fileno(), 0)
    # mm.seek(1500*1024*1024)
    # data = mm.read()
    print(type(f))
    # data = f.read() 
    f.readinto(buffers[:30])
    print(a[:30])
    f.seek(15)
    f.seek(0)
    f.readinto(buffers[:30])
    print(a[:30])
    f.readinto(buffers[100:500*1024*1024])
    f.seek(1000*1024*1024)
    f.readinto(buffers[500*1024*1024:])
# data=bytes(buffers)
print(a[:30])
print("mm read:", time.time()-t1)
data=buffers.tobytes()
print(type(data))
t2 = time.time()
loaded = load(data)
print("sf load2:", time.time()-t1, time.time() - t2)


fp = fp.replace(".safetensors", ".pickle")
# t1 = time.time()
# state = pickle.dump(state_dict, open(fp, "wb"))
# print("pickle save:",  time.time()-t1)
 

t1 = time.time()
state = pickle.load(open(fp, "rb"))
a = state["weight"]
print("pickle load:", time.time()-t1)
#a = np.ascontiguousarray(a)
t1 = time.time()
# c = copy.deepcopy(a)
c = np.copy(a)
# w[...] = a
# print(w[0,0,0])
print("numpy copy:", time.time()-t1)
