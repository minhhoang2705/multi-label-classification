# PyTorch Model Serving Patterns for Production FastAPI (2025)

## 1. Model Loading Strategies

### Lifespan Events (Modern, Recommended)
```python
from contextlib import asynccontextmanager
import torch

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model before accepting requests
    global model
    model = torch.load("model.pt")
    model.eval()
    model.to("cuda")
    yield
    # Shutdown: Cleanup
    del model
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)
```

### Gunicorn Preload (Memory-Efficient Multi-Worker)
```python
# app.py
model = torch.load("model.pt")
model.eval()
model.share_memory_()  # Critical for sharing across workers

app = FastAPI()

# gunicorn.conf.py
preload_app = True
workers = 4
timeout = 120  # Increase for large models (default 30s)
worker_class = "uvicorn.workers.UvicornWorker"
```

**Effect**: 1GB model × 4 workers = 1GB RAM (not 4GB). Avoid modifying model post-load (triggers copy-on-write).

### Lazy Loading vs Startup Loading
- **Startup**: Preferred. Fail-fast. No latency spike on first request
- **Lazy**: Use only for multi-model serving with infrequent access

## 2. Inference Optimization

### torch.compile (PyTorch 2.x)
```python
model = torch.compile(model, mode="max-autotune")  # Best inference speed
# Modes: "default" (fast compile), "reduce-overhead", "max-autotune" (slow compile, fast inference)

# Compile-time caching for faster restarts
torch._inductor.config.fx_graph_cache = True
```

**Performance**: Comparable to TensorRT. 1st inference slow (compilation), subsequent fast.

### Inference Mode (Preferred over no_grad)
```python
with torch.inference_mode():  # More optimizations than no_grad
    output = model(input_tensor)
```

### Threading Control
```python
torch.set_num_threads(4)  # Sweet spot for Uvicorn stability
```

### TorchScript/ONNX Considerations
- **TorchScript**: Use for mobile/edge. Overhead for server deployment
- **ONNX**: Consider for heterogeneous runtimes. torch.compile often sufficient

## 3. Batch Processing

### Dynamic Batching with Queue
```python
from asyncio import Queue, create_task, wait_for
from datetime import datetime

class BatchProcessor:
    def __init__(self, max_batch_size=32, max_latency_ms=100):
        self.queue = Queue()
        self.max_batch_size = max_batch_size
        self.max_latency_ms = max_latency_ms
        create_task(self._process_loop())

    async def predict(self, input_data):
        future = asyncio.Future()
        await self.queue.put((input_data, future))
        return await future

    async def _process_loop(self):
        while True:
            batch = []
            futures = []
            deadline = datetime.now() + timedelta(milliseconds=self.max_latency_ms)

            while len(batch) < self.max_batch_size:
                timeout = (deadline - datetime.now()).total_seconds()
                if timeout <= 0:
                    break
                try:
                    data, fut = await wait_for(self.queue.get(), timeout)
                    batch.append(data)
                    futures.append(fut)
                except asyncio.TimeoutError:
                    break

            if batch:
                results = await self._batch_inference(batch)
                for fut, result in zip(futures, results):
                    fut.set_result(result)

    async def _batch_inference(self, batch):
        with torch.inference_mode():
            tensors = torch.stack(batch).to("cuda")
            return model(tensors).cpu()
```

**Libraries**: `service-streamer`, `batched`, or custom queue-based implementation.

## 4. Memory Management

### GPU Memory Cleanup
```python
import gc

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# After inference
result = model(input)
del input
cleanup()
```

### Memory Profiling
```python
import torch.cuda.memory

torch.cuda.memory._record_memory_history(enabled=True)
# ... inference ...
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
```

### Multi-Worker GPU Memory
```python
# Limit per-process GPU memory
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
```

**Issue**: Multiple Uvicorn workers → OOM. Solution: Gunicorn preload or single worker per GPU.

## 5. Warm-Up Strategies

### Pre-Loading with Dummy Inference
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = torch.load("model.pt")
    model.eval()
    model.to("cuda")

    # Warm-up: Trigger torch.compile compilation, allocate GPU memory
    dummy_input = torch.randn(1, 3, 224, 224).to("cuda")
    with torch.inference_mode():
        _ = model(dummy_input)

    yield
    del model
    torch.cuda.empty_cache()
```

**Purpose**: Compile model, allocate GPU memory pools, prevent first-request latency spike.

## 6. Error Handling

### Model Loading Failures
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = torch.load("model.pt", weights_only=True)  # Security: weights_only=True
        model.eval()
        model.to("cuda")
    except FileNotFoundError:
        logger.error("Model file not found")
        raise RuntimeError("Model initialization failed")
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU OOM during model load")
        model.to("cpu")  # Fallback to CPU

    yield
```

### Inference Errors with Fallback
```python
@app.post("/predict")
async def predict(input: InputSchema):
    try:
        with torch.inference_mode():
            result = model(input.to_tensor())
        return {"prediction": result.tolist()}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(503, "GPU OOM, retry")
    except RuntimeError as e:
        logger.error(f"Inference failed: {e}")
        return {"prediction": None, "error": "inference_failed"}  # Fallback response
```

### Exception Propagation (TorchElastic)
```python
from torch.distributed.elastic.multiprocessing.errors import record

@record  # Writes uncaught exceptions to file for distributed debugging
def inference_worker():
    # ... inference logic ...
```

## 7. Multi-Worker Deployment

### Recommended Setup: Gunicorn + Uvicorn Workers
```bash
gunicorn app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --preload \
  --timeout 120 \
  --bind 0.0.0.0:8000
```

### Architecture Patterns

**Pattern 1: Single Worker per GPU**
```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 uvicorn app:app --port 8000 --workers 1

# GPU 1
CUDA_VISIBLE_DEVICES=1 uvicorn app:app --port 8001 --workers 1

# Load balancer distributes across ports
```

**Pattern 2: Async CPU Workers + GPU Queue**
```
[FastAPI CPU Workers] → [RabbitMQ] → [GPU Inference Worker] → [Redis Cache]
```
- FastAPI: 8+ async workers (request validation, response formatting)
- GPU Worker: 1 worker per GPU (batch inference)
- RabbitMQ: Message broker
- Redis: Result cache

### Threading Considerations
```python
# Limit PyTorch threads per worker
torch.set_num_threads(4)  # Prevents oversubscription with multiple workers
```

## Key Takeaways

1. **Lifespan events** for startup model loading (modern approach)
2. **Gunicorn preload** for memory-efficient multi-worker (1× model RAM vs N×)
3. **torch.compile(mode="max-autotune")** for 2024+ inference optimization
4. **Dynamic batching** via async queue for throughput (32 batch, 100ms timeout)
5. **Warm-up inference** in startup to trigger compilation, allocate GPU pools
6. **Inference mode** (not no_grad) for production inference
7. **Single worker per GPU** or **async CPU workers + GPU queue** for multi-GPU
8. **torch.set_num_threads(4)** to stabilize multi-worker performance

## Unresolved Questions

- Optimal batch size/latency trade-offs for specific model architectures?
- torch.compile compatibility with custom CUDA kernels?
- Multi-GPU model parallelism patterns for FastAPI (tensor/pipeline parallelism)?

## Sources

- [Lifespan Events - FastAPI](https://fastapi.tiangolo.com/advanced/events/)
- [Deploying Large Deep Learning Models in Production](https://www.streppone.it/cosimo/blog/2021/08/deploying-large-deep-learning-models-in-production/)
- [Maximizing PyTorch Throughput with FastAPI](https://jonathanc.net/blog/maximizing_pytorch_throughput)
- [Understanding CUDA Memory Usage - PyTorch](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)
- [Server Workers - Uvicorn with Workers - FastAPI](https://fastapi.tiangolo.com/deployment/server-workers/)
