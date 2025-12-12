# the following fixes is what I made in training enviroment
### 🔧 models/yolo.py — Fix `_initialize_biases()`

**File:** `models/yolo.py`
**Function:** `_initialize_biases(self, cf=None)`

```python
# avoid leaf in-place operation on torch variable view
b = mi.bias.view(m.na, -1).clone()   # (na, no)
```

### 🔧 numpy dtype fixes (global)

```bash
find . -name "*.py" -exec sed -i 's/np.int/int/g' {} +
find . -name "*.py" -exec sed -i 's/dtype=int16/dtype=np.int16/g' {} +
```

### 🔧 utils/general.py — Fix `build_targets()`

```python
gj = gj.clamp(0, int(gain[3].item() - 1e-3))
gi = gi.clamp(0, int(gain[2].item() - 1e-3))
indices.append((b, a, gj, gi))
```

### 🔧 utils/general.py — Fix `output_to_target()`

```python
if isinstance(targets, list):
    if len(targets):
        targets = torch.stack(targets, 0)
    else:
        return np.zeros((0, 6), dtype=np.float32)
elif isinstance(targets, torch.Tensor):
    if targets.numel() == 0:
        return np.zeros((0, 6), dtype=np.float32)
else:
    targets = torch.as_tensor(targets)
return targets.detach().cpu().numpy()
```

### 🔧 utils/datasets.py — Fix cache loading (PyTorch 2.x)

```bash
sed -i "s/torch.load(cache_path)/torch.load(cache_path, weights_only=False)/" utils/datasets.py
rm -f ../train/labels.cache ../valid/labels.cache
```

### 🔧 train.py — Fix `interp` → `np.interp`

```bash
sed -i 's/interp(/np.interp(/g' train.py
```

### 🔧 Disable plotting in test.py

```python
#plot_images(...)
```
