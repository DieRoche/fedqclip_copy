# FedQClip Implementation Audit

## 1. Scope

This document audits the **actual executable path** for `METHOD_NAME = FedQClip` in this repository, focusing on what the code really does at runtime when running `FedQClip.py`.

### Files inspected

| File | Role in active path | Notes |
|---|---|---|
| `FedQClip.py` | Main end-to-end FL training loop, client updates, server aggregation, traffic/FLOPs/WandB logging | Primary implementation target. |
| `config.py` | CLI argument definitions and defaults | Controls runtime behavior via `get_config()`. |
| `data_utils.py` | Dataset loading/splitting and client partitioning | Determines train/validation split and Dirichlet partitioning. |
| `ResNet18.py` / `effnet.py` | Model definitions selected via `--model` | Used by `build_model()`. |

Out of scope: theoretical claims not represented in executable code.

---

## 2. High-Level Verdict

**Verdict (code-faithfulness to a “FedQClip-like” name): PARTIAL.**

What is clearly implemented:
- Federated round loop with sampled clients and server aggregation.
- Per-client local training with a clipped step-size rule (`min(eta_c, gamma_c*eta_c/||grad||)`).
- Server update using aggregated client deltas and another clipped step-size rule.
- Optional quantization logic (`--quantize`, `--bit`) applied to the **aggregated server update tensor dict**, not per-client payload.
- WandB logging for accuracy bands, traffic estimates, and FLOPs-like counters.

Key mismatches/risks:
- `--method` exists but is **not used** to route logic.
- Quantization is not applied to each client payload before upload; upload traffic is still computed as if payload bit-width changed.
- Download traffic multiplies model size by fixed `10`, not by active participants.
- `acc_servers_highest` is not a historical best metric; it is round mean + std over a single server value.
- FLOPs values are parameter-count-based heuristics, not measured kernel FLOPs.

---

## 3. Actual Execution Pipeline

1. Parse CLI args via `get_config()`, initialize WandB with full arg config.
2. Set deterministic seeds and device.
3. Load and split dataset in `get_dataset(args)`:
   - random global split using `train_frac` (default 0.8),
   - Dirichlet non-IID split of train indices across clients.
4. Build global model from `--model` (`resnet` or `effnet`).
5. Estimate per-sample FLOPs proxy from parameter count.
6. Validate initial global model (round 0) on `val_loader` and log baseline metrics.
7. For each communication round:
   - sample active clients by `client_fraction`,
   - deepcopy global model per client,
   - local train each client model with clipped per-parameter gradient step,
   - compute client update as `global_state - updated_state_dict`,
   - sum into `aggregated_updates`.
8. Server aggregation:
   - optionally quantize/dequantize `aggregated_updates` tensor dict,
   - compute global clipped step size,
   - update global weights directly.
9. Validate updated global model.
10. Compute/log WandB metrics including traffic, FLOPs counters, and accuracy ranges.
11. Append selected scalar stats to `trainloss_<model>.txt`.

---

## 4. Method Activation and Required Flags

### Activation requirements

| Item | Status | Evidence |
|---|---|---|
| Dedicated method router based on `--method` | **Missing** | `--method` is defined but never referenced after parsing. |
| Running FedQClip logic | **Always active** when `FedQClip.py` is executed | Script contains a single hard-coded pipeline. |
| Quantization variant | **Optional** via `--quantize` (default `True`) and `--bit` (default `8`) | Used only in `aggregate_models(...)`. |

### Flags that materially affect behavior

| Flag | Default | Actual use |
|---|---:|---|
| `--n_client` | `10` | Total client pool and sampling range. |
| `--client_fraction` | `0.5` | Active clients each round: `max(1, int(n_client*client_fraction))`. |
| `--n_epoch` | `100` | Number of FL rounds. |
| `--n_client_epoch` | `5` | Local epochs per selected client per round. |
| `--lr` | `0.01` | Both `eta_c` and `eta_s`. |
| `--gamma_c` | `10` | Client step clipping factor. |
| `--gamma_s` | `1e6` | Server step clipping factor. |
| `--quantize` | `True` | Enables quantization of aggregated server-side update. |
| `--bit` | `8` | Bit-width for quantization and upload traffic estimation. |
| `--dirichlet` | `0.5` | Non-IID split concentration. |
| `--train_frac` | `0.8` | train/validation split ratio. |
| `--batch_size` | `128` | Client and validation loaders. |
| `--model` | `resnet` | Chooses architecture via `build_model`. |
| `--dataset` | `cifar10` | Dataset and transforms in `get_dataset`. |

### Code present but not used in active FedQClip path

- Imports unused in `FedQClip.py`: `optim`, `transforms`, `random_split`, `datasets`, `models`, `init`, `defaultdict`.
- `alpha = args.dirichlet` assignment in `FedQClip.py` is not later used directly.
- `args.device` exists in config but runtime device selection ignores it (uses CUDA availability test directly).

---

## 5. Client-Side Processing After Local Training

### Object immediately after local training

`train_client(...)` returns `model.state_dict()` as `updated_state_dict` (full local model weights), plus norms/loss/acc/FLOPs proxy.

### Transformations before server uses client result

| Stage | Implemented in active path? | Details |
|---|---|---|
| Delta computation | **Yes** | `update = {name: global_state[name] - updated_state_dict[name] ...}`. |
| Clipping/normalization of upload payload | **No explicit post-train transform on payload** | Clipping happens during local optimization step, not on final delta object. |
| Sparsification/masking/pruning | **No** | No zeroing mask construction before upload. |
| Client-side quantization/compression | **No** | No quantizer call on per-client `update` before adding to list. |
| Serialization/packing | **No** | Python dicts of tensors kept in memory. |
| Low-rank decomposition | **No** | Not implemented. |

Important nuance: even when `--quantize=True`, quantization occurs **after aggregation on server input dict**, not on each client upload object.

---

## 6. Client-to-Server Payload and Transmission Logic

### Practical “transmission” implementation

- There is no network transport layer.
- Handoff is in-memory within one process and one script loop.

### Actual uploaded payload object

| Aspect | Actual implementation |
|---|---|
| Payload type | `dict[str, torch.Tensor]` |
| Payload variable | `update` |
| Construction | Difference of global and client full weights (`global_state - updated_state_dict`) |
| Collection | Appended to `participating_updates` list |
| Handoff location | Round loop, before `aggregate_models(...)` call |
| Serialization before handoff | None |

So the effective upload representation is **dense full-model delta tensors per participating client** (in-memory).

---

## 7. Upload Traffic Validation

### Is `upload_traffic_per_client` based on actual payload type?

**Partially.**
- Payload type is per-client dense delta dict (correct object family).
- But byte estimate is derived from element count and assumed bit-width (`bit` if quantize else 32), not actual serialized message size.
- Since client payload is not actually quantized, using `bit` underestimates true in-memory tensor footprint when `quantize=True`.

### Formula check

Implemented:
- `upload_traffic = sum(tensor_dict_bytes(update, bit=upload_bit) for update in participating_updates)`
- `upload_traffic_per_client = upload_traffic / num_participants`

Therefore algebraically: `upload_traffic == upload_traffic_per_client * num_participants` (active users), up to floating conversion. This uses active participants, not total clients.

Conclusion: **Formula with active users is correct, but value is an estimate with potentially inconsistent bit assumption versus actual payload representation.**

---

## 8. Server-Side Reconstruction / Decoding

| Operation | Active? | Where / how |
|---|---|---|
| Deserialization | **No** | No serialized input exists. |
| Decompression | **No general codec** | None beyond quantizer math. |
| Dequantization | **Yes (inside Quantizer call)** | `x_qu -= b; x_qu /= k` performed immediately after rounding, returning float tensor. |
| Sparse reconstruction | **No** | No sparse payload type is used. |
| Low-rank reconstruction | **No** | Not implemented. |
| Delta reconstruction | **Not needed** | Clients already provide deltas directly (`update`). |

Because quantization/dequantization is done in-place in `aggregate_models` on already aggregated tensors, server never reconstructs individual client compressed payloads.

---

## 9. Global Aggregation / Global Update Logic

### Server update rule actually used

1. Sum client deltas into `aggregated_updates`.
2. Optionally quantize/dequantize `aggregated_updates`.
3. Compute norm of average delta and clipped global step size.
4. Update each global parameter by subtracting scaled aggregated delta:
   - `global = global - global_step_size * (aggregated_updates / (num_participants * eta_c))`

### Classification

| Behavior | Status |
|---|---|
| Aggregates client deltas | **Yes** |
| Aggregates full client weights directly | **No** |
| Reconstructs compressed client payloads before aggregation | **No** |
| Uses server optimizer object (e.g., SGD/Adam) | **No** |
| Performs server “training” beyond aggregation math | **No** (aggregation/update logic only) |
| Overwrites global model weights directly | **Yes** (`load_state_dict`) |

---

## 10. Server-to-Client Payload and Download Logic

### Actual outbound object

- The object conceptually sent back is `global_state` (deep copy of current global model state dict). Traffic calculation uses that shape.
- No explicit send API; clients next round copy from `global_model` via `copy.deepcopy(global_model)`.
- No compression/quantization/sparsification/serialization is applied to server-to-client model in active path.

### Helper vs active

- `tensor_dict_bytes` is only an estimator helper; no real encoded payload is materialized.

---

## 11. Download Traffic and Overall Traffic Validation

### Implemented formulas

- `download_traffic = tensor_dict_bytes(global_state, bit=32) * 10`
- `overall_traffic = total_upload_traffic + total_download_traffic`

### Validation findings

| Check | Result | Reason |
|---|---|---|
| Download based on actual processed outbound payload | **PARTIAL** | Uses model tensor size proxy, but assumes fixed 32-bit and fixed multiplier 10. |
| Uses active users for download multiplier | **FAIL** | Hard-coded `* 10` ignores `num_participants`; only coincides if active users = 10. |
| Overall traffic equals upload + download accumulators | **PASS** | Computed exactly as cumulative sum. |
| Includes serialization overhead | **No** | Tensor-element estimate only. |

So download and overall traffic are **simulated estimates**, not measured transport bytes.

---

## 12. FLOPs Logging Validation

### WandB keys present

- `round_flops`
- `total_flops`
- `round_flops_compression`
- `round_flops_decompression`
- `total_flops_compression`
- `total_flops_decompression`
- `total_flops_including_compression`

### What `round_flops` includes

- Sum of per-client local training proxy FLOPs (`forward_backward` proxy × samples).
- Plus server validation proxy FLOPs (`forward` proxy × validation samples).
- **Excludes** server aggregation arithmetic, quantization/dequantization FLOPs (tracked separately), and communication overhead FLOPs.

### What `total_flops` includes

- Initial validation FLOPs at round 0.
- Cumulative sum of each round’s `round_flops` thereafter.
- Still excludes compression/decompression unless separately combined in `total_flops_including_compression`.

### Correctness note

These are **parameter-count heuristics** (`forward = #params`, `forward_backward = 2*#params`) rather than operator-accurate FLOP measurements.

---

## 13. Compression / Decompression FLOPs Validation

### Is `total_flops_compression` comprehensive?

**No.** It only accumulates Quantizer-estimated ops in `aggregate_models` when quantization is enabled.

Included:
- “compression” ops for aggregated tensor quantization.
- “decompression” ops for immediate dequantization of same aggregated tensors.

Missing:
- Client-side compression FLOPs (none implemented in active path).
- Server-side decompression of per-client payloads (not applicable, none received).
- Server-to-client compression and client-side decompression FLOPs (not implemented).

Therefore metric name suggests broader coverage than actual counted operations.

---

## 14. Accuracy Logging Validation

### `acc_servers_highest` behavior

- Round 0: set equal to initial validation accuracy.
- Later rounds: computed as `acc_servers_mean + acc_servers_std` where `acc_servers = [acc.item()]` (single value), so std is zero and metric equals current round accuracy.

Implications:
- It is **not** historical-best accuracy.
- It uses the same validation split (`val_loader`) as server accuracy evaluation.
- Name is misleading if interpreted as “best so far”.

---

## 15. Experiment Configuration Validation

Validation target vs actual implementation:

| Item | Requested “standard” | Observed in code | Status |
|---|---:|---|---|
| Dirichlet alpha | 0.5 | Default `--dirichlet=0.5`; used in non-IID split | **Default matches, configurable** |
| Train/validation split | 80/20 | Default `--train_frac=0.8`; split done once by permutation | **Default matches, configurable** |
| Optimizer | SGD | No `torch.optim.SGD` object used in active path; manual gradient step update | **Different implementation** |
| Learning rate | 0.01 | Default `--lr=0.01`; used for both client and server step bases | **Default matches, configurable** |
| Batch size | 128 | Default `--batch_size=128`; used in DataLoaders | **Default matches, configurable** |

Important: because optimizer object is absent, “optimizer=SGD” cannot be claimed as enforced despite SGD-like manual update logic.

---

## 16. WandB Metrics Audit

### Metrics logged at round 0

| Key |
|---|
| `round` |
| `acc_servers` |
| `acc_servers_lowest` |
| `acc_servers_highest` |
| `round_flops` |
| `total_flops` |
| `total_flops_compression` |
| `total_flops_decompression` |
| `total_flops_including_compression` |

### Metrics logged each training round

| Category | Keys |
|---|---|
| Similarity | `cos_lowest`, `cos_highest` |
| Loss | `training_loss_lowest`, `training_loss_highest` |
| Client accuracy | `acc_clients_lowest`, `acc_clients_highest` |
| Server accuracy | `acc_servers_lowest`, `acc_servers_highest` |
| Traffic | `upload_traffic_per_client`, `upload_traffic`, `download_traffic`, `overall_traffic` |
| Sparsity | `upload_sparsity_mean`, `download_sparsity_mean` |
| FLOPs | `round_flops`, `total_flops`, `round_flops_compression`, `round_flops_decompression`, `total_flops_compression`, `total_flops_decompression`, `total_flops_including_compression` |
| Control | `round` |

### Missing/ambiguous wandb linkages

- No explicit metric logging for whether client-side quantization/compression was applied to payload objects.
- No logging of active client count per round.
- No logging of historical-best server accuracy despite `acc_servers_highest` naming.

---

## 17. Faithfulness to the Intended Method Structure

Expected structure implied by repository naming (`FedQClip`) vs implemented stages:

| Stage (conceptual) | Implemented reality | Classification |
|---|---|---|
| Federated rounds with partial participation | Implemented with random client sampling by `client_fraction` | **Correctly implemented** |
| Client local training with clipping | Per-parameter clipped step-size using gradient norm | **Partially implemented** (clipping present, exact algorithmic match to intended method unclear) |
| Client-side quantized upload | Not done; uploads are dense float deltas in-memory | **Missing** |
| Server-side handling of compressed client payloads | No per-client decode path | **Missing** |
| Server update with clipped/global control | Implemented via clipped global step | **Implemented differently/partially** |
| Communication-aware accounting aligned to actual payload | Estimated via tensor sizes and assumptions | **Partially implemented** |
| Consistent “best server accuracy” tracking | Current-round value only under misleading key name | **Implemented differently** |

Because the repository does not include a formal stage-by-stage method spec, exact equivalence to intended algorithmic details is **partly unclear**.

---

## 18. Mismatches, Risks, and Ambiguities

1. **Method switch ambiguity**: `--method` is non-functional; running script always executes same pipeline.
2. **Payload/traffic mismatch**: upload traffic assumes quantized bits when `--quantize=True`, but client payload is not quantized.
3. **Download multiplier bug-risk**: hard-coded `* 10` ties traffic to total clients/default, not active participants.
4. **FLOPs naming risk**: variables labeled FLOPs are coarse proxies; may be misread as true compute accounting.
5. **Accuracy naming risk**: `acc_servers_highest` does not represent historical maximum.
6. **Optimizer claim mismatch**: code uses manual state-dict updates, not optimizer API.
7. **Device arg unused**: `--device` does not control runtime device selection.
8. **Ambiguity in intended FedQClip mechanics**: without a linked formal spec in repo, some faithfulness judgments remain structural rather than theorem-level.

---

## 19. Final Checklist

| Validation item | Status | Evidence summary |
|---|---|---|
| 1. Method activation and actual pipeline | **PASS (with caveat)** | Pipeline identified; `--method` unused. |
| 2. Post-local-training preprocessing | **PASS** | Delta creation confirmed; no client-side compression pipeline. |
| 3. FLOPs reporting in wandb (`round_flops`, `total_flops`) | **PARTIAL** | Keys/logging exist; contents are proxies and incomplete for server/comms/compression unless separate keys used. |
| 4. Client-side compression logic | **FAIL** for actual client upload compression | Quantizer exists but not on per-client upload payload. |
| 5. Flags/config and wandb linkage | **PARTIAL** | Main flags linked; some flags unused (`method`, `device` runtime behavior). |
| 6. Client-to-server transmission process | **PASS** | In-memory delta dict payload identified. |
| 7. Upload traffic validation | **PARTIAL** | Active-user formula consistent, but value is estimated and can assume non-real bit-width. |
| 8. Server-side decoding/reconstruction | **PARTIAL** | Only aggregated quantize/dequantize path; no per-client decode pipeline. |
| 9. Global training/server update | **PASS** | Delta aggregation and direct overwrite update logic documented. |
| 10. Server-to-client processing | **PASS (none applied)** | No compression/serialization/masking before conceptual download. |
| 11. Download and overall traffic | **PARTIAL/FAIL for participant scaling** | Overall formula correct; download uses hard-coded `*10`, not active users. |
| 12. Compression/decompression FLOPs | **PARTIAL** | Only aggregated-update quantizer ops counted. |
| 13. Server-side accuracy (`acc_servers_highest`) | **FAIL (semantic naming)** | Not best-so-far; equals current accuracy band point. |
| 14. Standard config validation | **PARTIAL** | Defaults match for alpha/split/lr/batch; optimizer not SGD object. |
| 15. Comparison with intended behavior | **PASS (structural)** | Stage-by-stage classification provided with explicit uncertainty notes. |

