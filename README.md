# FedQClip Implementation Audit

## 1. Scope

This README documents the **actual executable implementation** of `METHOD_NAME = FedQClip` in this repository.

### Audited execution path

The active path is the single-script training pipeline in `FedQClip.py`:

1. Parse args via `get_config()`.
2. Build datasets/clients via `get_dataset(args)`.
3. Build model via `build_model(...)`.
4. Run FL rounds in the main training loop.
5. Log metrics to WandB via `wandb.log(...)`.

### Files inspected

| File | Why it matters |
|---|---|
| `FedQClip.py` | Main training loop, local training, payload processing, aggregation, FLOPs/traffic/accuracy logging. |
| `config.py` | Runtime flags/defaults controlling activation and variants. |
| `data_utils.py` | Train/validation split and Dirichlet client partitioning. |
| `ResNet18.py`, `effnet.py` | Model classes used by `build_model(...)`. |

### Non-goals

- This audit does **not** evaluate paper-level correctness directly.
- This audit describes what code does now, including mismatches or ambiguities.

---

## 2. High-Level Verdict

**Overall verdict: PARTIAL faithfulness to a “FedQClip” structure.**

What is clearly implemented in the active path:
- Federated rounds with random client subsampling.
- Local client training with a clipped gradient-step rule.
- Client delta construction (`global - local`) and server aggregation over deltas.
- Optional client-side payload quantization + serialization + server-side deserialization + dequantization when `--quantize=True`.
- WandB logging of traffic, FLOPs counters, and accuracy ranges.

Main caveats:
- `--method` exists but does not gate pipeline selection.
- Compression FLOPs counters do not include all possible compression/decompression stages (only upload-side quantize + server dequantize).
- `acc_servers_highest` is not a historical best metric; it is mean+std over a one-element list.
- FLOPs are parameter-count proxies, not hardware-measured operation counts.

---

## 3. Actual Execution Pipeline

End-to-end active pipeline in `FedQClip.py`:

1. **Configuration and tracking**
   - `args = get_config()`.
   - `wandb.init(project="compression_FL", config=...)`.
2. **Environment setup**
   - Seed control via `set_seed(args.seed)`.
   - Device chosen by CUDA availability check.
3. **Data setup**
   - `client_datasets, val_dataset, n_classes, _, _ = get_dataset(args)`.
   - `get_dataset` performs global random split with `args.train_frac` and non-IID Dirichlet split across clients.
4. **Model setup**
   - `global_model = build_model(model_name, n_classes, dataset_name)`.
5. **Baseline validation**
   - `validate_model(global_model, val_loader, ...)` logged at round 0.
6. **Per-round FL loop**
   - Sample active clients: `random.sample(range(num_clients), num_participants)`.
   - For each selected client:
     - clone global model,
     - run `train_client(...)`,
     - compute `update = global_state - updated_state_dict`,
     - if quantization enabled: quantize -> serialize -> deserialize -> dequantize,
     - append reconstructed update and add into `aggregated_updates`.
   - Server update via `aggregate_models(...)`.
   - Validate new global model via `validate_model(...)`.
   - Compute/log WandB report and append trainloss text row.

---

## 4. Method Activation and Required Flags

### 4.1 Activation mechanism

| Item | Status | Evidence |
|---|---|---|
| Dedicated method router using `--method` | **Implemented but not used** | `config.py` defines `--method`, but `FedQClip.py` never branches on it. |
| FedQClip logic activation | **Always active when `FedQClip.py` is run** | Single hardcoded pipeline in script body. |

### 4.2 Flags that actually change runtime behavior

| Flag | Default | Used in active path? | Effect |
|---|---:|---|---|
| `--n_client` | 10 | Yes | Total client population and sampling range. |
| `--client_fraction` | 0.5 | Yes | Active clients per round (`num_participants`). |
| `--n_epoch` | 100 | Yes | Number of rounds. |
| `--n_client_epoch` | 5 | Yes | Local epochs per selected client. |
| `--lr` | 0.01 | Yes | Both `eta_c` and `eta_s`. |
| `--gamma_c` | 10 | Yes | Client gradient-step clipping factor. |
| `--gamma_s` | 1e6 | Yes | Server global-step clipping factor. |
| `--quantize` | True | Yes | Enables upload quantization/serialization path. |
| `--bit` | 8 | Yes | Quantization bit-width for upload payload entries. |
| `--dirichlet` | 0.5 | Yes | Dirichlet alpha in data split. |
| `--train_frac` | 0.8 | Yes | Train/validation split ratio. |
| `--batch_size` | 128 | Yes | Client and validation batch size. |
| `--model` | resnet | Yes | Selects `ResNet18` or `EfficientNetB0_CIFAR`. |
| `--dataset` | cifar10 | Yes | Dataset and transform path. |
| `--device` | cuda | **No** | Not used for device selection in script. |

### 4.3 Method-specific vs generic vs unused

- **Method-specific (FedQClip-labeled behavior):** quantization/serialization/dequantization functions and clipped client/server step sizes.
- **Generic FL pipeline:** client sampling, local training loops, delta accumulation, server aggregation, validation, logging.
- **Exists but not used:** `Quantizer` class is defined but never invoked in active loop.

---

## 5. Client-Side Processing After Local Training

### 5.1 Immediate post-training object

`train_client(...)` returns `model.state_dict()` (`updated_state_dict`) plus scalar diagnostics.

### 5.2 Transformations before server consumes update

| Processing step | Status in active pipeline | Exact behavior |
|---|---|---|
| Delta computation | **Implemented and used** | `update[name] = global_state[name] - updated_state_dict[name]`. |
| Clipping | **Partially (during optimizer-like step, not post-hoc payload clipping)** | Gradient step size clipped inside `train_client`. |
| Normalization of payload | **Missing** | No dedicated normalization transform on `update`. |
| Sparsification/masking | **Missing** | No sparse mask or threshold step. |
| Quantization | **Implemented and used when `--quantize=True`** | `quantize_client_payload(update, bit)`. |
| Serialization/packing | **Implemented and used** | `serialize_client_payload(...)` returns bytes. |
| Compression codec beyond quantization | **Missing** | No entropy coding/compressed sparse format. |
| Low-rank decomposition | **Missing** | No rank factorization stage. |

### 5.3 Active object sequence (quantized path)

`updated_state_dict` (full weights)
→ `update` (dense delta dict)
→ `quantized_payload` (dict of shape/min/max/bit/values)
→ `serialized_payload` (`bytes` packet)
→ server deserialized packet
→ `reconstructed_update` (float tensor dict used for aggregation).

---

## 6. Client-to-Server Payload and Transmission Logic

### 6.1 What is transmitted in practice?

Even without sockets, the code simulates transmission through explicit packetization:

- **Upload payload object before serialization:** `update` (`dict[str, torch.Tensor]`), dense model deltas.
- **Transmitted representation:** `serialized_payload` (`bytes`) produced by `serialize_client_payload(...)`.
- **Server receive representation:** output of `deserialize_client_payload(serialized_payload)`.
- **Server aggregation input:** `reconstructed_update` (`dict[str, torch.FloatTensor]`).

### 6.2 Handoff location

Per-client inside round loop in `FedQClip.py`:
1. Build `update`.
2. Serialize payload.
3. Deserialize on server side immediately (same process).
4. Add reconstructed tensors into `aggregated_updates`.

### 6.3 Communication realism classification

| Property | Status |
|---|---|
| In-memory only | Yes (single process). |
| Explicit serialization before handoff | Yes (`bytes` packet). |
| Per-client packet length available | Yes (`len(serialized_payload)`). |
| Real network stack | No. |

---

## 7. Upload Traffic Validation

### 7.1 Does `upload_traffic_per_client` correspond to real payload?

**Yes (for this implementation).**

`upload_traffic` is accumulated from `len(serialized_payload)` for each active client, i.e., the exact byte length of the constructed upload packet in the active code path.

### 7.2 Formula validation

The code sets:
- `upload_traffic = round_upload_traffic`
- `upload_traffic_per_client = upload_traffic / num_participants`

Hence:
- `upload_traffic == upload_traffic_per_client * number_of_active_users` (up to float formatting).
- Multiplier uses **active users** (`num_participants`), not total users.

### 7.3 Estimated vs measured

- It is **measured from simulated serialized bytes**, not from OS/network transport capture.
- It is still meaningful for payload-size accounting because the packet is concretely materialized.

---

## 8. Server-Side Reconstruction / Decoding

| Operation | Status | Where |
|---|---|---|
| Deserialization | **Implemented and used** | `deserialize_client_payload(...)`. |
| Dequantization | **Implemented and used when quantized** | `dequantize_client_payload(...)`. |
| Decompression beyond dequantization | **Missing** | No additional codec stage. |
| Sparse reconstruction | **Missing** | No sparse representation to reconstruct. |
| Low-rank reconstruction | **Missing** | No low-rank factors in payload. |
| Delta reconstruction | **Not needed** | Clients already send deltas. |

Result: server always aggregates float tensor dicts (`reconstructed_update`), reconstructed from serialized packets.

---

## 9. Global Aggregation / Global Update Logic

### 9.1 Actual server update rule

- Aggregate by sum of reconstructed client deltas into `aggregated_updates`.
- Compute norm proxy from averaged delta.
- Compute clipped server step:
  `global_step_size = min(eta_s, (gamma_s * eta_s) / (param_diffs_norm / (num_participants * num_epochs_per_round)))`.
- Apply update directly to global state dict:
  `global_dict[name] = global_dict[name] - global_step_size * (aggregated_updates[name] / (num_participants * eta_c))`.

### 9.2 Classification

| Candidate behavior | Status |
|---|---|
| Aggregate client deltas | Implemented and used |
| Aggregate full weights | Not used |
| Reconstruct before aggregation | Implemented (deserialize/dequantize path) |
| Server optimizer object | Not used |
| Server-side gradient training | Not used |
| Direct overwrite/load of global state | Implemented (`load_state_dict`) |

---

## 10. Server-to-Client Payload and Download Logic

### 10.1 Outbound payload representation

- The model distributed to clients is represented by `global_state` (state dict snapshot) and serialized as:
  `global_model_packet = serialize_client_payload(global_state, quantized=False)`.
- `download_traffic_per_client = len(global_model_packet)`.
- `download_traffic = download_traffic_per_client * num_participants`.

### 10.2 Processing before “download”

| Processing type | Status in active path |
|---|---|
| Compression | Missing |
| Quantization | Missing |
| Sparsification/masking | Missing |
| Low-rank processing | Missing |
| Serialization | Implemented and used |

So download accounting is based on **serialized float32 full-model payload**.

---

## 11. Download Traffic and Overall Traffic Validation

### 11.1 Download traffic fidelity

- `download_traffic` is computed from actual serialized server-to-client payload bytes (`len(global_model_packet)`) times active clients.
- This is simulated (in-process) but directly tied to constructed payload format.

### 11.2 Overall traffic equation

The code accumulates:
- `total_upload_traffic += upload_traffic`
- `total_download_traffic += download_traffic`
- `overall_traffic = total_upload_traffic + total_download_traffic`

So `overall_traffic = upload + download` is explicitly enforced on cumulative totals.

---

## 12. FLOPs Logging Validation

### 12.1 Required keys

| Key | Present? | Meaning in implementation |
|---|---|---|
| `round_flops` | Yes | Per-round local training proxy FLOPs + server validation proxy FLOPs. |
| `total_flops` | Yes | Cumulative sum of `round_flops` plus round-0 validation FLOPs baseline. |

### 12.2 What is included in `round_flops`/`total_flops`

| Component | Included? | Notes |
|---|---|---|
| Local training FLOPs | Yes | Proxy: `forward_backward` per sample from parameter count. |
| Server update/aggregation FLOPs | No | Not added to counters. |
| Compression/decompression FLOPs | No (separate fields) | Tracked in `round_flops_compression` / `round_flops_decompression`. |
| Evaluation FLOPs | Yes | Validation pass proxy included each round and at round 0. |
| Communication FLOPs | No | No comm-kernel FLOPs counted. |

### 12.3 Correctness caveat

Names imply FLOPs, but values are **heuristic proxies** based on model parameter count, not operation-level profiled FLOPs.

---

## 13. Compression / Decompression FLOPs Validation

### 13.1 `total_flops_compression` coverage

`total_flops_compression` accumulates only `client_compression_flops` returned by `quantize_client_payload` during upload path.

### 13.2 `total_flops_decompression` coverage

`total_flops_decompression` accumulates only `client_decompression_flops` returned by `dequantize_client_payload` on server receive path.

### 13.3 Missing components relative to broad label

| Stage | Counted? |
|---|---|
| Client-side upload quantization | Yes |
| Server-side upload dequantization | Yes |
| Server-side compression for download | No (not implemented) |
| Client-side download decompression | No (not implemented) |
| Serialization/deserialization overhead FLOPs | No |

Conclusion: metric names are acceptable for current implemented stages, but they do **not** represent a full bidirectional compression/decompression lifecycle.

---

## 14. Accuracy Logging Validation

### 14.1 `acc_servers_highest`

- Variable is logged every round in `report`.
- It is computed from:
  - `acc_servers = [acc.item()]`
  - `acc_servers_mean = np.mean(acc_servers)`
  - `acc_servers_std = np.std(acc_servers)`
  - `acc_servers_highest = acc_servers_mean + acc_servers_std`

Given one-element list, std = 0, so `acc_servers_highest == acc.item()` each round.

### 14.2 Interpretation

| Question | Answer |
|---|---|
| Is it updated? | Yes, every round. |
| Dataset/split used | `val_loader` built from `val_dataset` returned by `get_dataset` (the holdout split from `train_frac`). |
| Is it historical best global accuracy? | **No.** |
| Does the name match behavior? | **Partially/poorly.** It indicates “highest” but actually equals current-round server accuracy. |

---

## 15. Experiment Configuration Validation

Requested standard setup validation:

| Item | Requested | Repository reality | Status |
|---|---|---|---|
| Dirichlet alpha | 0.5 | `--dirichlet` default is `0.5`; used in `split_noniid(...)`. | **Default matches; configurable** |
| Train/validation split | 80/20 | `--train_frac` default `0.8`; split via `int(len(data)*args.train_frac)`. | **Default matches; configurable** |
| Optimizer | SGD | No `torch.optim` optimizer object used; manual gradient-step update implemented. | **Different implementation** |
| Learning rate | 0.01 | `--lr` default `0.01`; used as both `eta_c` and `eta_s`. | **Default matches; configurable** |
| Batch size | 128 | `--batch_size` default `128`; used in both train and validation loaders. | **Default matches; configurable** |

Important note: “optimizer = SGD” cannot be strictly confirmed because explicit optimizer class is absent; behavior is SGD-like manual update with clipping.

---

## 16. WandB Metrics Audit

### 16.1 Metrics logged at round 0 (baseline)

- `round`
- `acc_servers`
- `acc_servers_lowest`
- `acc_servers_highest`
- `round_flops`
- `total_flops`
- `total_flops_compression`
- `total_flops_decompression`
- `total_flops_including_compression`

### 16.2 Metrics logged each training round

| Category | Metrics |
|---|---|
| Similarity | `cos_lowest`, `cos_highest` |
| Loss/accuracy bands | `training_loss_lowest`, `training_loss_highest`, `acc_clients_lowest`, `acc_clients_highest`, `acc_servers_lowest`, `acc_servers_highest` |
| Participation | `num_active_clients` |
| Upload traffic | `upload_traffic_per_client`, `upload_traffic_per_client_min`, `upload_traffic_per_client_max`, `upload_traffic` |
| Download traffic | `download_traffic_per_client`, `download_traffic`, `round_total_traffic`, `overall_traffic` |
| Sparsity diagnostics | `upload_sparsity_mean`, `download_sparsity_mean` |
| FLOPs | `round_flops`, `total_flops`, `round_flops_compression`, `round_flops_decompression`, `total_flops_compression`, `total_flops_decompression`, `total_flops_including_compression` |
| Index | `round` |

### 16.3 Flag-to-WandB linkage

- `--quantize` / `--bit` affect upload payload format and therefore upload traffic + compression FLOPs metrics.
- `--client_fraction` affects `num_active_clients` and scales round traffic/FLOPs.
- `--n_client_epoch`, `--batch_size`, `--model` affect training/eval FLOPs and accuracy trends.
- `--method` is logged in WandB config but does not alter execution path.

---

## 17. Faithfulness to the Intended Method Structure

Comparison is based on method name/repository structure only (not external paper text).

| Expected stage for a FedQClip-like method | Observed implementation | Classification |
|---|---|---|
| Federated round-based training | Implemented with client subsampling and server updates | **Correctly implemented** |
| Client local optimization with clipping | Implemented via per-parameter clipped step in `train_client` | **Correctly implemented** |
| Client update payload generation | Dense delta dict generated (`global - local`) | **Correctly implemented** |
| Client-side quantized upload | Implemented when `--quantize=True` | **Correctly implemented (optional)** |
| Payload serialization for communication | Implemented via custom binary packet | **Correctly implemented** |
| Server-side decode + dequantize | Implemented before aggregation | **Correctly implemented** |
| Aggregation/update with clipped server step | Implemented in `aggregate_models` | **Correctly implemented** |
| Download-side compression pipeline | Not implemented | **Missing** |
| Historical-best server accuracy metric | Not implemented (`acc_servers_highest` is current round) | **Implemented differently** |
| Method selector routing via `--method` | Not implemented | **Missing** |

---

## 18. Mismatches, Risks, and Ambiguities

1. **`--method` ambiguity:** flag suggests multi-method framework, but script ignores it.
2. **`acc_servers_highest` naming risk:** name implies running maximum but actual computation is single-round value.
3. **FLOPs interpretation risk:** counter names imply precise FLOPs, but implementation uses parameter-count proxies.
4. **Compression scope mismatch risk:** only upload quantize/dequantize path is counted; download compression is absent.
5. **`--device` ambiguity:** config includes it, but execution path ignores it.
6. **Serialization realism:** packet size is concrete, but no real transport overhead (protocol/socket) is modeled.

If stricter paper-faithfulness validation is needed later, these are priority checks against the original method description.

---

## 19. Final Checklist

| Validation item | Result | Notes |
|---|---|---|
| 1) Method activation and pipeline identified | **PASS** | Single active pipeline in `FedQClip.py`; `--method` not used for routing. |
| 2) Post-local-training preprocessing verified | **PASS** | Delta + optional quantize/serialize path documented. |
| 3) `round_flops` / `total_flops` in WandB verified | **PASS** | Present and logged each round (plus baseline). |
| 4) Client-side compression logic audited | **PASS** | Quantization+serialization implemented/used; sparsity/low-rank missing. |
| 5) Flags/config and WandB linkage validated | **PASS** | Key flags mapped; inactive flags noted. |
| 6) Client-to-server transmission process defined | **PASS** | Explicit bytes packet in-memory handoff documented. |
| 7) Upload traffic validation (`per_client * active_users`) | **PASS** | Uses active users (`num_participants`). |
| 8) Server-side decoding/reconstruction validated | **PASS** | Deserialization + optional dequantization used. |
| 9) Global/server update logic validated | **PASS** | Delta aggregation + clipped direct update verified. |
| 10) Server-to-client processing validated | **PASS** | Serialization only; no compression/quantization. |
| 11) Download/overall traffic validated | **PASS** | Derived from serialized global payload and cumulative sum equation. |
| 12) Compression/decompression FLOPs audited | **PARTIAL** | Covers upload quantize + server dequantize only. |
| 13) `acc_servers_highest` semantics validated | **PARTIAL** | Logged, but not historical best; naming mismatch. |
| 14) Standard experiment config validated | **PARTIAL** | Most defaults match; optimizer differs (manual SGD-like). |
| 15) Intended-vs-actual method structure comparison | **PASS** | Stage-by-stage classification provided. |

