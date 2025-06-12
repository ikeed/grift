# GRIFT Cloud Architecture Plan

This document provides a detailed plan for the GRIFT cloud pipeline architecture using GCP services. Each service is described with its input source, output destination, relevant data schemas, state characteristics, and resumability.

---

## 1. **tick_fetcher_decoupler**
**Type**: GKE Deployment (long-running)

### Input:
- **Source**: REST polling from OANDA FX Tick API
- **Stateful**: ✅ Yes
  - Keeps `last_tick_timestamp` per currency pair in memory
  - Keeps `last_price` per currency pair in memory (latest tick bid/ask or midprice) for decoupling
  - Persists `last_timestamp` and `last_price` to Firestore after each successful block
- **Resumable**: ✅ Yes
  - Resumes from Firestore on startup

### Firestore Payload:
```json
{
  "instrument": "EUR_USD",
  "last_timestamp": "2025-06-09T18:53:00Z",
  "last_price": 1.0742
}
```

### Output:
- **Destination**: Pub/Sub topic `w.latent.raw`

### Output Schema (`w.latent.raw`):
```json
{
  "timestamp": "2025-06-09T18:53:00Z",
  "w": [0.20, -0.14, 0.07, ...]
}
```

### Training/Tuning Plan:
- **Not applicable.** This component is purely infrastructural. No tunable or learnable parameters are planned.

---

## 2. **w-rollup-aggregator**
**Type**: GKE Deployment (long-running)

### Input:
- **Source**: Pub/Sub topic `w.latent.raw`
- **Stateful**: ✅ Yes (ephemeral)
  - Maintains in-memory window buffers using Redis per granularity
- **Resumable**: ❌ No
  - On restart, windows in progress are lost

### Redis Payload (ephemeral example):
```
Key: rollup:5s:20250609T185300Z
Value: {
  "count": 3,
  "sum": [0.60, -0.36, 0.21, ...]
}
```

### Output:
- **Destinations**:
  - Pub/Sub topic `w.latent.5s`
  - Pub/Sub topic `w.latent.15s`

### Output Schema (per topic):
```json
{
  "start_time": "2025-06-09T18:53:00Z",
  "end_time": "2025-06-09T18:53:05Z",
  "aggregated_w": [0.18, -0.12, 0.06, ...]
}
```

### Training/Tuning Plan:
- **Potential tuning:** Window size, aggregation method (e.g., mean vs EWMA).
- **Test metrics:**
  - MAE or RMSE between rollup and raw latent vectors.
  - Forecast accuracy downstream using aggregated data.
- **Note:** Training not currently prioritized unless downstream sensitivity is detected.

---

## 3. **matrix_solver_fn**
**Type**: Cloud Function or Cloud Run (stateless)

### Input:
- **Source**: Pub/Sub topics (one per granularity):
  - `w.latent.5s`
  - `w.latent.15s`
- **Stateful**: ✅ Yes (transient)
  - Keeps last vector in memory for pairing
- **Resumable**: ❌ No
  - If restarted mid-pair, prior state is lost

### Output:
- **Destination**: Pub/Sub topic `M.latent.{granularity}`

### Output Schema:
```json
{
  "t0": "2025-06-09T18:53:00Z",
  "t1": "2025-06-09T18:53:05Z",
  "matrix": [
    [0.95, 0.02, 0.03],
    [0.01, 0.97, 0.02],
    [0.04, 0.01, 0.95]
  ],
  "w_t1": [0.18, -0.12, 0.06, ...]
}
```

### Training/Tuning Plan:
- **Already tuned.** Optimization method and constraint formulation are static.
- **No further training planned.**

---

## 4. **matrix_blender**
**Type**: GKE Deployment (long-running)

### Input:
- **Sources**:
  - `M.latent.{granularity}` (stream of derived matrices)
- **Stateful**: ✅ Yes (transient)
  - Maintains last `k` matrices in memory per granularity
  - Applies blending method (e.g., linear average, weighted average, or matrix product)
- **Resumable**: ❌ No
  - Restart loses blend history unless explicitly persisted (not currently planned)

### Output:
- **Destination**: Pub/Sub topic `M.blended.{granularity}`

### Output Schema:
```json
{
  "t_blend_end": "2025-06-09T18:53:10Z",
  "matrix": [
    [0.93, 0.03, 0.04],
    [0.02, 0.95, 0.03],
    [0.05, 0.02, 0.93]
  ],
  "w_t1": [0.18, -0.12, 0.06, ...]
}
```

### Training/Tuning Plan:
- **Parameters to learn:** Number of matrices `k`, blend weights or method.
- **Test metrics:**
  - MAE or RMSE of predicted `w(t+1)` using `M_blend` and `w(t)`.
  - Matrix stability (e.g., spectral norm drift, diagonal dominance).
  - **Evaluation strategy:** Compare `M_blend * w(t)` to actual future `w(t+1)`.

### Notes:
- This component performs matrix blending only, without forecasting.
- Blending strategy is currently undecided (e.g., linear average, matrix product).
- Includes the most recent `w_t1` from the last matrix for convenience.
- Eliminates need for downstream consumers to subscribe to `w.latent.{granularity}`.
- Forecasting is handled downstream.
- Output is a smoothed or composite matrix suitable for multi-step predictions.

---

## 5. Observability & Replayability

### 5.1 Observability Hooks
- **Metrics**: Emit custom Cloud Monitoring metrics at each stage (tick-fetcher, rollup, solver, blender, forecasting).
- **Logs**: Structured logs with trace IDs and minimal payload summaries.
- **Tracing**: Use Cloud Trace spans across Pub/Sub publishes and consumes.

### 5.2 Replay via Pub/Sub Topic
- **Enable message retention** on each Pub/Sub topic (e.g., 7 days).
- **Create snapshots** of the live subscription when starting a replay session:
  ```bash
  gcloud pubsub subscriptions snapshots create     --subscription=projects/PROJECT/subscriptions/SUB_NAME     SNAPSHOT_ID
  ```
- **Seek** the subscription back to the snapshot or a given timestamp:
  ```bash
  gcloud pubsub subscriptions seek     --subscription=projects/PROJECT/subscriptions/SUB_NAME     --snapshot=projects/PROJECT/snapshots/SNAPSHOT_ID
  ```
  or
  ```bash
  gcloud pubsub subscriptions seek     --subscription=projects/PROJECT/subscriptions/SUB_NAME     --time="2025-06-09T00:00:00Z"
  ```
- **Replay**: Consumers will automatically receive retained or replayed messages as if live.
- **Isolation**: Use separate subscriptions or labels to distinguish live vs replay traffic.

---

**Document version: 1.1**