# Performance Issues Analysis Report

This report identifies performance anti-patterns, N+1 queries, inefficient algorithms, and other optimization opportunities in the Compass codebase.

---

## Critical Issues

### 1. N+1 Query Pattern in TopStocksService

**File:** `webservice/stockpredictor/predictor/services.py:366-394`

```python
# Get all active stocks
stocks = Stock.objects.all()  # 1 query
predictions = []

for stock in stocks:  # N iterations
    prediction = self.prediction_service.get_latest_prediction(stock.ticker)  # +1 query each
```

**Problem:** For 1000 stocks, this executes 1001 database queries instead of 1-2.

**Fix:** Batch fetch all predictions in a single query:
```python
from django.db.models import Prefetch

stocks = Stock.objects.prefetch_related(
    Prefetch('stockprediction_set',
             queryset=StockPrediction.objects.order_by('-prediction_date')[:1],
             to_attr='latest_predictions')
).all()
```

---

### 2. N+1 Queries in Views

**File:** `webservice/stockpredictor/predictor/views.py`

Multiple view functions exhibit the same pattern:

| Location | Pattern |
|----------|---------|
| Lines 89-92 | `Stock.objects.get()` then `StockPrediction.objects.filter()` |
| Lines 136-139 | Same in `get_prediction_graph()` |
| Lines 152-155 | Same in `prediction_history()` |
| Lines 221-224 | Same in `market_stats()` |

**Fix:** Use `select_related()` to join in a single query:
```python
prediction = StockPrediction.objects.select_related('stock').filter(
    stock__ticker=ticker
).order_by('-prediction_date').first()
```

---

### 3. O(n²) Complexity in Ensemble Training

**File:** `training/ensemble.py:378-394`

```python
n_models = len(self.ensemble_models)
interaction_features = []
for i in range(n_models):           # O(n)
    for j in range(i + 1, n_models):  # O(n)
        diff = meta_train[:, i] - meta_train[:, j]
        interaction_features.append(tf.expand_dims(diff, -1))
        ratio = meta_train[:, i] / (tf.abs(meta_train[:, j]) + 1e-7)
        interaction_features.append(tf.expand_dims(ratio, -1))
```

**Problem:** Creates n(n-1) tensor operations. With 10 models = 90 tensors per call. This logic is also **duplicated** at lines 416-422 for validation data.

**Fix:**
1. Extract to a reusable function
2. Use vectorized operations instead of nested loops:
```python
def compute_interaction_features(meta_data, n_models):
    """Vectorized interaction feature computation."""
    # Use tf.meshgrid and broadcasting for vectorized computation
    indices = tf.range(n_models)
    i_idx, j_idx = tf.meshgrid(indices, indices, indexing='ij')
    mask = i_idx < j_idx  # Upper triangle

    # Vectorized difference and ratio computation
    expanded = tf.expand_dims(meta_data, -1)
    diff_matrix = expanded - tf.transpose(expanded, [0, 2, 1])
    ratio_matrix = expanded / (tf.abs(tf.transpose(expanded, [0, 2, 1])) + 1e-7)

    # Extract upper triangle values
    return tf.concat([diff_matrix, ratio_matrix], axis=-1)
```

---

## High Severity Issues

### 4. Inefficient `iterrows()` Usage (14 instances)

**Affected Files:**

| File | Line(s) | Context |
|------|---------|---------|
| `shared/technical_analysis/indicators.py` | 140 | Building OHLCV list |
| `shared/technical_analysis/orchestrator.py` | 162 | Processing data rows |
| `shared/technical_analysis/elliott_wave.py` | 350 | Window iteration |
| `downloader/downloader.py` | 466 | Chunk processing for DB insert |
| `downloader/technical_analysis.py` | 282, 389, 864, 1550, 1626 | Multiple analysis functions |
| `training/technical_analysis.py` | 281, 388, 886, 1572, 1648 | Same patterns |

**Problem:** `iterrows()` is ~100x slower than vectorized operations. Each iteration creates a new Series object.

**Fix Examples:**

```python
# BAD - indicators.py:132-141
ohlcv_data = [
    OHLCV(open=float(row['open']), ...)
    for _, row in data.iterrows()
]

# GOOD - Use numpy arrays directly
ohlcv_data = [
    OHLCV(open=o, high=h, low=l, close=c, volume=v)
    for o, h, l, c, v in zip(
        data['open'].values,
        data['high'].values,
        data['low'].values,
        data['close'].values,
        data['volume'].values
    )
]

# BETTER - Use to_records() for bulk conversion
records = data[['open', 'high', 'low', 'close', 'volume']].to_records(index=False)
ohlcv_data = [OHLCV(*r) for r in records]
```

---

### 5. List Extension in Loops Instead of Pre-allocation

**File:** `training/ensemble.py:307-338`

```python
all_predictions = [[] for _ in self.ensemble_models]  # Lists that grow
actual_targets = []

for X_batch, y_batch in dataset:
    for idx, model in enumerate(self.ensemble_models):
        pred = model.predict(X_batch, verbose=0)
        all_predictions[idx].extend(pred.flatten())  # Slow extension
    actual_targets.extend(y_batch.numpy().flatten())  # Slow extension

# Convert at the end
meta_features = np.array(all_predictions).T  # Expensive conversion
```

**Problem:** Python lists resize dynamically (1.5x growth factor), causing memory thrashing with large datasets.

**Fix:** Pre-allocate numpy arrays:
```python
# Count total samples first
total_samples = sum(y.shape[0] for _, y in dataset)

# Pre-allocate
all_predictions = np.zeros((len(self.ensemble_models), total_samples))
actual_targets = np.zeros(total_samples)

# Fill in-place
offset = 0
for X_batch, y_batch in dataset:
    batch_size = y_batch.shape[0]
    for idx, model in enumerate(self.ensemble_models):
        all_predictions[idx, offset:offset+batch_size] = model.predict(X_batch, verbose=0).flatten()
    actual_targets[offset:offset+batch_size] = y_batch.numpy().flatten()
    offset += batch_size
```

---

## Medium Severity Issues

### 6. Unnecessary DataFrame Copies

**Locations:**
- `shared/technical_analysis/indicators.py:69`
- `shared/technical_analysis/orchestrator.py:101`
- `shared/technical_analysis/elliott_wave.py:224`
- `training/technical_analysis.py:1387`
- `downloader/technical_analysis.py:1365`

```python
data = data.copy()  # Avoid SettingWithCopyWarning
```

**Problem:** Full DataFrame copies double memory usage temporarily. For large stock datasets, this is significant.

**Fix:** Use `pd.options.mode.copy_on_write = True` (pandas 2.0+) or operate on views where safe:
```python
# Instead of copy(), use loc[] for explicit modification
data.loc[:, 'new_col'] = values
```

---

### 7. Inefficient QuerySet Handling

**File:** `webservice/stockpredictor/predictor/views.py`

| Line | Issue |
|------|-------|
| 67 | `Stock.objects.all()[:10]` - Fetches all, then slices |
| 171 | `Stock.objects.all().order_by('-market_cap')[:20]` - Orders all records |
| 173 | `list(stocks)` - Forces evaluation when QuerySet would work |
| 296 | `list(stock_data.values())` - Full query evaluation for caching |

**Fix:** Use pagination or database-level limits:
```python
# Use .only() to limit fields
stocks = Stock.objects.only('ticker', 'market_cap').order_by('-market_cap')[:20]

# Don't convert to list unless necessary
# QuerySets are lazy - converting forces immediate evaluation
```

---

### 8. Single Database Connection Without Pooling

**File:** `shared/technical_analysis/orchestrator.py:45-58`

```python
def _setup_db_connection(self) -> None:
    try:
        self.db_connection = psycopg2.connect(...)  # Single connection
    except psycopg2.Error as e:
        self.db_connection = None
```

**Problem:** Single persistent connection can become a bottleneck. No connection pooling means:
- No concurrent query handling
- Connection can become stale
- No automatic reconnection

**Fix:** Use connection pooling consistently:
```python
from psycopg2 import pool

class TechnicalIndicatorOrchestrator:
    def __init__(self):
        self.connection_pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            host=os.getenv('DB_HOST'),
            ...
        )

    def get_connection(self):
        return self.connection_pool.getconn()

    def release_connection(self, conn):
        self.connection_pool.putconn(conn)
```

---

### 9. Synchronous Stock Processing

**File:** `webservice/stockpredictor/predictor/services.py:371-394`

```python
for stock in stocks:  # Synchronous iteration
    prediction = self.prediction_service.get_latest_prediction(stock.ticker)
    # ... process each stock one by one
```

**Problem:** Processing thousands of stocks sequentially is slow when they could be processed in parallel.

**Fix:** Use async/await or batch processing:
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def get_top_stocks_async(self, limit=10):
    stocks = await Stock.objects.all().aasync()

    async def process_stock(stock):
        prediction = await self.prediction_service.get_latest_prediction_async(stock.ticker)
        # ... return processed result

    tasks = [process_stock(stock) for stock in stocks]
    return await asyncio.gather(*tasks)
```

---

## Summary Table

| Priority | Issue | File(s) | Impact |
|----------|-------|---------|--------|
| **CRITICAL** | N+1 queries in services | services.py:366-394 | 1000x slower for 1000 stocks |
| **CRITICAL** | N+1 queries in views | views.py (multiple) | Multiple DB hits per request |
| **HIGH** | O(n²) ensemble features | ensemble.py:378-394, 416-422 | Scales poorly with models |
| **HIGH** | iterrows() usage | 14 instances across 6 files | ~100x slower than vectorized |
| **HIGH** | List extend in loops | ensemble.py:307-338 | Memory thrashing |
| **MEDIUM** | DataFrame copies | 5 instances | 2x memory overhead |
| **MEDIUM** | Inefficient QuerySets | views.py | Unnecessary data loading |
| **MEDIUM** | No connection pooling | orchestrator.py:45-58 | Connection bottleneck |
| **MEDIUM** | Synchronous processing | services.py:371 | Sequential bottleneck |

---

## Recommended Fix Priority

1. **Immediate (High ROI):**
   - Add `select_related()`/`prefetch_related()` to eliminate N+1 queries
   - Replace `iterrows()` with vectorized operations in hot paths

2. **Short-term:**
   - Pre-allocate numpy arrays instead of list extension
   - Extract duplicated O(n²) logic to reusable function
   - Add connection pooling to orchestrator

3. **Medium-term:**
   - Implement async processing for batch stock operations
   - Review and optimize DataFrame copies
   - Add proper pagination to QuerySets

---

*Report generated: 2026-01-13*
