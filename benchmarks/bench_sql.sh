#!/usr/bin/env bash
#
# SQL benchmark suite for lsm_pg_extension
#
# Usage:
#   ./benchmarks/bench_sql.sh [--port PORT] [--db DB] [--rows N]
#
# Defaults: port=28817 (pgrx), db=postgres, rows=10000

set -euo pipefail

PORT="${PORT:-28817}"
DB="${DB:-postgres}"
ROWS="${ROWS:-10000}"
VECTOR_DIM=128
VECTOR_ROWS="${VECTOR_ROWS:-5000}"
KNN_K=10

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)  PORT="$2"; shift 2;;
        --db)    DB="$2"; shift 2;;
        --rows)  ROWS="$2"; VECTOR_ROWS="$((ROWS / 2))"; shift 2;;
        *)       echo "Unknown arg: $1"; exit 1;;
    esac
done

PSQL="psql -h localhost -p $PORT -d $DB -X -q"

header() { printf "\n\033[1;36m=== %s ===\033[0m\n" "$1"; }
timing() { printf "  \033[33m%-40s\033[0m %s\n" "$1" "$2"; }

random_vector() {
    local dim=$1
    local vals=""
    for ((i=0; i<dim; i++)); do
        if [ $i -gt 0 ]; then vals+=","; fi
        vals+=$(awk "BEGIN{printf \"%.6f\", rand()}" <<< "")
    done
    echo "[$vals]"
}

# ─────────────────────────────────────────────────────────────────────
header "Environment"
echo "  Port:         $PORT"
echo "  Database:     $DB"
echo "  KV Rows:      $ROWS"
echo "  Vector Rows:  $VECTOR_ROWS"
echo "  Vector Dim:   $VECTOR_DIM"

$PSQL -c "SELECT lsm_postgres_version();" 2>/dev/null || {
    echo "ERROR: Cannot connect or extension not loaded. Is pgrx running?"
    exit 1
}

# ─────────────────────────────────────────────────────────────────────
header "Benchmark 1: Key-Value INSERT throughput"

START=$(date +%s%N)
for ((i=1; i<=ROWS; i++)); do
    echo "SELECT lsm_s3_insert('bench_kv', 'key_$(printf '%08d' $i)', 'value_$(printf '%08d' $i)');"
done | $PSQL > /dev/null
END=$(date +%s%N)
ELAPSED_MS=$(( (END - START) / 1000000 ))
OPS_PER_SEC=$(awk "BEGIN{printf \"%.0f\", $ROWS / ($ELAPSED_MS / 1000.0)}")
timing "$ROWS inserts" "${ELAPSED_MS}ms (${OPS_PER_SEC} ops/sec)"

# ─────────────────────────────────────────────────────────────────────
header "Benchmark 2: Key-Value point SELECT (random)"

LOOKUPS=$((ROWS < 1000 ? ROWS : 1000))
START=$(date +%s%N)
for ((i=1; i<=LOOKUPS; i++)); do
    KEY_NUM=$(( (RANDOM % ROWS) + 1 ))
    echo "SELECT lsm_s3_select('bench_kv', 'key_$(printf '%08d' $KEY_NUM)');"
done | $PSQL > /dev/null
END=$(date +%s%N)
ELAPSED_MS=$(( (END - START) / 1000000 ))
OPS_PER_SEC=$(awk "BEGIN{printf \"%.0f\", $LOOKUPS / ($ELAPSED_MS / 1000.0)}")
timing "$LOOKUPS random reads" "${ELAPSED_MS}ms (${OPS_PER_SEC} ops/sec)"

# ─────────────────────────────────────────────────────────────────────
header "Benchmark 3: Key-Value range SCAN"

for SCAN_SIZE in 10 100 1000; do
    START_KEY="key_00000001"
    END_KEY="key_$(printf '%08d' $SCAN_SIZE)"

    START=$(date +%s%N)
    for ((rep=0; rep<10; rep++)); do
        echo "SELECT count(*) FROM lsm_s3_scan_range('bench_kv', '$START_KEY', '$END_KEY');"
    done | $PSQL > /dev/null
    END=$(date +%s%N)
    ELAPSED_MS=$(( (END - START) / 1000000 ))
    AVG_MS=$(awk "BEGIN{printf \"%.1f\", $ELAPSED_MS / 10.0}")
    timing "scan $SCAN_SIZE keys (avg of 10)" "${AVG_MS}ms"
done

# ─────────────────────────────────────────────────────────────────────
header "Benchmark 4: Flush to S3"

START=$(date +%s%N)
$PSQL -c "SELECT lsm_s3_flush_table('bench_kv');" > /dev/null
END=$(date +%s%N)
ELAPSED_MS=$(( (END - START) / 1000000 ))
timing "flush bench_kv ($ROWS rows)" "${ELAPSED_MS}ms"

# ─────────────────────────────────────────────────────────────────────
header "Benchmark 5: Post-flush point SELECT"

START=$(date +%s%N)
for ((i=1; i<=LOOKUPS; i++)); do
    KEY_NUM=$(( (RANDOM % ROWS) + 1 ))
    echo "SELECT lsm_s3_select('bench_kv', 'key_$(printf '%08d' $KEY_NUM)');"
done | $PSQL > /dev/null
END=$(date +%s%N)
ELAPSED_MS=$(( (END - START) / 1000000 ))
OPS_PER_SEC=$(awk "BEGIN{printf \"%.0f\", $LOOKUPS / ($ELAPSED_MS / 1000.0)}")
timing "$LOOKUPS post-flush reads" "${ELAPSED_MS}ms (${OPS_PER_SEC} ops/sec)"

# ─────────────────────────────────────────────────────────────────────
header "Benchmark 6: Vector INSERT"

START=$(date +%s%N)
for ((i=1; i<=VECTOR_ROWS; i++)); do
    VALS=""
    for ((d=0; d<VECTOR_DIM; d++)); do
        if [ $d -gt 0 ]; then VALS+=","; fi
        VALS+="$(awk -v seed=$((i * VECTOR_DIM + d)) 'BEGIN{srand(seed); printf "%.4f", rand()}')"
    done
    echo "SELECT lsm_s3_insert_vector('bench_vec', 'vec_$(printf '%06d' $i)', '[$VALS]'::lsm_vector);"
done | $PSQL > /dev/null
END=$(date +%s%N)
ELAPSED_MS=$(( (END - START) / 1000000 ))
OPS_PER_SEC=$(awk "BEGIN{printf \"%.0f\", $VECTOR_ROWS / ($ELAPSED_MS / 1000.0)}")
timing "$VECTOR_ROWS vector inserts (dim=$VECTOR_DIM)" "${ELAPSED_MS}ms (${OPS_PER_SEC} ops/sec)"

# ─────────────────────────────────────────────────────────────────────
header "Benchmark 7: Vector KNN Search (HNSW)"

$PSQL -c "SELECT lsm_s3_create_vector_index('bench_idx', 'bench_vec', $VECTOR_DIM, 16, 200, 'l2');" > /dev/null 2>&1 || true

SEARCH_ITERS=50
START=$(date +%s%N)
for ((i=1; i<=SEARCH_ITERS; i++)); do
    VALS=""
    for ((d=0; d<VECTOR_DIM; d++)); do
        if [ $d -gt 0 ]; then VALS+=","; fi
        VALS+="$(awk -v seed=$((i * 1000 + d)) 'BEGIN{srand(seed); printf "%.4f", rand()}')"
    done
    echo "SELECT lsm_s3_vector_search('bench_idx', '[$VALS]'::lsm_vector, $KNN_K);"
done | $PSQL > /dev/null
END=$(date +%s%N)
ELAPSED_MS=$(( (END - START) / 1000000 ))
AVG_MS=$(awk "BEGIN{printf \"%.1f\", $ELAPSED_MS / $SEARCH_ITERS}")
timing "KNN k=$KNN_K (avg of $SEARCH_ITERS queries)" "${AVG_MS}ms"

# ─────────────────────────────────────────────────────────────────────
header "Benchmark 8: Mixed read/write workload"

MIXED_OPS=2000
START=$(date +%s%N)
for ((i=1; i<=MIXED_OPS; i++)); do
    if (( RANDOM % 5 == 0 )); then
        echo "SELECT lsm_s3_insert('bench_mixed', 'mkey_$(printf '%08d' $i)', 'mval_$i');"
    else
        KEY_NUM=$(( (RANDOM % ROWS) + 1 ))
        echo "SELECT lsm_s3_select('bench_kv', 'key_$(printf '%08d' $KEY_NUM)');"
    fi
done | $PSQL > /dev/null
END=$(date +%s%N)
ELAPSED_MS=$(( (END - START) / 1000000 ))
OPS_PER_SEC=$(awk "BEGIN{printf \"%.0f\", $MIXED_OPS / ($ELAPSED_MS / 1000.0)}")
timing "$MIXED_OPS ops (80% read, 20% write)" "${ELAPSED_MS}ms (${OPS_PER_SEC} ops/sec)"

# ─────────────────────────────────────────────────────────────────────
header "Summary"
echo "  All benchmarks completed."
echo "  Note: In-memory storage (default). For S3, set lsm_s3.endpoint first."
