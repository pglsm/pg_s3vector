use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lsm_engine::memtable::MemTableValue;
use lsm_engine::{LsmConfig, LsmStore};
use rand::Rng;
use std::sync::Arc;
use std::time::Duration;

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn random_value(rng: &mut impl Rng, size: usize) -> Vec<u8> {
    let mut buf = vec![0u8; size];
    rng.fill(&mut buf[..]);
    buf
}

fn make_key(i: u64) -> Vec<u8> {
    format!("key-{:010}", i).into_bytes()
}

// ---------------------------------------------------------------------------
// MemTable benchmarks (pure in-memory writes and reads)
// ---------------------------------------------------------------------------

fn bench_memtable_put(c: &mut Criterion) {
    let mut group = c.benchmark_group("memtable_put");
    let value_size = 256;

    for &count in &[1_000u64, 10_000, 100_000] {
        group.throughput(Throughput::Elements(count));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &n| {
            b.iter(|| {
                let mt = lsm_engine::memtable::MemTable::new(0);
                let mut rng = rand::thread_rng();
                for i in 0..n {
                    let val = random_value(&mut rng, value_size);
                    mt.put(&make_key(i), &val).unwrap();
                }
            });
        });
    }
    group.finish();
}

fn bench_memtable_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("memtable_get");
    let n = 10_000u64;
    let value_size = 256;

    let mt = lsm_engine::memtable::MemTable::new(0);
    let mut rng = rand::thread_rng();
    for i in 0..n {
        mt.put(&make_key(i), &random_value(&mut rng, value_size)).unwrap();
    }

    group.throughput(Throughput::Elements(1));
    group.bench_function("random_hit", |b| {
        let mut rng = rand::thread_rng();
        b.iter(|| {
            let i = rng.gen_range(0..n);
            mt.get(&make_key(i))
        });
    });
    group.bench_function("miss", |b| {
        b.iter(|| mt.get(b"nonexistent-key-xxxxxxxxx"));
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// LsmStore end-to-end benchmarks (in-memory object store)
// ---------------------------------------------------------------------------

fn bench_store_put(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_put");
    group.measurement_time(Duration::from_secs(10));
    let rt = runtime();

    for &count in &[100u64, 1_000, 10_000] {
        group.throughput(Throughput::Elements(count));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, &n| {
            b.to_async(&rt).iter(|| async move {
                let config = LsmConfig::in_memory("/bench")
                    .with_wal(false)
                    .with_memtable_size(256 * 1024 * 1024);
                let store = LsmStore::open(config).await.unwrap();
                let mut rng = rand::thread_rng();
                for i in 0..n {
                    let val = random_value(&mut rng, 256);
                    store.put(&make_key(i), &val).await.unwrap();
                }
                store.close().await.unwrap();
            });
        });
    }
    group.finish();
}

fn bench_store_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_get");
    let rt = runtime();
    let n = 5_000u64;

    let store = Arc::new(rt.block_on(async {
        let config = LsmConfig::in_memory("/bench-get")
            .with_wal(false)
            .with_memtable_size(256 * 1024 * 1024);
        let store = LsmStore::open(config).await.unwrap();
        let mut rng = rand::thread_rng();
        for i in 0..n {
            store.put(&make_key(i), &random_value(&mut rng, 256)).await.unwrap();
        }
        store
    }));

    group.throughput(Throughput::Elements(1));

    let s = store.clone();
    group.bench_function("random_hit", |b| {
        let mut rng = rand::thread_rng();
        let s = s.clone();
        b.to_async(&rt).iter(|| {
            let s = s.clone();
            let key = make_key(rng.gen_range(0..n));
            async move { s.get(&key).await.unwrap() }
        });
    });

    let s = store.clone();
    group.bench_function("miss", |b| {
        let s = s.clone();
        b.to_async(&rt).iter(|| {
            let s = s.clone();
            async move { s.get(b"nonexistent-key-xxxxxxxxx").await.unwrap() }
        });
    });
    group.finish();

    if let Ok(s) = Arc::try_unwrap(store) {
        rt.block_on(s.close()).unwrap();
    }
}

fn bench_store_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_scan");
    let rt = runtime();
    let n = 10_000u64;

    let store = Arc::new(rt.block_on(async {
        let config = LsmConfig::in_memory("/bench-scan")
            .with_wal(false)
            .with_memtable_size(256 * 1024 * 1024);
        let store = LsmStore::open(config).await.unwrap();
        let mut rng = rand::thread_rng();
        for i in 0..n {
            store.put(&make_key(i), &random_value(&mut rng, 256)).await.unwrap();
        }
        store
    }));

    for &scan_size in &[10u64, 100, 1000] {
        group.throughput(Throughput::Elements(scan_size));
        let s = store.clone();
        group.bench_with_input(
            BenchmarkId::new("range", scan_size),
            &scan_size,
            |b, &sz| {
                let s = s.clone();
                b.to_async(&rt).iter(|| {
                    let s = s.clone();
                    let start = make_key(0);
                    let end = make_key(sz);
                    async move { s.scan(&start, &end).await.unwrap() }
                });
            },
        );
    }
    group.finish();

    if let Ok(s) = Arc::try_unwrap(store) {
        rt.block_on(s.close()).unwrap();
    }
}

// ---------------------------------------------------------------------------
// SSTable build + read benchmarks
// ---------------------------------------------------------------------------

fn bench_sstable_build(c: &mut Criterion) {
    use lsm_engine::sstable::SSTableBuilder;

    let mut group = c.benchmark_group("sstable_build");

    for &count in &[1_000u64, 10_000] {
        group.throughput(Throughput::Elements(count));
        group.bench_with_input(
            BenchmarkId::new("uncompressed", count),
            &count,
            |b, &n| {
                let mut rng = rand::thread_rng();
                let entries: Vec<_> = (0..n)
                    .map(|i| (make_key(i), random_value(&mut rng, 256), i))
                    .collect();
                b.iter(|| {
                    let mut builder = SSTableBuilder::new(4096, false);
                    for (key, val, seq) in &entries {
                        builder.add(key.clone(), *seq, MemTableValue::Put(val.clone()));
                    }
                    builder.build().unwrap()
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("compressed", count),
            &count,
            |b, &n| {
                let mut rng = rand::thread_rng();
                let entries: Vec<_> = (0..n)
                    .map(|i| (make_key(i), random_value(&mut rng, 256), i))
                    .collect();
                b.iter(|| {
                    let mut builder = SSTableBuilder::new(4096, true);
                    for (key, val, seq) in &entries {
                        builder.add(key.clone(), *seq, MemTableValue::Put(val.clone()));
                    }
                    builder.build().unwrap()
                });
            },
        );
    }
    group.finish();
}

fn bench_sstable_read(c: &mut Criterion) {
    use lsm_engine::sstable::{SSTableBuilder, SSTableReader};

    let mut group = c.benchmark_group("sstable_read");
    let n = 10_000u64;

    let mut rng = rand::thread_rng();
    let mut builder = SSTableBuilder::new(4096, false);
    for i in 0..n {
        let val = random_value(&mut rng, 256);
        builder.add(make_key(i), i, MemTableValue::Put(val));
    }
    let (data, _meta) = builder.build().unwrap();

    group.throughput(Throughput::Elements(1));
    group.bench_function("point_lookup", |b| {
        let mut rng = rand::thread_rng();
        b.iter(|| {
            let i = rng.gen_range(0..n);
            let reader = SSTableReader::open(data.clone()).unwrap();
            reader.get(&make_key(i)).unwrap()
        });
    });
    group.bench_function("range_scan_100", |b| {
        b.iter(|| {
            let reader = SSTableReader::open(data.clone()).unwrap();
            reader.scan(&make_key(0), &make_key(100)).unwrap()
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Mixed workload: read/write ratio
// ---------------------------------------------------------------------------

fn bench_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_workload");
    group.measurement_time(Duration::from_secs(15));
    let rt = runtime();

    let ops = 5_000u64;
    group.throughput(Throughput::Elements(ops));

    group.bench_function("80read_20write", |b| {
        b.to_async(&rt).iter(|| async {
            let config = LsmConfig::in_memory("/bench-mixed")
                .with_wal(false)
                .with_memtable_size(256 * 1024 * 1024);
            let store = LsmStore::open(config).await.unwrap();
            let mut rng = rand::thread_rng();

            for i in 0..1000u64 {
                store.put(&make_key(i), &random_value(&mut rng, 256)).await.unwrap();
            }

            let mut write_counter = 1000u64;
            for _ in 0..ops {
                if rng.gen_ratio(1, 5) {
                    store.put(&make_key(write_counter), &random_value(&mut rng, 256)).await.unwrap();
                    write_counter += 1;
                } else {
                    let i = rng.gen_range(0..write_counter);
                    store.get(&make_key(i)).await.unwrap();
                }
            }
            store.close().await.unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_memtable_put,
    bench_memtable_get,
    bench_store_put,
    bench_store_get,
    bench_store_scan,
    bench_sstable_build,
    bench_sstable_read,
    bench_mixed_workload,
);
criterion_main!(benches);
