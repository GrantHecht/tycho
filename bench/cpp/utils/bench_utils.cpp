///////////////////////////////////////////////////////////////////////////////
// Utils benchmarks — TypeStorage SBO, MemoryManager, ThreadPool
///////////////////////////////////////////////////////////////////////////////

#include "Utils/TypeStorage.h"
#include "Tycho.h"
#include <benchmark/benchmark.h>

using namespace Tycho;

///////////////////////////////////////////////////////////////////////////////
// TypeStorage SBO benchmarks
///////////////////////////////////////////////////////////////////////////////

namespace {

struct BenchBase {
    virtual ~BenchBase() = default;
    virtual void clone_into(TypeStorage<BenchBase> &s) const = 0;
    virtual int value() const = 0;
};

struct BenchModel : BenchBase {
    int v_;
    explicit BenchModel(int v) : v_(v) {}
    void clone_into(TypeStorage<BenchBase> &s) const override { s.emplace<BenchModel>(v_); }
    int value() const override { return v_; }
};

} // namespace

static void BM_TypeStorage_EmplaceSmall(benchmark::State &state) {
    for (auto _ : state) {
        TypeStorage<BenchBase> s;
        s.emplace<BenchModel>(42);
        benchmark::DoNotOptimize(s);
    }
}
BENCHMARK(BM_TypeStorage_EmplaceSmall);

static void BM_TypeStorage_CopySmall(benchmark::State &state) {
    TypeStorage<BenchBase> src;
    src.emplace<BenchModel>(42);
    for (auto _ : state) {
        TypeStorage<BenchBase> copy(src);
        benchmark::DoNotOptimize(copy);
    }
}
BENCHMARK(BM_TypeStorage_CopySmall);

static void BM_TypeStorage_MoveSmall(benchmark::State &state) {
    for (auto _ : state) {
        TypeStorage<BenchBase> src;
        src.emplace<BenchModel>(42);
        TypeStorage<BenchBase> dst(std::move(src));
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_TypeStorage_MoveSmall);

///////////////////////////////////////////////////////////////////////////////
// MemoryManager
///////////////////////////////////////////////////////////////////////////////

static void BM_MemoryManager_Resize(benchmark::State &state) {
    for (auto _ : state) {
        MemoryManager::resize(256, 256);
    }
}
BENCHMARK(BM_MemoryManager_Resize);

///////////////////////////////////////////////////////////////////////////////
// ThreadPool
///////////////////////////////////////////////////////////////////////////////

static void BM_ThreadPool_Dispatch(benchmark::State &state) {
    ctpl::ThreadPool pool(1);
    for (auto _ : state) {
        auto fut = pool.push([](int) { return 42; });
        int val = fut.get();
        benchmark::DoNotOptimize(val);
    }
}
BENCHMARK(BM_ThreadPool_Dispatch);
