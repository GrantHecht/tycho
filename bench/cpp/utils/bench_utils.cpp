///////////////////////////////////////////////////////////////////////////////
// Utils benchmarks — TypeStorage SBO, MemoryManager, ThreadPool
///////////////////////////////////////////////////////////////////////////////

#include "../bench_common.h"
#include <benchmark/benchmark.h>
#include <future>

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
// MemoryManager — alternate sizes to force actual reallocation
///////////////////////////////////////////////////////////////////////////////

static void BM_MemoryManager_Resize(benchmark::State &state) {
    for (auto _ : state) {
        MemoryManager::resize(256, 256);
        MemoryManager::resize(1, 1);
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

static void BM_ThreadPool_BatchDispatch(benchmark::State &state) {
    int N = static_cast<int>(state.range(0));
    ctpl::ThreadPool pool(4);
    for (auto _ : state) {
        std::vector<std::future<int>> futures;
        futures.reserve(N);
        for (int i = 0; i < N; ++i) {
            futures.push_back(pool.push([](int) { return 42; }));
        }
        int sum = 0;
        for (auto &f : futures) {
            sum += f.get();
        }
        benchmark::DoNotOptimize(sum);
    }
}
BENCHMARK(BM_ThreadPool_BatchDispatch)->Arg(10)->Arg(100);
