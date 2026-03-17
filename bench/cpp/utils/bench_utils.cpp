///////////////////////////////////////////////////////////////////////////////
// Utils benchmarks — TypeStorage SBO, BumpAllocator, ThreadPool
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
// BumpAllocator benchmarks
///////////////////////////////////////////////////////////////////////////////

static void BM_BumpAllocator_Resize(benchmark::State &state) {
    for (auto _ : state) {
        BumpAllocator::resize(256, 256);
        BumpAllocator::resize(1, 1);
    }
}
BENCHMARK(BM_BumpAllocator_Resize);

static void BM_AllocateRun_SmallVec(benchmark::State &state) {
    BumpAllocator::resize(256);
    using VType = Eigen::VectorXd;
    for (auto _ : state) {
        BumpAllocator::allocate_run([](auto &v) { benchmark::DoNotOptimize(v.data()); },
                                    TempSpec<VType>(6, 1));
    }
}
BENCHMARK(BM_AllocateRun_SmallVec);

static void BM_AllocateRun_JacobianBlock(benchmark::State &state) {
    BumpAllocator::resize(256);
    using MType = Eigen::MatrixXd;
    for (auto _ : state) {
        BumpAllocator::allocate_run([](auto &m) { benchmark::DoNotOptimize(m.data()); },
                                    TempSpec<MType>(6, 7));
    }
}
BENCHMARK(BM_AllocateRun_JacobianBlock);

static void BM_AllocateRun_MultiTemp(benchmark::State &state) {
    BumpAllocator::resize(512);
    using VType = Eigen::VectorXd;
    using MType = Eigen::MatrixXd;
    for (auto _ : state) {
        BumpAllocator::allocate_run(
            [](auto &v, auto &j, auto &h) {
                benchmark::DoNotOptimize(v.data());
                benchmark::DoNotOptimize(j.data());
                benchmark::DoNotOptimize(h.data());
            },
            TempSpec<VType>(6, 1), TempSpec<MType>(6, 7), TempSpec<MType>(7, 7));
    }
}
BENCHMARK(BM_AllocateRun_MultiTemp);

static void BM_AllocateRun_Scaled(benchmark::State &state) {
    int sz = static_cast<int>(state.range(0));
    BumpAllocator::resize(sz * 2);
    using VType = Eigen::VectorXd;
    for (auto _ : state) {
        BumpAllocator::allocate_run([](auto &v) { benchmark::DoNotOptimize(v.data()); },
                                    TempSpec<VType>(sz, 1));
    }
}
BENCHMARK(BM_AllocateRun_Scaled)->Arg(64)->Arg(256)->Arg(1024)->Arg(4096);

static void BM_AllocateRun_Overflow(benchmark::State &state) {
    BumpAllocator::resize(16);
    using VType = Eigen::VectorXd;
    for (auto _ : state) {
        BumpAllocator::allocate_run([](auto &v) { benchmark::DoNotOptimize(v.data()); },
                                    TempSpec<VType>(256, 1));
    }
    BumpAllocator::resize(256);
}
BENCHMARK(BM_AllocateRun_Overflow);

static void BM_AllocateRun_Nested(benchmark::State &state) {
    BumpAllocator::resize(512);
    using VType = Eigen::VectorXd;
    for (auto _ : state) {
        BumpAllocator::allocate_run(
            [](auto &outer) {
                benchmark::DoNotOptimize(outer.data());
                BumpAllocator::allocate_run(
                    [](auto &inner) { benchmark::DoNotOptimize(inner.data()); },
                    TempSpec<VType>(8, 1));
            },
            TempSpec<VType>(10, 1));
    }
}
BENCHMARK(BM_AllocateRun_Nested);

///////////////////////////////////////////////////////////////////////////////
// ThreadPool
///////////////////////////////////////////////////////////////////////////////

static void BM_ThreadPool_Dispatch(benchmark::State &state) {
    BS::thread_pool<> pool(1);
    for (auto _ : state) {
        auto fut = pool.submit_task([] { return 42; });
        int val = fut.get();
        benchmark::DoNotOptimize(val);
    }
}
BENCHMARK(BM_ThreadPool_Dispatch);

static void BM_ThreadPool_BatchDispatch(benchmark::State &state) {
    int N = static_cast<int>(state.range(0));
    BS::thread_pool<> pool(4);
    for (auto _ : state) {
        std::vector<std::future<int>> futures;
        futures.reserve(N);
        for (int i = 0; i < N; ++i) {
            futures.push_back(pool.submit_task([] { return 42; }));
        }
        int sum = 0;
        for (auto &f : futures) {
            sum += f.get();
        }
        benchmark::DoNotOptimize(sum);
    }
}
BENCHMARK(BM_ThreadPool_BatchDispatch)->Arg(10)->Arg(100);
