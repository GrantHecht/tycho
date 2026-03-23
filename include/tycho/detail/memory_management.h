// =============================================================================
// Originally from ASSET (AlabamaASRL/asset_asrl)
// Copyright 2020-present The University of Alabama-Astrodynamics and Space
//   Research Lab. Licensed under the Apache License, Version 2.0
// License: notices/asset-apache2.txt.
// Source: https://github.com/AlabamaASRL/asset_asrl
// Original Developer: James B. Pezent
//
// Defines Tycho's memory manager for allocation of temporary matrices in
// VectorFunction expressions
//
// Modifications in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt):
//   - Namespace renamed: asset -> Tycho
//   - Python binding methods (Build(py::module)) moved to src/Bindings/ (PR 2)
//   - pybind11 header references removed
//   - Full rewrite: BumpStack with save/restore, SIMD alignment, learning
// =============================================================================

#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <functional>
#include <new>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "tycho/detail/eigen_types.h"
#include "tycho/detail/tuple_iterator.h"

namespace Tycho {

namespace detail {

/// Number of Scalar elements needed to reach EIGEN_MAX_ALIGN_BYTES alignment.
template <class Scalar>
inline constexpr size_t AlignElems = std::max<size_t>(1, EIGEN_MAX_ALIGN_BYTES / sizeof(Scalar));

/// Round `offset` up to the next aligned position for Scalar.
template <class Scalar> inline constexpr size_t align_up(size_t offset) {
    constexpr size_t A = AlignElems<Scalar>;
    return (offset + A - 1) & ~(A - 1);
}

/// Bump stack allocator with SIMD-aligned buffer, save/restore semantics,
/// and high-water learning.
template <class Scalar> struct BumpStack {

    BumpStack() : BumpStack(64) {}

    explicit BumpStack(size_t init_size) { resize(init_size); }

    ~BumpStack() { free_buffer(); }

    BumpStack(const BumpStack &) = delete;
    BumpStack &operator=(const BumpStack &) = delete;

    struct SavePoint {
        size_t offset;
        size_t overflow_count;
        size_t cumulative_overflow;
    };

    SavePoint save() const { return {offset_, overflow_.size(), cumulative_overflow_}; }

    void restore(SavePoint sp) {
        offset_ = sp.offset;
        overflow_.resize(sp.overflow_count);
        cumulative_overflow_ = sp.cumulative_overflow;
        if (offset_ == 0 && overflow_.empty() && high_water_ > capacity_) {
            resize(high_water_);
        }
    }

    Scalar *allocate(size_t n) {
        size_t aligned = align_up<Scalar>(offset_);
        size_t end = aligned + n;
        if (overflow_.empty() && end <= capacity_) {
            std::memset(data_ + aligned, 0, n * sizeof(Scalar));
            offset_ = end;
            if (offset_ > high_water_)
                high_water_ = offset_;
            return data_ + aligned;
        }
        // Overflow: allocation spills to a separate heap block. Track cumulative
        // overflow size so high_water_ reflects the true peak and the arena
        // converges to the right capacity in one warm-up cycle.
        overflow_.emplace_back(n);
        auto &blk = overflow_.back();
        blk.setZero();
        cumulative_overflow_ += n;
        size_t total = offset_ + cumulative_overflow_;
        if (total > high_water_)
            high_water_ = total;
        return blk.data();
    }

    void resize(size_t new_cap) {
        assert(offset_ == 0 && overflow_.empty() && "resize() while allocations active");
        free_buffer();
        capacity_ = new_cap;
        offset_ = 0;
        high_water_ = 0;
        cumulative_overflow_ = 0;
        overflow_.clear();
        if (capacity_ > 0) {
            constexpr size_t align = EIGEN_MAX_ALIGN_BYTES;
            data_ = static_cast<Scalar *>(
                ::operator new(capacity_ * sizeof(Scalar), std::align_val_t(align)));
        } else {
            data_ = nullptr;
        }
    }

    size_t size() const { return capacity_; }

  private:
    Scalar *data_ = nullptr;
    size_t capacity_ = 0;
    size_t offset_ = 0;
    size_t high_water_ = 0;
    size_t cumulative_overflow_ = 0;
    std::vector<VectorX<Scalar>> overflow_;

    void free_buffer() {
        if (data_) {
            constexpr size_t align = EIGEN_MAX_ALIGN_BYTES;
            ::operator delete(data_, std::align_val_t(align));
            data_ = nullptr;
        }
    }
};

template <int R, int C> struct RCBase {
    static const int rows = R;
    static const int cols = C;
    RCBase(int, int) {}
};
template <int R> struct RCBase<R, -1> {
    static const int rows = R;
    int cols;
    RCBase(int r, int c) : cols(c) {}
};
template <int C> struct RCBase<-1, C> {
    int rows;
    static const int cols = C;
    RCBase(int r, int c) : rows(r) {}
};
template <> struct RCBase<-1, -1> {
    int rows;
    int cols;
    RCBase(int r, int c) : rows(r), cols(c) {}
};

template <class... TempSpecs> struct ExactTempPack {
    std::tuple<typename std::remove_cvref_t<TempSpecs>::ExactTempType...> data;
    ExactTempPack() {}
};

} // namespace detail

/// Template type for specifying the type and size of temporary matrix that
/// must be created by the allocator.
template <class T> struct TempSpec : detail::RCBase<T::RowsAtCompileTime, T::ColsAtCompileTime> {
    using Base = detail::RCBase<T::RowsAtCompileTime, T::ColsAtCompileTime>;
    using ExactTempType = T;
    using MatType = T;
    using Scalar = typename T::Scalar;
    static constexpr int RowsAtCompileTime = T::RowsAtCompileTime;
    static constexpr int ColsAtCompileTime = T::ColsAtCompileTime;
    static constexpr bool IsConstantSize = (RowsAtCompileTime >= 0) && (ColsAtCompileTime >= 0);
    static constexpr bool IsArray = false;
    static constexpr bool IsTuple = false;

    TempSpec(int rows, int cols) : Base(rows, cols) {}
};

/// Template type for specifying a constant size array of TempSpecs that must
/// be allocated.
template <class T, int Size> struct ArrayOfTempSpecs {
    using Scalar = typename T::Scalar;
    using ExactTempType = std::array<T, Size>;
    using MatType = T;
    static constexpr int size = Size;
    static constexpr bool IsConstantSize = TempSpec<T>::IsConstantSize;
    static constexpr bool IsArray = true;
    static constexpr bool IsTuple = false;
    TempSpec<T> tspec;
    ArrayOfTempSpecs(int rows, int cols) : tspec(rows, cols) {}
};

/// Template type for specifying a heterogeneous tuple of TempSpecs that must
/// be allocated.
template <class... T> struct TupleOfTempSpecs {
    using Scalar = typename std::remove_cvref_t<decltype(std::get<0>(std::tuple<T...>()))>::Scalar;
    using ExactTempType = std::tuple<T...>;

    static constexpr bool IsConstantSize = (... && TempSpec<T>::IsConstantSize);
    static constexpr bool IsArray = false;
    static constexpr bool IsTuple = true;
    static constexpr int size = sizeof...(T);

    std::tuple<TempSpec<T>...> tspecs;
    TupleOfTempSpecs(std::tuple<TempSpec<T>...> tsp) : tspecs(tsp) {}
};

struct BumpAllocator {
    using ScalarStackType = detail::BumpStack<double>;
    using SuperScalarStackType = detail::BumpStack<DefaultSuperScalar>;

    template <class Func, class... TempSpecs>
    static void allocate_run(Func &&f, const TempSpecs &...tspecs) {
        if constexpr ((... && TempSpecs::IsConstantSize)) {
            auto Temps = detail::ExactTempPack<std::remove_cvref_t<TempSpecs>...>();
            std::apply(f, Temps.data);
        } else {
            BumpAllocator::run_arena_impl(f, tspecs...);
        }
    }

    static void resize(int sizeScalar, int sizeSuper) {
        BumpAllocator::ScalarStack.resize(sizeScalar);
        BumpAllocator::SuperScalarStack.resize(sizeSuper);
    }
    static void resize(int size) { BumpAllocator::resize(size, size); }
    static int size_scalar() { return static_cast<int>(BumpAllocator::ScalarStack.size()); }
    static int size_super_scalar() {
        return static_cast<int>(BumpAllocator::SuperScalarStack.size());
    }

  private:
    static thread_local ScalarStackType ScalarStack;
    static thread_local SuperScalarStackType SuperScalarStack;

    template <class Scalar> static detail::BumpStack<Scalar> &get_stack() {
        if constexpr (std::is_same_v<Scalar, double>) {
            return BumpAllocator::ScalarStack;
        } else {
            return BumpAllocator::SuperScalarStack;
        }
    }

    template <class... TempSpecs>
    inline static size_t count_blocksize_aligned(const TempSpecs &...tspecs) {
        size_t blksize = 0;

        auto CalcSpecSize = [](const auto &tspec) -> size_t {
            return static_cast<size_t>(tspec.rows) * static_cast<size_t>(tspec.cols);
        };

        auto CountSpace = [&](const auto &tspec) {
            using type = std::remove_cvref_t<decltype(tspec)>;
            using Scalar = typename type::Scalar;

            if constexpr (type::IsConstantSize) {
                // Constant-size: allocated on the stack, no arena space needed
            } else if constexpr (type::IsArray) {
                for (int i = 0; i < type::size; ++i) {
                    blksize = detail::align_up<Scalar>(blksize);
                    blksize += CalcSpecSize(tspec.tspec);
                }
            } else if constexpr (type::IsTuple) {
                Tycho::tuple_for_each(tspec.tspecs, [&](const auto &tspeci) {
                    using S = typename std::remove_cvref_t<decltype(tspeci)>::Scalar;
                    blksize = detail::align_up<S>(blksize);
                    blksize += CalcSpecSize(tspeci);
                });
            } else {
                blksize = detail::align_up<Scalar>(blksize);
                blksize += CalcSpecSize(tspec);
            }
        };
        (CountSpace(tspecs), ...);
        return blksize;
    }

    template <class Scalar, class... TempSpecs>
    inline static auto make_temps(Scalar *data, const TempSpecs &...tspecs) {
        size_t start = 0;

        auto make_map = [&](const auto &tspec) {
            using type = std::remove_cvref_t<decltype(tspec)>;
            using MAP = Eigen::Map<typename type::MatType, Eigen::AlignedMax>;
            start = detail::align_up<Scalar>(start);
            size_t start_t = start;
            start += static_cast<size_t>(tspec.rows) * static_cast<size_t>(tspec.cols);
            return MAP(data + start_t, tspec.rows, tspec.cols);
        };

        auto make_temp = [&](const auto &tspec) {
            using type = std::remove_cvref_t<decltype(tspec)>;
            if constexpr (type::IsConstantSize) {
                return typename type::ExactTempType();
            } else if constexpr (type::IsArray) {
                auto array_temp = [&](auto i) { return make_map(tspec.tspec); };
                return BumpAllocator::make_map_array(array_temp,
                                                     std::integral_constant<int, type::size>());
            } else if constexpr (type::IsTuple) {
                auto tuple_temp = [&](const auto &...tspeci) {
                    return std::tuple{make_map(tspeci)...};
                };
                return std::apply(tuple_temp, tspec.tspecs);
            } else {
                return make_map(tspec);
            }
        };

        return std::tuple{make_temp(tspecs)...};
    }

    template <class Func, class... TempSpecs>
    inline static void run_arena_impl(Func &&f, const TempSpecs &...tspecs) {
        using Scalar =
            typename std::remove_cvref_t<decltype(std::get<0>(std::tuple{tspecs...}))>::Scalar;

        auto &stack = BumpAllocator::get_stack<Scalar>();
        size_t blksize = BumpAllocator::count_blocksize_aligned(tspecs...);

        auto sp = stack.save();
        Scalar *data = stack.allocate(blksize);

        auto Temps = BumpAllocator::make_temps(data, tspecs...);
        std::apply(f, Temps);

        stack.restore(sp);
    }

    template <class Function, std::size_t... Indices>
    static auto make_map_array_helper(Function f, std::index_sequence<Indices...>)
        -> std::array<typename std::invoke_result<Function, std::size_t>::type,
                      sizeof...(Indices)> {
        return {{f(Indices)...}};
    }

    template <int N, class Function>
    static auto make_map_array(Function f, std::integral_constant<int, N> n)
        -> std::array<typename std::invoke_result<Function, std::size_t>::type, N> {
        return make_map_array_helper(f, std::make_index_sequence<N>{});
    }
};

} // namespace Tycho
