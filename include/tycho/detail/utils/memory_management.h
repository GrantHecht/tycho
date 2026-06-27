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
//   - Namespace renamed: asset -> tycho (with sub-namespaces tycho::vf, tycho::oc, etc.)
//   - Python binding methods moved to src/bindings/ (nanobind)
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

#include "tycho/detail/typedefs/eigen_types.h"
#include "tycho/detail/utils/tuple_iterator.h"

namespace tycho::utils {

namespace detail {

/// Number of Scalar elements needed to reach EIGEN_MAX_ALIGN_BYTES alignment.
template <class Scalar>
inline constexpr size_t AlignElems = std::max<size_t>(1, EIGEN_MAX_ALIGN_BYTES / sizeof(Scalar));

/// Round `offset` up to the next aligned position for Scalar.
template <class Scalar> inline constexpr size_t align_up(size_t offset) {
    constexpr size_t A = AlignElems<Scalar>;
    return (offset + A - 1) & ~(A - 1);
}

/// @internal
/// @brief Bump stack allocator with SIMD-aligned buffer, save/restore semantics,
/// and high-water learning.
template <class Scalar> struct BumpStack {

    /// @internal
    /// @brief Construct with an initial buffer of 64 elements.
    BumpStack() : BumpStack(64) {}

    /// @internal
    /// @brief Construct with an initial buffer of `init_size` elements.
    explicit BumpStack(size_t init_size) { resize(init_size); }

    ~BumpStack() { free_buffer(); }

    BumpStack(const BumpStack &) = delete;
    BumpStack &operator=(const BumpStack &) = delete;

    /// @internal
    /// @brief Snapshot of the allocator state for save/restore.
    struct SavePoint {
        size_t offset;           ///< @internal Current bump offset in elements.
        size_t overflow_count;   ///< @internal Number of active overflow blocks.
        size_t cumulative_overflow; ///< @internal Total elements in overflow blocks.
    };

    /// @internal
    /// @brief Capture the current allocator state.
    SavePoint save() const { return {offset_, overflow_.size(), cumulative_overflow_}; }

    /// @internal
    /// @brief Restore the allocator to a previously saved state.
    void restore(SavePoint sp) {
        offset_ = sp.offset;
        overflow_.resize(sp.overflow_count);
        cumulative_overflow_ = sp.cumulative_overflow;
        if (offset_ == 0 && overflow_.empty() && high_water_ > capacity_) {
            resize(high_water_);
        }
    }

    /// @internal
    /// @brief Allocate `n` elements; falls back to heap overflow if the arena is full.
    Scalar *allocate(size_t n) {
        size_t aligned = align_up<Scalar>(offset_);
        size_t end = aligned + n;
        if (overflow_.empty() && end <= capacity_) {
            std::memset(static_cast<void *>(data_ + aligned), 0, n * sizeof(Scalar));
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

    /// @internal
    /// @brief Resize the arena to `new_cap` elements. Must be called with no active
    /// allocations.
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

    /// @internal
    /// @brief Return the current arena capacity in elements.
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

/// @internal
/// @brief Compile-time row/column storage for TempSpec; fully static when both R and C >= 0.
template <int R, int C> struct RCBase {
    static constexpr int rows = R; ///< @internal Compile-time row count.
    static constexpr int cols = C; ///< @internal Compile-time column count.
    /// @internal
    /// @brief Construct; runtime arguments are ignored when both dimensions are static.
    RCBase(int, int) {}
};

/// @internal
/// @brief Partial specialisation: static rows, dynamic columns.
template <int R> struct RCBase<R, -1> {
    static constexpr int rows = R; ///< @internal Compile-time row count.
    int cols;                      ///< @internal Runtime column count.
    /// @internal
    /// @brief Construct with runtime column count.
    RCBase(int r, int c) : cols(c) {}
};

/// @internal
/// @brief Partial specialisation: dynamic rows, static columns.
template <int C> struct RCBase<-1, C> {
    int rows;                      ///< @internal Runtime row count.
    static constexpr int cols = C; ///< @internal Compile-time column count.
    /// @internal
    /// @brief Construct with runtime row count.
    RCBase(int r, int c) : rows(r) {}
};

/// @internal
/// @brief Fully-dynamic specialisation: both row and column counts are runtime values.
template <> struct RCBase<-1, -1> {
    int rows; ///< @internal Runtime row count.
    int cols; ///< @internal Runtime column count.
    /// @internal
    /// @brief Construct with runtime row and column counts.
    RCBase(int r, int c) : rows(r), cols(c) {}
};

/// @internal
/// @brief Tuple of default-constructed stack temporaries for fully compile-time-sized specs.
template <class... TempSpecs> struct ExactTempPack {
    /// @internal
    /// @brief Stack-allocated temporaries.
    std::tuple<typename std::remove_cvref_t<TempSpecs>::ExactTempType...> data;
    ExactTempPack() {}
};

} // namespace detail

/// @brief Descriptor for a single temporary matrix allocation by `BumpAllocator`.
///
/// Encodes the Eigen matrix type `T` together with its runtime dimensions (when
/// either dimension is dynamic). Pass one or more `TempSpec` values to
/// `BumpAllocator::allocate_run()`.
///
/// @tparam T Eigen matrix or array type whose dimensions are described.
template <class T> struct TempSpec : detail::RCBase<T::RowsAtCompileTime, T::ColsAtCompileTime> {
    /// @internal @brief RCBase specialisation.
    using Base = detail::RCBase<T::RowsAtCompileTime, T::ColsAtCompileTime>;
    using ExactTempType = T;               ///< @internal Exact Eigen type to instantiate.
    using MatType = T;                     ///< @internal Underlying matrix type.
    using Scalar = typename T::Scalar;     ///< @internal Scalar element type.
    /// @internal @brief Compile-time row count (-1 if dynamic).
    static constexpr int RowsAtCompileTime = T::RowsAtCompileTime;
    /// @internal @brief Compile-time column count (-1 if dynamic).
    static constexpr int ColsAtCompileTime = T::ColsAtCompileTime;
    /// @internal @brief True when both dimensions are compile-time constants.
    static constexpr bool IsConstantSize =
        (RowsAtCompileTime >= 0) && (ColsAtCompileTime >= 0);
    static constexpr bool IsArray = false; ///< @internal False for scalar TempSpec.
    static constexpr bool IsTuple = false; ///< @internal False for scalar TempSpec.

    /// @brief Construct with runtime row and column counts.
    /// @param rows Number of rows (ignored for compile-time-sized types).
    /// @param cols Number of columns (ignored for compile-time-sized types).
    TempSpec(int rows, int cols) : Base(rows, cols) {}
};

/// @brief Descriptor for a fixed-length array of identical temporary matrices.
///
/// Pass to `BumpAllocator::allocate_run()` to allocate `std::array<T, Size>` temporaries.
///
/// @tparam T    Eigen matrix or array type for each element.
/// @tparam Size Number of elements in the array.
template <class T, int Size> struct ArrayOfTempSpecs {
    using Scalar = typename T::Scalar;     ///< @internal Scalar element type.
    /// @internal @brief Exact `std::array` type to instantiate.
    using ExactTempType = std::array<T, Size>;
    /// @internal @brief Underlying matrix type of each element.
    using MatType = T;
    static constexpr int size = Size;      ///< @internal Number of array elements.
    /// @internal @brief True when T has compile-time dimensions.
    static constexpr bool IsConstantSize = TempSpec<T>::IsConstantSize;
    static constexpr bool IsArray = true;  ///< @internal True for ArrayOfTempSpecs.
    static constexpr bool IsTuple = false; ///< @internal False for ArrayOfTempSpecs.
    TempSpec<T> tspec;                     ///< @internal Descriptor for each element.
    /// @brief Construct with runtime row and column counts for each element.
    ArrayOfTempSpecs(int rows, int cols) : tspec(rows, cols) {}
};

/// @brief Descriptor for a heterogeneous tuple of temporary matrices.
///
/// Pass to `BumpAllocator::allocate_run()` to allocate `std::tuple<T...>` temporaries
/// where each element may be a different Eigen type.
///
/// @tparam T... Eigen matrix or array types for each element.
template <class... T> struct TupleOfTempSpecs {
    /// @internal @brief Scalar type of the first element.
    using Scalar =
        typename std::remove_cvref_t<decltype(std::get<0>(std::tuple<T...>()))>::Scalar;
    using ExactTempType = std::tuple<T...>; ///< @internal Exact `std::tuple` type to instantiate.

    /// @internal @brief True when all elements have compile-time dimensions.
    static constexpr bool IsConstantSize = (... && TempSpec<T>::IsConstantSize);
    static constexpr bool IsArray = false;    ///< @internal False for TupleOfTempSpecs.
    static constexpr bool IsTuple = true;     ///< @internal True for TupleOfTempSpecs.
    static constexpr int size = sizeof...(T); ///< @internal Number of tuple elements.

    std::tuple<TempSpec<T>...> tspecs; ///< @internal Descriptors for each element.
    /// @brief Construct from a tuple of per-element `TempSpec` descriptors.
    /// @param tsp Tuple of `TempSpec<T>` descriptors, one per element type.
    TupleOfTempSpecs(std::tuple<TempSpec<T>...> tsp) : tspecs(tsp) {}
};

/// @brief Thread-local bump allocator for temporary Eigen matrices used in VectorFunction
/// evaluation.
///
/// `allocate_run()` allocates one or more temporaries described by `TempSpec`,
/// `ArrayOfTempSpecs`, or `TupleOfTempSpecs`, calls `f` with the allocated objects, and
/// then releases the memory. When all dimensions are compile-time constants the allocations
/// are placed on the real stack with zero arena overhead.
///
/// Call `resize()` at startup to pre-size the per-thread arenas and avoid reallocation
/// during hot-path evaluation.
struct BumpAllocator {
    using ScalarStackType = detail::BumpStack<double>;         ///< @internal Scalar arena type.
    /// @internal @brief SuperScalar arena type.
    using SuperScalarStackType = detail::BumpStack<tycho::DefaultSuperScalar>;

    /// @brief Allocate temporaries and invoke `f` with them.
    ///
    /// When all specs have compile-time sizes, temporaries are stack-allocated and the
    /// arena is bypassed entirely. Otherwise the thread-local bump arena is used.
    ///
    /// @tparam Func      Callable whose parameter types match the allocated temporaries.
    /// @tparam TempSpecs Pack of `TempSpec`, `ArrayOfTempSpecs`, or `TupleOfTempSpecs`.
    /// @param f       Callable to invoke with the allocated temporaries.
    /// @param tspecs  Descriptors specifying the type and runtime size of each temporary.
    template <class Func, class... TempSpecs>
    static void allocate_run(Func &&f, const TempSpecs &...tspecs) {
        if constexpr ((... && std::remove_cvref_t<TempSpecs>::IsConstantSize)) {
            auto Temps = detail::ExactTempPack<std::remove_cvref_t<TempSpecs>...>();
            std::apply(f, Temps.data);
        } else {
            BumpAllocator::run_arena_impl(f, tspecs...);
        }
    }

    /// @brief Resize the per-thread scalar and super-scalar arenas independently.
    /// @param sizeScalar     New capacity in `double` elements for the scalar arena.
    /// @param sizeSuper      New capacity in `DefaultSuperScalar` elements for the
    ///                       super-scalar arena.
    static void resize(int sizeScalar, int sizeSuper) {
        BumpAllocator::ScalarStack.resize(sizeScalar);
        BumpAllocator::SuperScalarStack.resize(sizeSuper);
    }

    /// @brief Resize both per-thread arenas to the same element count.
    /// @param size New capacity in elements for both arenas.
    static void resize(int size) { BumpAllocator::resize(size, size); }

    /// @brief Return the current capacity of the per-thread scalar arena in elements.
    static int size_scalar() { return static_cast<int>(BumpAllocator::ScalarStack.size()); }

    /// @brief Return the current capacity of the per-thread super-scalar arena in elements.
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
                tuple_for_each(tspec.tspecs, [&](const auto &tspeci) {
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

} // namespace tycho::utils
