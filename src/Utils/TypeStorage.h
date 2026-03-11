// =============================================================================
// New file in Tycho fork (Copyright 2026-present Grant R. Hecht,
//   Apache 2.0 — see LICENSE.txt)
//
// General-purpose SBO container with value semantics. Replaces
// rubber_types::TypeErasure for GenericConditional, ConstraintInterface,
// and ObjectiveInterface. GFStorage (GenericFunction) keeps shared_ptr.
//
// Convention: the base class C must declare
//   virtual void clone_into(TypeStorage<C, SBO_CAP>&) const = 0.
// Each concrete Model implements this as s.emplace<Model>(data_).
// =============================================================================

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

namespace Tycho {

template <typename C, std::size_t SBO_CAP = 128> class TypeStorage {
    static constexpr std::size_t SBO_ALIGN = alignof(std::max_align_t);
    enum class Kind : uint8_t { Empty, Inline, Heap };

    Kind kind_ = Kind::Empty;
    union {
        alignas(SBO_ALIGN) std::byte buf_[SBO_CAP];
        C *ptr_;
    };

    C *inline_ptr() noexcept { return reinterpret_cast<C *>(buf_); }
    const C *inline_ptr() const noexcept { return reinterpret_cast<const C *>(buf_); }

    void destroy() noexcept {
        if (kind_ == Kind::Inline) {
            inline_ptr()->~C();
        } else if (kind_ == Kind::Heap) {
            delete ptr_;
        }
        kind_ = Kind::Empty;
    }

  public:
    TypeStorage() noexcept = default;

    // Construct a Model(obj) in the SBO buffer (if it fits) or on the heap.
    // Model must derive from C and be constructible from T.
    template <typename Model, typename T> void emplace(T obj) {
        destroy();
        if constexpr (sizeof(Model) <= SBO_CAP && alignof(Model) <= SBO_ALIGN) {
            ::new (static_cast<void *>(buf_)) Model(std::move(obj));
            kind_ = Kind::Inline;
        } else {
            ptr_ = new Model(std::move(obj));
            kind_ = Kind::Heap;
        }
    }

    // Deep copy via virtual clone_into
    TypeStorage(const TypeStorage &o) : kind_(Kind::Empty) {
        if (!o.empty()) {
            o.get().clone_into(*this);
        }
    }

    // Move: memcpy for inline (vtable is stable), pointer steal for heap
    TypeStorage(TypeStorage &&o) noexcept : kind_(Kind::Empty) {
        if (o.kind_ == Kind::Inline) {
            std::memcpy(buf_, o.buf_, SBO_CAP);
            kind_ = Kind::Inline;
            o.kind_ = Kind::Empty;
        } else if (o.kind_ == Kind::Heap) {
            ptr_ = o.ptr_;
            kind_ = Kind::Heap;
            o.ptr_ = nullptr;
            o.kind_ = Kind::Empty;
        }
    }

    TypeStorage &operator=(const TypeStorage &o) {
        if (this != &o) {
            destroy();
            if (!o.empty()) {
                o.get().clone_into(*this);
            }
        }
        return *this;
    }

    TypeStorage &operator=(TypeStorage &&o) noexcept {
        if (this != &o) {
            destroy();
            if (o.kind_ == Kind::Inline) {
                std::memcpy(buf_, o.buf_, SBO_CAP);
                kind_ = Kind::Inline;
                o.kind_ = Kind::Empty;
            } else if (o.kind_ == Kind::Heap) {
                ptr_ = o.ptr_;
                kind_ = Kind::Heap;
                o.ptr_ = nullptr;
                o.kind_ = Kind::Empty;
            }
        }
        return *this;
    }

    ~TypeStorage() { destroy(); }

    [[nodiscard]] bool empty() const noexcept { return kind_ == Kind::Empty; }

    C &get() const noexcept {
        if (kind_ == Kind::Inline) {
            return *const_cast<C *>(inline_ptr());
        }
        return *ptr_;
    }

    C *get_ptr() const noexcept {
        if (kind_ == Kind::Inline) {
            return const_cast<C *>(inline_ptr());
        }
        return ptr_;
    }
};

// Concept: C must support clone_into(TypeStorage<C,N>&) const
template <typename C, std::size_t N>
concept TypeStorageBase = requires(const C &c, TypeStorage<C, N> &s) { c.clone_into(s); };

} // namespace Tycho
