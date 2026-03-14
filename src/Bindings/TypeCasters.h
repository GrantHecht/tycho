#pragma once

#include "InterfaceTypes.h"
#include <nanobind/ndarray.h>
#include <unsupported/Eigen/CXX11/Tensor>

// These must be included before any type_caster body that references
// make_caster<std::string> or make_caster<std::vector<T>>, to prevent the
// compiler from implicitly instantiating those specializations from the primary
// template before our explicit specializations are defined.
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

// ============================================================================
// nanobind type casters for Tycho interface types.
//
// Include this header in any binding .cpp that registers functions whose
// signatures contain these types, so that nanobind can automatically convert
// Python objects to/from the corresponding C++ variants.
// ============================================================================

namespace nanobind {
namespace detail {

// ---------------------------------------------------------------------------
// Eigen::Tensor<Scalar, N>
//
// Python numpy array (C-contiguous, CPU) <-> Eigen::Tensor.
// Data is copied in both directions; Eigen::Tensor owns its buffer.
// ---------------------------------------------------------------------------
template <typename Scalar, int N> struct type_caster<Eigen::Tensor<Scalar, N>> {
    // NB_TYPE_CASTER cannot handle template types with commas; use alias.
    using TensorType = Eigen::Tensor<Scalar, N>;
    NB_TYPE_CASTER(TensorType, const_name("numpy.ndarray"))

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
        using Array = nb::ndarray<Scalar, nb::c_contig, nb::device::cpu>;
        auto caster = make_caster<Array>();
        if (!caster.from_python(src, flags, cleanup))
            return false;
        Array arr = (Array &&)caster.value;
        if ((int)arr.ndim() != N)
            return false;
        Eigen::array<Eigen::Index, N> dims;
        for (int i = 0; i < N; ++i)
            dims[i] = (Eigen::Index)arr.shape(i);
        value.resize(dims);
        std::memcpy(value.data(), arr.data(), value.size() * sizeof(Scalar));
        return true;
    }

    static handle from_cpp(const Eigen::Tensor<Scalar, N> &tensor, rv_policy policy,
                           cleanup_list *cleanup) {
        size_t shape[N];
        for (int i = 0; i < N; ++i)
            shape[i] = (size_t)tensor.dimension(i);
        Scalar *data = new Scalar[tensor.size()];
        std::memcpy(data, tensor.data(), tensor.size() * sizeof(Scalar));
        nb::capsule deleter(data, [](void *p) noexcept { delete[] (Scalar *)p; });
        return nb::ndarray<nb::numpy, Scalar>(data, N, shape, deleter).release();
    }
};

// ---------------------------------------------------------------------------
// Eigen::VectorXd  (= Eigen::Matrix<double, -1, 1>)
//
// Explicit full specialization: takes priority over nanobind's partial Eigen
// specialization.  Accepts numpy arrays (existing behaviour) AND any Python
// sequence of numeric scalars (list, tuple, range-valued, …).
//
// IMPORTANT: this must appear before any type_caster body that calls
// make_caster<Eigen::VectorXd> (e.g. VarIndexType, ScaleType), to prevent an
// implicit instantiation from the Eigen partial spec preceding this definition.
// ---------------------------------------------------------------------------
template <> struct type_caster<Eigen::VectorXd> {
    NB_TYPE_CASTER(Eigen::VectorXd, const_name("numpy.ndarray"))

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
        // First: try nanobind's ndarray caster (numpy arrays).
        using Array = nb::ndarray<double, nb::c_contig, nb::device::cpu>;
        auto caster = make_caster<Array>();
        if (caster.from_python(src, flags & ~(uint8_t)nb::detail::cast_flags::accepts_none,
                               cleanup)) {
            const Array &arr = caster.value;
            if (arr.ndim() != 1)
                return false;
            value.resize((Eigen::Index)arr.shape(0));
            std::memcpy(value.data(), arr.data(), arr.shape(0) * sizeof(double));
            return true;
        }
        // Fallback: Python sequence of numeric scalars (list, tuple, …).
        PyErr_Clear();
        PyObject *seq = PySequence_Fast(src.ptr(), "");
        if (!seq) {
            PyErr_Clear();
            return false;
        }
        Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
        value.resize(n);
        bool ok = true;
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject *f = PyNumber_Float(PySequence_Fast_GET_ITEM(seq, i));
            if (!f) {
                PyErr_Clear();
                ok = false;
                break;
            }
            value[i] = PyFloat_AS_DOUBLE(f);
            Py_DECREF(f);
        }
        Py_DECREF(seq);
        return ok;
    }

    static handle from_cpp(const Eigen::VectorXd &src, rv_policy /*policy*/,
                           cleanup_list *cleanup) {
        // rv_policy is intentionally ignored: nanobind cannot safely share an
        // Eigen::VectorXd's internal buffer with Python's GC (no owner object).
        // We always copy. Use Eigen::Ref<> parameters on the C++ side to avoid
        // the copy on the input path instead.
        using Array = nb::ndarray<nb::numpy, double>;
        size_t shape[1] = {(size_t)src.size()};
        double *data = new double[src.size()];
        std::memcpy(data, src.data(), src.size() * sizeof(double));
        nb::capsule deleter(data, [](void *p) noexcept { delete[] (double *)p; });
        Array arr(data, 1, shape, deleter);
        return make_caster<Array>::from_cpp(arr, rv_policy::take_ownership, cleanup);
    }
};

// ---------------------------------------------------------------------------
// Eigen::VectorXi  (= Eigen::Matrix<int, -1, 1>)
//
// Explicit full specialization: accepts numpy int32 arrays (existing behaviour)
// AND any Python sequence of integers (list, tuple, range, …).
//
// IMPORTANT: this must appear before VarIndexType (which calls
// make_caster<Eigen::VectorXi> in its from_cpp std::visit body).
// ---------------------------------------------------------------------------
template <> struct type_caster<Eigen::VectorXi> {
    NB_TYPE_CASTER(Eigen::VectorXi, const_name("numpy.ndarray"))

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
        // First: try nanobind's ndarray caster (numpy int32 arrays).
        using Scalar = Eigen::VectorXi::Scalar;
        using Array = nb::ndarray<Scalar, nb::c_contig, nb::device::cpu>;
        auto caster = make_caster<Array>();
        if (caster.from_python(src, flags & ~(uint8_t)nb::detail::cast_flags::accepts_none,
                               cleanup)) {
            const Array &arr = caster.value;
            if (arr.ndim() != 1)
                return false;
            value.resize((Eigen::Index)arr.shape(0));
            std::memcpy(value.data(), arr.data(), arr.shape(0) * sizeof(Scalar));
            return true;
        }
        // Fallback: Python sequence of integers (list, tuple, range, …).
        PyErr_Clear();
        PyObject *seq = PySequence_Fast(src.ptr(), "");
        if (!seq) {
            PyErr_Clear();
            return false;
        }
        Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
        value.resize(n);
        bool ok = true;
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject *as_long = PyNumber_Long(PySequence_Fast_GET_ITEM(seq, i));
            if (!as_long) {
                PyErr_Clear();
                ok = false;
                break;
            }
            value[i] = (int)PyLong_AsLong(as_long);
            Py_DECREF(as_long);
        }
        Py_DECREF(seq);
        return ok;
    }

    static handle from_cpp(const Eigen::VectorXi &src, rv_policy /*policy*/,
                           cleanup_list *cleanup) {
        // rv_policy is intentionally ignored: nanobind cannot safely share an
        // Eigen::VectorXi's internal buffer with Python's GC (no owner object).
        // We always copy. Use Eigen::Ref<> parameters on the C++ side to avoid
        // the copy on the input path instead.
        using Scalar = Eigen::VectorXi::Scalar;
        static_assert(sizeof(Scalar) == sizeof(int32_t),
                      "VectorXi::Scalar must be 32-bit on this platform");
        using Array = nb::ndarray<nb::numpy, int32_t>; // int32_t guarantees np.int32 dtype
        size_t shape[1] = {(size_t)src.size()};
        int32_t *data = new int32_t[src.size()];
        std::memcpy(data, src.data(), src.size() * sizeof(int32_t));
        nb::capsule deleter(data, [](void *p) noexcept { delete[] (int32_t *)p; });
        Array arr(data, 1, shape, deleter);
        return make_caster<Array>::from_cpp(arr, rv_policy::take_ownership, cleanup);
    }
};

// ---------------------------------------------------------------------------
// VarIndexType  =  std::variant<int, Eigen::VectorXi, std::string, std::vector<std::string>>
//
// Python int    -> int
// Python str    -> std::string
// Python list of str -> std::vector<std::string>
// Python sequence of int (list, range, tuple, numpy int array) -> Eigen::VectorXi
// ---------------------------------------------------------------------------
template <> struct type_caster<Tycho::VarIndexType> {
    NB_TYPE_CASTER(Tycho::VarIndexType, const_name("int | Sequence[int] | str | list[str]"))

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
        // int
        if (PyLong_Check(src.ptr())) {
            long v = PyLong_AsLong(src.ptr());
            if (!PyErr_Occurred()) {
                value = (int)v;
                return true;
            }
            PyErr_Clear();
        }
        // string
        if (PyUnicode_Check(src.ptr())) {
            Py_ssize_t sz;
            const char *s = PyUnicode_AsUTF8AndSize(src.ptr(), &sz);
            if (s) {
                value = std::string(s, sz);
                return true;
            }
            PyErr_Clear();
        }
        // Try any iterable (list, tuple, range, numpy array) via PySequence_Fast.
        // Note: PySequence_Check returns 0 for range objects (they use mp_subscript
        // rather than sq_item), so we skip that check and call PySequence_Fast directly.
        PyErr_Clear(); // clear any pending error before C API calls
        PyObject *seq = PySequence_Fast(src.ptr(), "");
        if (!seq) {
            PyErr_Clear();
            return false;
        }
        Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
        if (n == 0) {
            Py_DECREF(seq);
            value = Eigen::VectorXi(0);
            return true;
        }
        PyObject *first = PySequence_Fast_GET_ITEM(seq, 0); // borrowed
        // List of strings?
        if (PyUnicode_Check(first)) {
            std::vector<std::string> vs;
            bool ok = true;
            for (Py_ssize_t i = 0; i < n; ++i) {
                PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
                if (!PyUnicode_Check(item)) {
                    ok = false;
                    break;
                }
                Py_ssize_t ssz;
                const char *ss = PyUnicode_AsUTF8AndSize(item, &ssz);
                if (!ss) {
                    PyErr_Clear();
                    ok = false;
                    break;
                }
                vs.emplace_back(ss, ssz);
            }
            Py_DECREF(seq);
            if (ok) {
                value = std::move(vs);
                return true;
            }
            return false;
        }
        // Sequence of ints -> VectorXi
        {
            Eigen::VectorXi v(n);
            bool ok = true;
            for (Py_ssize_t i = 0; i < n; ++i) {
                PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
                PyObject *as_long = PyNumber_Long(item);
                if (!as_long) {
                    PyErr_Clear();
                    ok = false;
                    break;
                }
                v[i] = (int)PyLong_AsLong(as_long);
                Py_DECREF(as_long);
            }
            Py_DECREF(seq);
            if (ok) {
                value = std::move(v);
                return true;
            }
        }
        return false;
    }

    static handle from_cpp(const Tycho::VarIndexType &src, rv_policy policy,
                           cleanup_list *cleanup) {
        return std::visit(
            [&](const auto &v) -> handle {
                return make_caster<std::decay_t<decltype(v)>>::from_cpp(v, policy, cleanup);
            },
            src);
    }
};

// ---------------------------------------------------------------------------
// std::vector<Eigen::VectorXd>
//
// Python list of numpy arrays OR list of sequences (list-of-lists, list-of-tuples)
// -> std::vector<Eigen::VectorXd>.
// Returned from C++ as a Python list of 1-D numpy arrays.
// ---------------------------------------------------------------------------
template <> struct type_caster<std::vector<Eigen::VectorXd>> {
    using T = std::vector<Eigen::VectorXd>;
    NB_TYPE_CASTER(T, const_name("list[numpy.ndarray]"))

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
        PyObject *seq = PySequence_Fast(src.ptr(), "");
        if (!seq) {
            PyErr_Clear();
            return false;
        }
        Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
        value.resize((size_t)n);
        for (Py_ssize_t i = 0; i < n; ++i) {
            handle item(PySequence_Fast_GET_ITEM(seq, i)); // borrowed ref
            // First: try nanobind's Eigen VectorXd caster (handles numpy arrays).
            auto caster = make_caster<Eigen::VectorXd>();
            if (caster.from_python(item, flags, cleanup)) {
                value[i] = std::move(caster.value);
                continue;
            }
            // Fallback: treat element as a Python sequence of scalars.
            PyErr_Clear();
            PyObject *sub = PySequence_Fast(item.ptr(), "");
            if (!sub) {
                PyErr_Clear();
                Py_DECREF(seq);
                return false;
            }
            Py_ssize_t m = PySequence_Fast_GET_SIZE(sub);
            value[i].resize(m);
            bool ok = true;
            for (Py_ssize_t j = 0; j < m; ++j) {
                PyObject *f = PyNumber_Float(PySequence_Fast_GET_ITEM(sub, j));
                if (!f) {
                    PyErr_Clear();
                    ok = false;
                    break;
                }
                value[i][j] = PyFloat_AS_DOUBLE(f);
                Py_DECREF(f);
            }
            Py_DECREF(sub);
            if (!ok) {
                Py_DECREF(seq);
                return false;
            }
        }
        Py_DECREF(seq);
        return true;
    }

    static handle from_cpp(const T &src, rv_policy /*policy*/, cleanup_list *cleanup) {
        // Each element is an independent Python object in a new list; always copy.
        PyObject *list = PyList_New((Py_ssize_t)src.size());
        if (!list)
            return handle();
        for (Py_ssize_t i = 0; i < (Py_ssize_t)src.size(); ++i) {
            handle h =
                make_caster<Eigen::VectorXd>::from_cpp(src[i], rv_policy::take_ownership, cleanup);
            if (!h) {
                Py_DECREF(list);
                return handle();
            }
            PyList_SET_ITEM(list, i, h.ptr()); // steals reference
        }
        return handle(list);
    }
};

// ---------------------------------------------------------------------------
// std::variant<double, Eigen::VectorXd>  (inline value type for boundary values)
//
// Python float/int -> double
// Python numpy array -> Eigen::VectorXd
// Python sequence of float (list, tuple) -> Eigen::VectorXd
// ---------------------------------------------------------------------------
template <> struct type_caster<std::variant<double, Eigen::VectorXd>> {
    using T = std::variant<double, Eigen::VectorXd>;
    NB_TYPE_CASTER(T, const_name("float | numpy.ndarray | Sequence[float]"))

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
        // float
        if (PyFloat_Check(src.ptr())) {
            value = PyFloat_AS_DOUBLE(src.ptr());
            return true;
        }
        // int -> double
        if (PyLong_Check(src.ptr())) {
            long v = PyLong_AsLong(src.ptr());
            if (!PyErr_Occurred()) {
                value = (double)v;
                return true;
            }
            PyErr_Clear();
        }
        // numpy array -> VectorXd (fast path via no-throw make_caster API)
        {
            auto vc = make_caster<Eigen::VectorXd>();
            if (vc.from_python(src, flags, cleanup)) {
                value = std::move(vc.value);
                return true;
            }
        }
        // Python sequence (list, tuple, range, numpy array) -> VectorXd
        {
            PyObject *seq = PySequence_Fast(src.ptr(), "");
            if (seq) {
                Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
                Eigen::VectorXd v(n);
                bool ok = true;
                for (Py_ssize_t i = 0; i < n; ++i) {
                    PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
                    PyObject *f = PyNumber_Float(item);
                    if (!f) {
                        PyErr_Clear();
                        ok = false;
                        break;
                    }
                    v[i] = PyFloat_AS_DOUBLE(f);
                    Py_DECREF(f);
                }
                Py_DECREF(seq);
                if (ok) {
                    value = std::move(v);
                    return true;
                }
            } else {
                PyErr_Clear();
            }
        }
        return false;
    }

    static handle from_cpp(const T &src, rv_policy policy, cleanup_list *cleanup) {
        return std::visit(
            [&](const auto &v) -> handle {
                return make_caster<std::decay_t<decltype(v)>>::from_cpp(v, policy, cleanup);
            },
            src);
    }
};

// ---------------------------------------------------------------------------
// ScaleType  =  std::variant<double, Eigen::VectorXd, Tycho::ScaleModes, std::string>
//
// Python  None         -> Tycho::ScaleModes::NONE
// Python  ScaleModes   -> Tycho::ScaleModes (enum value)
// Python  str          -> std::string
// Python  float        -> double
// Python  array        -> Eigen::VectorXd
// ---------------------------------------------------------------------------
template <> struct type_caster<Tycho::ScaleType> {
    NB_TYPE_CASTER(Tycho::ScaleType,
                   const_name("Union[float, numpy.ndarray, OptimalControl.ScaleModes, str, None]"))

    bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
        if (src.is_none()) {
            value = Tycho::ScaleModes::NONE;
            return true;
        }
        // ScaleModes enum (nb::enum_<ScaleModes>)
        {
            auto sc = make_caster<Tycho::ScaleModes>();
            if (sc.from_python(src, flags, cleanup)) {
                value = sc.value;
                return true;
            }
        }
        if (PyUnicode_Check(src.ptr())) {
            value = nb::cast<std::string>(src);
            return true;
        }
        if (PyFloat_Check(src.ptr()) || PyLong_Check(src.ptr())) {
            value = nb::cast<double>(src);
            return true;
        }
        {
            auto vc = make_caster<Eigen::VectorXd>();
            if (vc.from_python(src, flags, cleanup)) {
                value = std::move(vc.value);
                return true;
            }
        }
        return false;
    }

    static handle from_cpp(Tycho::ScaleType src, rv_policy policy, cleanup_list *cleanup) {
        return std::visit(
            [&](auto &&v) -> handle {
                using T = std::decay_t<decltype(v)>;
                return make_caster<T>::from_cpp(std::forward<decltype(v)>(v), policy, cleanup);
            },
            src);
    }
};

} // namespace detail
} // namespace nanobind
