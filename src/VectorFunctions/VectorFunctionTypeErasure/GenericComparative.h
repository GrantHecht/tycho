#pragma once

#include "GenericConditional.h"
#include "VectorFunctions/CommonFunctions/CommonFunctions.h"
#include "pch.h"

namespace Tycho {

template <int IR> struct GenericComparative : rubber_types::TypeErasure<ConditionalSpec<IR>> {
    using Base = rubber_types::TypeErasure<ConditionalSpec<IR>>;

    GenericComparative() {}
    template <class T> GenericComparative(const T &t) : Base(t) {}
    GenericComparative(const GenericComparative<IR> &obj) {
        this->reset_container(obj.get_container());
    }

};

} // namespace Tycho
