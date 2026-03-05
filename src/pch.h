#ifndef PCH_H
#define PCH_H

#include <math.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "TypeDefs/EigenTypes.h"
#include "Utils/EigenSTL.h"
#include "Utils/GetCoreCount.h"
#include "Utils/LambdaJumpTable.h"
#include "Utils/MathFunctions.h"
#include "Utils/STDExtensions.h"
#include "Utils/ThreadPool.h"
#include "Utils/Timer.h"
#include "Utils/TupleIterator.h"
#include "Utils/TypeErasure.h"
#include "Utils/TypeName.h"
#include "Utils/fmtlib.h"

// Python binding headers live in src/Bindings/pch_nb.h and are baked into the
// pch_bindings precompiled header used exclusively by _tycho and tycho_extensions.

#endif // PCH_H
