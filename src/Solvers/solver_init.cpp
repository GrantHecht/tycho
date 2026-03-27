#include "SolverInit.h"

#include <mutex>

#include "Utils/Timer.h"

#ifdef USE_ACCELERATE_SPARSE
#include "AccelerateUtils.h"
#else
#include <mkl.h>
#endif

namespace Tycho {

double ensure_solver_initialized() {
    static std::once_flag flag;
    static double time_ms = 0.0;
    std::call_once(flag, []() {
        Utils::Timer t;
        t.start();
#ifdef USE_ACCELERATE_SPARSE
        ensure_accelerate_initialized(TYCHO_DEFAULT_QP_THREADS);
#else
        dsecnd();
#endif
        t.stop();
        time_ms = double(t.count<std::chrono::microseconds>()) / 1000.0;
    });
    return time_ms;
}

} // namespace Tycho
