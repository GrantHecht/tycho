#include "MKLInit.h"

#ifndef USE_ACCELERATE_SPARSE
#include <mkl.h>
#include <mutex>

#include "Utils/Timer.h"
#endif

namespace Tycho {

double ensure_mkl_initialized() {
#ifndef USE_ACCELERATE_SPARSE
    static std::once_flag flag;
    static double time_ms = 0.0;
    std::call_once(flag, []() {
        Utils::Timer t;
        t.start();
        dsecnd();
        t.stop();
        time_ms = double(t.count<std::chrono::microseconds>()) / 1000.0;
    });
    return time_ms;
#else
    return 0.0;
#endif
}

} // namespace Tycho
