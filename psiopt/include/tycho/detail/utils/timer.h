#pragma once
#include <chrono>
/*
Copyright (c) 2017 André L. Maravilha
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

namespace tycho::utils {

/// @brief Simple wall-clock timer with start/stop/reset semantics.
///
/// Wraps `std::chrono::steady_clock`. Elapsed time accumulates across multiple
/// start/stop cycles. Retrieve it via `count<duration_t>()`.
class Timer {

  public:
    /// @brief Construct a timer, optionally starting it immediately.
    /// @param start If true, the timer begins running immediately after construction.
    Timer(bool start = false)
        : started_(false), paused_(false), reference_(std::chrono::steady_clock::now()),
          accumulated_(std::chrono::duration<long double>(0)) {
        if (start) {
            this->start();
        }
    }

    /// @brief Start or resume the timer.
    ///
    /// Has no effect if the timer is already running.
    void start() {
        if (!started_) {
            started_ = true;
            paused_ = false;
            accumulated_ = std::chrono::duration<long double>(0);
            reference_ = std::chrono::steady_clock::now();
        } else if (paused_) {
            reference_ = std::chrono::steady_clock::now();
            paused_ = false;
        }
    }

    /// @brief Stop (pause) the timer.
    ///
    /// Accumulates elapsed time since the last `start()`. Has no effect if the
    /// timer is not running.
    void stop() {
        if (started_ && !paused_) {
            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            accumulated_ =
                accumulated_ +
                std::chrono::duration_cast<std::chrono::duration<long double>>(now - reference_);
            paused_ = true;
        }
    }

    /// @brief Reset the timer to the initial stopped state.
    ///
    /// Clears all accumulated time. Has no effect if the timer has never been started.
    void reset() {
        if (started_) {
            started_ = false;
            paused_ = false;
            reference_ = std::chrono::steady_clock::now();
            accumulated_ = std::chrono::duration<long double>(0);
        }
    }

    /// @brief Return the elapsed time in the requested duration unit.
    ///
    /// @tparam duration_t `std::chrono` duration type for the return value;
    ///         defaults to `std::chrono::milliseconds`.
    /// @return Elapsed time as `duration_t::rep`. Returns 0 if the timer has
    ///         never been started.
    template <class duration_t = std::chrono::milliseconds> typename duration_t::rep count() const {
        if (started_) {
            if (paused_) {
                return std::chrono::duration_cast<duration_t>(accumulated_).count();
            } else {
                return std::chrono::duration_cast<duration_t>(
                           accumulated_ + (std::chrono::steady_clock::now() - reference_))
                    .count();
            }
        } else {
            return duration_t(0).count();
        }
    }

  private:
    bool started_;
    bool paused_;
    std::chrono::steady_clock::time_point reference_;
    std::chrono::duration<long double> accumulated_;
};

} // namespace tycho::utils
