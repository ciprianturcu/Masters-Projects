#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class Timer {
public:
    Timer() {
        reset();
    }
    void reset() {
        start = std::chrono::high_resolution_clock::now();
    }
    // Returns elapsed time in milliseconds.
    double elapsedMilliseconds() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start;
};

#endif // TIMER_H

