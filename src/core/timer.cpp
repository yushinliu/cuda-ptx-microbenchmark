#include "core/gpu_timer.h"

#include <stdexcept>
#include <string>

namespace cpm {

GpuTimer::GpuTimer() : running_(false) {
    cudaError_t err = cudaEventCreate(&start_event_);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to create start event: ") +
                                 cudaGetErrorString(err));
    }
    err = cudaEventCreate(&stop_event_);
    if (err != cudaSuccess) {
        cudaEventDestroy(start_event_);
        throw std::runtime_error(std::string("Failed to create stop event: ") +
                                 cudaGetErrorString(err));
    }
}

GpuTimer::~GpuTimer() {
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
}

GpuTimer::GpuTimer(GpuTimer&& other) noexcept
    : start_event_(other.start_event_)
    , stop_event_(other.stop_event_)
    , running_(other.running_) {
    other.start_event_ = nullptr;
    other.stop_event_ = nullptr;
}

GpuTimer& GpuTimer::operator=(GpuTimer&& other) noexcept {
    if (this != &other) {
        cudaEventDestroy(start_event_);
        cudaEventDestroy(stop_event_);

        start_event_ = other.start_event_;
        stop_event_ = other.stop_event_;
        running_ = other.running_;

        other.start_event_ = nullptr;
        other.stop_event_ = nullptr;
    }
    return *this;
}

void GpuTimer::start() {
    cudaEventRecord(start_event_);
    running_ = true;
}

void GpuTimer::stop() {
    cudaEventRecord(stop_event_);
    cudaEventSynchronize(stop_event_);
    running_ = false;
}

float GpuTimer::elapsed_ms() const {
    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, start_event_, stop_event_);
    return elapsed;
}

float GpuTimer::elapsed_us() const {
    return elapsed_ms() * 1000.0f;
}

void GpuTimer::reset() {
    running_ = false;
}

}  // namespace cpm
