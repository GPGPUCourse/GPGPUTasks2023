#pragma once

#include <array>
#include <iostream>
#include <string>
#include <vector>

#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <numeric>

#include "cl/sum_cl.h"
#include "error_handler.h"
#include "libgpu/work_size.h"


namespace utils {
    struct Params {
        const std::string &device;
        std::vector<unsigned int> &as;
        unsigned int benchmarking_iters;
        unsigned int expect;
        unsigned int src_n;
    };

    struct ParamsUsually : public Params {
        ParamsUsually(const std::string &device, std::vector<unsigned int> &as, unsigned int benchmarking_iters,
                      unsigned int expect, unsigned int src_n)
            : Params{device, as, benchmarking_iters, expect, src_n} {
        }
    };

    struct ParamsTree : public Params {
        unsigned int res_n;

        ParamsTree(const std::string &device, std::vector<unsigned int> &as, unsigned int benchmarking_iters,
                   unsigned int expect, unsigned int src_n, unsigned int res_n)
            : Params{device, as, benchmarking_iters, expect, src_n}, res_n(res_n) {
        }
    };

    namespace {
        inline void printAnswer(const timer &t, const std::string &msg, const unsigned int n) {
            std::cout << msg << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << msg << ": " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
            std::cout << std::endl;
        }

        inline void initMem(gpu::gpu_mem_32u &src_gpu, gpu::gpu_mem_32u &res_gpu, const ParamsUsually &params) {
            src_gpu.resizeN(params.src_n);
            src_gpu.writeN(params.as.data(), params.src_n);
            res_gpu.resizeN(1);
        }

        inline void initMem(gpu::gpu_mem_32u &src_gpu, gpu::gpu_mem_32u &res_gpu, const ParamsTree &params) {
            src_gpu.resizeN(params.src_n);
            src_gpu.writeN(params.as.data(), params.src_n);
            res_gpu.resizeN(params.res_n);
        }
    }// namespace

    inline void sum(const gpu::WorkSize &work_size, const ParamsUsually &params, const std::string &msg,
                    const std::string &func) {
        ocl::Kernel adder(sum_kernel, sum_kernel_length, func);
        adder.compile();
        gpu::gpu_mem_32u src_gpu, res_gpu;
        initMem(src_gpu, res_gpu, params);

        timer t;
        for (int iter = 0; iter < params.benchmarking_iters; ++iter) {
            unsigned int sum = 0;
            res_gpu.writeN(&sum, 1);
            adder.exec(work_size, src_gpu, res_gpu, params.src_n);
            res_gpu.readN(&sum, 1, 0);
            eh::EXPECT_THE_SAME(params.expect, sum, "the \"" + func + "\" method does not sum correctly!");
            t.nextLap();
        }
        printAnswer(t, msg, params.src_n);
    }

    inline void sum(const gpu::WorkSize &work_size, const ParamsTree &params, const std::string &msg,
                    const std::string &func) {
        ocl::Kernel adder(sum_kernel, sum_kernel_length, func);
        adder.compile();

        gpu::gpu_mem_32u src_gpu, res_gpu;
        initMem(src_gpu, res_gpu, params);

        timer t;
        for (int iter = 0; iter < params.benchmarking_iters; ++iter) {
            std::vector<unsigned int> sum(params.res_n, 0);
            res_gpu.writeN(sum.data(), params.res_n);
            adder.exec(work_size, src_gpu, res_gpu, params.src_n);
            res_gpu.readN(sum.data(), params.res_n, 0);
            sum[0] = std::accumulate(sum.begin(), sum.end(), 0);
            eh::EXPECT_THE_SAME(params.expect, sum[0], "the \"" + func + "\" method does not sum correctly!");
            t.nextLap();
        }
        printAnswer(t, msg, params.src_n);
    }
};// namespace utils