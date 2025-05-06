#ifndef BENCHMARK_INCLUDE_GUARD
#define BENCHMARK_INCLUDE_GUARD

#define BENCHMARK_HAS_NO_MAIN
#include "benchmark/benchmark.h"
#include "tensor.hpp"

class bench
{
public:

    static constexpr size_t tensor_dim_max_size = 100;

    static constexpr number_t tensor_max_value  = 5000;

    static void generate_two_tensors(const size_t size);

    static inline tensor lhs;

    static inline tensor rhs;

    static void simple_mult_bench(benchmark::State& state);

    static void cache_friendly_mult_bench(benchmark::State& state);
    
    static void tiling_mult_bench(benchmark::State& state);
    
    static void optimized_mult_bench(benchmark::State& state);
};

#endif