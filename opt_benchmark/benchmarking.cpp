#include "benchmarking.hpp"
#include <random>
#include "tensor.hpp"

void bench::generate_two_tensors()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::uniform_int_distribution<size_t>  size(1, tensor_dim_max_size);
    std::uniform_real_distribution<double> val((-1) * tensor_max_value, tensor_max_value);
    
    size_t N    = size(gen);
    size_t C    = size(gen);
    size_t H1   = size(gen);
    size_t W1H2 = size(gen);
    size_t W2   = size(gen);
    
    std::vector<number_t> data1 = {};
    for (index_t i = 0; i < N*C*H1*W1H2; i++)
    data1.push_back(val(gen));
    
    std::vector<number_t> data2 = {};
    for (index_t i = 0; i < N*C*W1H2*W2; i++)
    data2.push_back(val(gen));
    
    lhs = {N, C, H1, W1H2, data1};
    rhs = {N, C, W1H2, W2, data2};    
}

void bench::simple_mult_bench(benchmark::State& state)
{
    for (auto _ : state)
        lhs.simple_mul(rhs);
}

BENCHMARK(bench::simple_mult_bench);

void bench::cache_friendly_mult_bench(benchmark::State& state)
{
    for (auto _ : state)
        lhs.cache_friendly_mul(rhs);
}

BENCHMARK(bench::cache_friendly_mult_bench);

void bench::tiling_mult_bench(benchmark::State& state)
{
    for (auto _ : state)
        lhs.tiling_mul(rhs);
}

BENCHMARK(bench::tiling_mult_bench);

void bench::optimized_mult_bench(benchmark::State& state)
{
    for (auto _ : state);
}

int main(int argc, char *argv[])
{
    bench::generate_two_tensors();

    benchmark::Initialize(&argc, argv);

    benchmark::RunSpecifiedBenchmarks();

    benchmark::Shutdown();

    return 0;
}
