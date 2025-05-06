#include "benchmarking.hpp"
#include <random>
#include "tensor.hpp"

void bench::generate_two_tensors(const size_t size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::uniform_real_distribution<double> val((-1) * tensor_max_value, tensor_max_value);
    
    std::vector<number_t> data1 = {};
    for (index_t i = 0; i < size * size; i++)
    data1.push_back(val(gen));
    
    std::vector<number_t> data2 = {};
    for (index_t i = 0; i < size * size; i++)
    data2.push_back(val(gen));
    
    lhs = {1, 1, size, size, data1};
    rhs = {1, 1, size, size, data2};    
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
    for (auto _ : state)
        lhs * rhs;
}

BENCHMARK(bench::optimized_mult_bench);

int main(int argc, char *argv[])
{
    bench::generate_two_tensors(200);

    benchmark::Initialize(&argc, argv);

    benchmark::RunSpecifiedBenchmarks();

    benchmark::Shutdown();

    return 0;
}
