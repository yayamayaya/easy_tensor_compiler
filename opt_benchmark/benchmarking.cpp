#include "benchmarking.hpp"
#include <random>
#include <iostream>
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

    std::vector<number_t> data3 = {};
    for (index_t i = 0; i < (size / 2) * (size / 2); i++)
    data3.push_back(val(gen));
    
    lhs = {1, 1, size, size, data1};
    rhs = {1, 1, size, size, data2};    
    conv_filter = {1, 1, size / 2, size / 2, data3};
}

void bench::simple_mult_bench(benchmark::State& state)
{
    for (auto _ : state)
        lhs.simple_mul(rhs);
}

BENCHMARK(bench::simple_mult_bench)->Unit(benchmark::kMillisecond);

void bench::cache_friendly_mult_bench(benchmark::State& state)
{
    for (auto _ : state)
        lhs.cache_friendly_mul(rhs);
}

BENCHMARK(bench::cache_friendly_mult_bench)->Unit(benchmark::kMillisecond);

void bench::tiling_mult_bench(benchmark::State& state)
{
    for (auto _ : state)
        lhs.tiling_mul(rhs);
}

BENCHMARK(bench::tiling_mult_bench)->Unit(benchmark::kMillisecond);

void bench::optimized_mult_bench(benchmark::State& state)
{
    for (auto _ : state)
        lhs * rhs;
}

BENCHMARK(bench::optimized_mult_bench)->Unit(benchmark::kMillisecond);

void bench::simple_conv_bench(benchmark::State& state)
{
    for (auto _ : state)
        lhs.simple_conv(conv_filter);
}

BENCHMARK(bench::simple_conv_bench)->Unit(benchmark::kMillisecond);

void bench::optimized_conv_bench(benchmark::State& state)
{
    for (auto _ : state)
        lhs / conv_filter;
}

BENCHMARK(bench::optimized_conv_bench)->Unit(benchmark::kMillisecond);

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "> Please, input testing matrix size\n";
        return 1;
    }

    bench::generate_two_tensors(atoi(argv[1]));

    benchmark::Initialize(&argc, argv);

    benchmark::RunSpecifiedBenchmarks();

    benchmark::Shutdown();

    return 0;
}
