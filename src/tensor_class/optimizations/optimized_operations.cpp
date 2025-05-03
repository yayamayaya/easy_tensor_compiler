#include <algorithm>
#include <immintrin.h>
#include <x86intrin.h>
#include <iostream>
#include "tensor.hpp"

tensor
tensor::transpose() const
{
    tensor res(std::vector<number_t>{});
    res.set_tensor_size(size.N, size.C, size.W, size.H);

    for (index_t i = 0; i < size.N; i++)  
        for (index_t j = 0; j < size.C; j++)
            for (index_t k = 0; k < size.H; k++)
                for (index_t r = 0; r < size.W; r++)
                        res(i, j, r, k) = (*this)(i, j, k, r);                

    return res;
}

tensor
tensor::simple_mul(const tensor& rhs) const
{
    if ((size.N != rhs.size.N) || (size.C != rhs.size.C))
        throw std::logic_error("tensor batch number and matrice number doesn't match");
    
    if (size.W != rhs.size.H)
        throw std::logic_error("tensor sizes are not compatible");
    
    std::vector<number_t> new_tensor;
        
    for (index_t i = 0; i < size.N; i++)
        for (index_t j = 0; j < size.C; j++)
            for (index_t l = 0; l < size.H; l++)
                for (index_t k = 0; k < rhs.size.W; k++)
                {
                    number_t val = 0;
                    
                    for (index_t r = 0; r < size.W; r++)
                        val += (*this)(i, j, l, r) * rhs(i, j, r, k);

                    new_tensor.push_back(val);
                }
    
    tensor res(new_tensor);
    res.set_tensor_size(size.N, size.C, size.H, rhs.size.W);

    return res;
}

tensor
tensor::cache_friendly_mul(const tensor& rhs) const
{
    if ((size.N != rhs.size.N) || (size.C != rhs.size.C))
    throw std::logic_error("tensor batch number and matrice number doesn't match");

    if (size.W != rhs.size.H)
        throw std::logic_error("tensor sizes are not compatible");

    tensor rhs_transposed = rhs.transpose();

    std::vector<number_t> new_tensor;
        
    for (index_t i = 0; i < size.N; i++)
        for (index_t j = 0; j < size.C; j++)
            for (index_t l = 0; l < size.H; l++)
                for (index_t k = 0; k < rhs.size.W; k++)
                {
                    number_t val = 0;
                    
                    for (index_t r = 0; r < size.W; r++)
                        val += this->data[] * rhs_transposed(i, j, k, r);

                    new_tensor.push_back(val);
                }

    tensor res(new_tensor);
    res.set_tensor_size(size.N, size.C, size.H, rhs.size.W);

    return res; 
}

tensor
tensor::tiling_mul(const tensor& rhs) const
{
    if ((size.N != rhs.size.N) || (size.C != rhs.size.C))
    throw std::logic_error("tensor batch number and matrice number doesn't match");

    if (size.W != rhs.size.H)
        throw std::logic_error("tensor sizes are not compatible");

    tensor rhs_transposed = rhs.transpose();

    tensor res(std::vector<number_t>{});
    res.set_tensor_size(size.N, size.C, size.H, rhs.size.W);
        
    for (index_t i = 0; i < size.N; i++)
        for (index_t j = 0; j < size.C; j++)
            for (index_t t0 = 0; t0 < size.H; t0 += tile_size)
                for (index_t t1 = 0; t1 < rhs.size.W; t1 += tile_size)
                    for (index_t t2 = 0; t2 < size.W; t2 += tile_size)
                        for (index_t l = 0; l < std::min(t0 + tile_size, size.H); l++)
                            for (index_t k = 0; k < std::min(t1 + tile_size, rhs.size.W); k++)
                                for (index_t r = 0; r < std::min(t2 + tile_size, size.W); r++)
                                    res(i, j, l, k) += (*this)(i, j, l, r) * rhs_transposed(i, j, k, r);   

    return res; 
}


//Полностью оптимизоравнное умножение матриц
tensor
tensor::operator *(const tensor& rhs) const
{
    if ((size.N != rhs.size.N) || (size.C != rhs.size.C))
    throw std::logic_error("tensor batch number and matrice number doesn't match");

    if (size.W != rhs.size.H)
        throw std::logic_error("tensor sizes are not compatible");

    tensor rhs_transposed = rhs.transpose();

    std::vector<number_t> new_tensor;
        
    for (index_t i = 0; i < size.N; i++)
        for (index_t j = 0; j < size.C; j++)
            for (index_t l = 0; l < size.H; l++)
                for (index_t k = 0; k < rhs.size.W; k++)
                {
                    number_t val = 0;
                    
                    for (index_t r = 0; r < size.W; r++)
                        val += (*this)(i, j, l, r) * rhs_transposed(i, j, k, r);

                    new_tensor.push_back(val);
                }

    tensor res(new_tensor);
    res.set_tensor_size(size.N, size.C, size.H, rhs.size.W);

    return res; 
}