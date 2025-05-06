#include <algorithm>
#include <cstring>
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
                        val += (*this)(i, j, l, r) * rhs_transposed(i, j, k, r);

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
                    index_t start_ind_f = i * size.H * size.C * size.W + j * size.H * size.W + l * size.W;
                    index_t start_ind_s = i * size.H * size.C * size.W + j * size.H * size.W + k * size.W;
                    
                    new_tensor.push_back(vector_mult_sum(this->data.data() + start_ind_f, \
                        rhs_transposed.data.data() + start_ind_s, size.W));
                }

    tensor res(new_tensor);
    res.set_tensor_size(size.N, size.C, size.H, rhs.size.W);

    return res; 
}

number_t tensor::vector_mult_sum(const number_t *v1, const number_t *v2, const size_t size)
{
    number_t res = 0;

    number_t *f = get_padded(v1, size, VEC_SIZE);
    number_t *s = get_padded(v2, size, VEC_SIZE);

    for (size_t i = 0; i < size / VEC_SIZE + 1; i++)
    {
        __m512d vec1 = _mm512_loadu_pd(f + VEC_SIZE * i);
        __m512d vec2 = _mm512_loadu_pd(s + VEC_SIZE * i);
    
        res += _mm512_reduce_add_pd(_mm512_mul_pd(vec1, vec2));
    }

    delete [] f;
    delete [] s;

    return res;
}

number_t *tensor::get_padded(const number_t *ptr, const size_t size, const size_t divider)
{
    size_t new_size = size + divider - size % divider;

    number_t *new_ptr = new number_t [new_size];
    std::memset(new_ptr, 0, new_size * sizeof(number_t));
    std::memcpy(new_ptr, ptr, size * sizeof(number_t));

    return new_ptr;
}

// tensor 
// tensor::operator /(const tensor& rhs) const
// {
//     if ((size.N != rhs.size.N) || (size.C != rhs.size.C))
//         throw std::logic_error("tensor batch number and matrice number doesn't match");

//     if (!is_square() || !rhs.is_square())
//         throw std::logic_error("tensors are not square matrices for convolution");

//     std::vector<number_t> result;

//     for (index_t i = 0; i < size.N; i++)
//         for (index_t j = 0; j < size.C; j++)
//             for (index_t l = 0; l < rhs.size.H; l++)
//                 for (index_t k = 0; k < rhs.size.W; k++)
//                 {
//                     number_t val = 0;

//                     for (index_t r = 0; r < rhs.size.H; r++)
//                         for (index_t t = 0; t < rhs.size.W; t++)
//                             val += (*this)(i, j, l + r, k + t) * rhs(i, j, r, t);

//                     result.push_back(val);
//                 }
 
//     tensor res(result);
//     res.set_tensor_size(rhs.size);

//     return res;
// }