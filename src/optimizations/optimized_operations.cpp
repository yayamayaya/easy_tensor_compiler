#include "tensor.hpp"
#include <algorithm>

#define OPTIMIZED_TENSORS

tensor
tensor::transpose(const tensor& t)
{
    std::vector<number_t> transposed_data;

    for (index_t i = 0; i < size.N; i++)  
        for (index_t j = 0; j < size.C; j++)
            for (index_t k = 0; k < size.H; k++)
                for (index_t r = 0; r < size.W; r++)
                    std::swap((*this)(i, j, k, r), (*this)(i, j, r, k));
    
    tensor res(transposed_data);
    res.set_tensor_size(size);

    return res;
}

