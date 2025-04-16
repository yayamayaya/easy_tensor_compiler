#include "tensor.hpp"
#include <algorithm>
#include <iostream>

tensor
tensor::transpose()
{
    tensor res(std::vector<number_t>{});
    res.set_tensor_size(size);

    for (index_t i = 0; i < size.N; i++)  
        for (index_t j = 0; j < size.C; j++)
            for (index_t k = 0; k < size.H; k++)
                for (index_t r = 0; r < size.W; r++)
                    res(i, j, k, r) = (*this)(i, j, r, k);    

    std::cout << "res: " << res << std::endl;

    return res;
}