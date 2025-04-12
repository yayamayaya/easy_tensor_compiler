#include "tensor.hpp"
#include <exception>

number_t& 
tensor::operator ()(const size_t n, const size_t c, const size_t h, const size_t w)
{ 
    index_t ind = n * H * C * W + c * H * W + h * W + w;
    if (ind > data.size())
        throw std::out_of_range("attempt to access not allocated memory");

    return data[ind];
}

number_t 
tensor::operator ()(const size_t n, const size_t c, const size_t h, const size_t w) const
{ 
    index_t ind = n * H * C * W + c * H * W + h * W + w;
    if (ind > data.size())
        throw std::out_of_range("attempt to access not allocated memory");

    return data[ind];
}

const tensor& 
tensor::operator =(std::vector<number_t> vec)
{
    _LOG << "= operator at work" << END_;
    if (vec.size() != N * C * H * W)
        _LOG[WARNING] << "Giving not equally sized data to tensor, make sure dimensions are right" << END_;
        
    data = vec;

    return *this;
}

std::ostream & 
operator <<(std::ostream & stream, const tensor &t)
{
    for (size_t i = 0; i < t.N; i++)
    {
        stream << std::endl << "> Batch number: " << i << std::endl;

        for (size_t j = 0; j < t.C; j++)
        {
            stream << "-------------------------------------------\n";

            for (size_t n = 0; n < t.H; n++)
            {
                stream << "|| ";

                for (size_t m = 0; m < t.W; m++)
                    stream << t(i, j, n, m) << " ";

                stream << "||\n";
            }
        }
    }

    return stream;
}

size_t 
tensor::get_tensor_size() const
{
    return N * C * H * W;
}

number_t 
tensor::sum_op(const number_t num1, const number_t num2)
{
    return num1 + num2;
}

tensor 
tensor::operator +(const tensor& rhs) const
{
    if (get_tensor_size() != rhs.get_tensor_size())
        throw std::logic_error("tensor sizes are not compatible");

    return make_operation(N, C, H, W, rhs, sum_op);
}

number_t
tensor::sub_op(const number_t num1, const number_t num2)
{
    return num1 - num2;
}

tensor
tensor::operator -(const tensor& rhs) const
{
    if (get_tensor_size() != rhs.get_tensor_size())
        throw std::logic_error("tensor sizes are not compatible");

    return make_operation(N, C, H, W, rhs, sub_op);
}

tensor 
tensor::make_operation(const size_t N, const size_t C, const size_t H, const size_t W, const tensor &r_operand, \
    number_t(op(const number_t num1, const number_t num2))) const
{
    std::vector<number_t> res_data;

    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < C; j++)
            for (size_t n = 0; n < H; n++)
                for (size_t m = 0; m < W; m++)
                    res_data.push_back(op((*this)(i, j, n, m), r_operand(i, j, n, m)));
                    
    tensor res(N, C, H, W);
    res = res_data;

    return res;
}