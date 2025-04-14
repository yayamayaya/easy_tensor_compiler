#include <exception>
#include "tensor.hpp"

bool
tensor_dim::operator !=(const tensor_dim& r_op) const
{
    return (N != r_op.N) || (C != r_op.C) || (H != r_op.H) || (W != r_op.W);
}

bool
tensor_dim::operator ==(const tensor_dim& r_op) const
{
    return (N == r_op.N) && (C == r_op.C) && (H == r_op.H) && (W == r_op.W);
}

void
tensor::set_tensor_size(const size_t N_val, const size_t C_val, const size_t H_val, const size_t W_val)
{
    size = tensor_dim(N_val, C_val, H_val, W_val);
}

tensor_dim
tensor::get_size() const
{
    return size;
}

number_t& 
tensor::operator ()(const size_t n, const size_t c, const size_t h, const size_t w)
{ 
    index_t ind = n * size.H * size.C * size.W + c * size.H * size.W + h * size.W + w;
    if (ind > data.size())
        throw std::out_of_range("attempt to access not allocated memory");

    return data[ind];
}

number_t 
tensor::operator ()(const size_t n, const size_t c, const size_t h, const size_t w) const
{ 
    index_t ind = n * size.H * size.C * size.W + c * size.H * size.W + h * size.W + w;
    if (ind > data.size())
        throw std::out_of_range("attempt to access not allocated memory");

    return data[ind];
}

std::ostream & 
operator <<(std::ostream & stream, const tensor &t)
{
    for (size_t i = 0; i < t.size.N; i++)
    {
        stream << std::endl << "> Batch number: " << i << std::endl;

        for (size_t j = 0; j < t.size.C; j++)
        {
            stream << "-------------------------------------------\n";

            for (size_t n = 0; n < t.size.H; n++)
            {
                stream << "|| ";

                for (size_t m = 0; m < t.size.W; m++)
                    stream << t(i, j, n, m) << " ";

                stream << "||\n";
            }
        }
    }

    return stream;
}

tensor
tensor::operator +(const tensor& t) const
{
    std::vector<number_t> new_tensor;

    if (get_size() != t.get_size())
        throw std::logic_error("tensor sizes are not compatible");

    for (index_t i = 0; i < data.size(); i++)
        new_tensor.push_back(data[i] + t.data[i]);
    
    tensor res(new_tensor);
    res.set_tensor_size(get_size());

    return res;
}

tensor
tensor::operator -(const tensor& t) const
{
    std::vector<number_t> new_tensor;

    if (get_size() != t.get_size())
        throw std::logic_error("tensor sizes are not compatible");

    for (index_t i = 0; i < data.size(); i++)
        new_tensor.push_back(data[i] - t.data[i]);
    
    tensor res(new_tensor);
    res.set_tensor_size(get_size());

    return res;
}

tensor
tensor::operator *(const number_t val) const
{
    std::vector<number_t> new_tensor;

    for (index_t i = 0; i < data.size(); i++)
        new_tensor.push_back(data[i] * val);
    
    tensor res(new_tensor);
    res.set_tensor_size(get_size());

    return res;
}

tensor
tensor::operator *(const tensor& rhs)  const
{
    std::vector<number_t> new_tensor;

    tensor_dim tensor_size = get_size();

    if (tensor_size.W != rhs.get_size().H)
        throw std::logic_error("tensor sizes are not compatible");

        
    for (index_t i = 0; i < tensor_size.N; i++)
        for (index_t j = 0; j < tensor_size.C; j++)
            for (index_t l = 0; l < tensor_size.H; l++)
                for (index_t k = 0; k < rhs.get_size().W; k++)
                {
                    number_t val = 0;
                    
                    for (index_t r = 0; r < tensor_size.W; r++)
                        val += (*this)(i, j, l, r) * rhs(i, j, r, k);

                    new_tensor.push_back(val);
                }
    
    tensor res(new_tensor);
    res.set_tensor_size(tensor_size.N, tensor_size.C, tensor_size.H, rhs.get_size().W);

    return res;
}

bool
tensor::operator ==(const tensor& rhs) const
{
    return (get_size() == rhs.get_size()) && (data == rhs.data);
}