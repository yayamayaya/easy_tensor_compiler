#ifndef TENSOR_CLASS_HEADER
#define TENSOR_CLASS_HEADER

#include <vector>
#include "logging.hpp"

using number_t = int;

using index_t  = long unsigned int;

class tensor
{
public:

    //Create class with n data batches, c matrixes in each batch, each matrix is h x w
    tensor(const size_t n, const size_t c, const size_t h, const size_t w): data(), N(n), C(c), H(h), W(w) {};

    tensor() =delete;

    size_t get_tensor_size() const;

    //Take a tensor value at index
    number_t & operator ()(const size_t n, const size_t c, const size_t h, const size_t w);

    //Take a tensor value at index
    number_t operator ()(const size_t n, const size_t c, const size_t h, const size_t w) const;

    //Assign data to tensor
    const tensor & operator =(std::vector<number_t> vec);

    //Print tensor to stream
    friend std::ostream & operator <<(std::ostream & stream, const tensor &t);

//Перенести операторы в классы операций
    tensor operator +(const tensor& rhs)  const;  

    tensor operator -(const tensor& rhs)  const;

    tensor operator *(const number_t num) const;

    tensor operator *(const tensor& rhs)  const;

    ~tensor() {};
    
private:
    
    std::vector<number_t> data;

    size_t N;
    size_t C;
    size_t H;
    size_t W;

//Перенести операции в классы операций
    tensor make_operation(const size_t N, const size_t C, const size_t H, const size_t W, const tensor &r_operand, \
            number_t(op(const number_t num1, const number_t num2))) const;

    static number_t sum_op(const number_t num1, const number_t num2);

    static number_t sub_op(const number_t num1, const number_t num2);
};


#endif