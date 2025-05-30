#ifndef TENSOR_CLASS_HEADER
#define TENSOR_CLASS_HEADER

#include <vector>
#include <ostream>

using number_t = double;

using index_t  = long unsigned int;

struct tensor_dim
{
    size_t N;
    size_t C;
    size_t H;
    size_t W;

    tensor_dim(): N(0), C(0), H(0), W(0) {};

    tensor_dim(const size_t Nv, const size_t Cv, const size_t Hv, const size_t Wv): N(Nv), C(Cv), H(Hv), W(Wv) {}; 

    bool operator !=(const tensor_dim& r_op) const;
    
    bool operator ==(const tensor_dim& r_op) const;
};

class tensor
{
public:

    //Create tensor with data and given size
    tensor(const size_t N, const size_t C, const size_t H, const size_t W,\
            std::vector<number_t> numbers): data(numbers), size(N, C, H, W) {};

    ~tensor() {};

    //set tensor size with n data batches, c matrixes in each batch, each matrix is h x w
    void set_tensor_size(const size_t N, const size_t C, const size_t H, const size_t W);

    //Take a tensor value at index
    number_t & operator ()(const size_t n, const size_t c, const size_t h, const size_t w);

    //Take a tensor value at index
    number_t operator ()(const size_t n, const size_t c, const size_t h, const size_t w) const;

    //Print tensor to stream
    friend std::ostream & operator <<(std::ostream & stream, const tensor &t);

    //Arithmetic operations with tensors
    tensor operator +(const tensor& rhs)  const;  

    tensor operator -(const tensor& rhs)  const;

    tensor operator *(const number_t num) const;

    tensor operator *(const tensor& rhs)  const;

    tensor operator /(const tensor& rhs)  const;

    friend class relu_op;

    friend class softmax_op;

    bool   operator ==(const tensor& rhs) const;
    
private:

    std::vector<number_t> data;

    tensor_dim size;

    tensor_dim get_size()  const;

#ifdef OPTIMIZED_OPERATIONS

    tensor transpose() const;

    static constexpr index_t tile_size = 64;

    tensor simple_mul        (const tensor& rhs) const;

    tensor cache_friendly_mul(const tensor& rhs) const;

    tensor tiling_mul        (const tensor& rhs) const;

    tensor transposed_mult   (const tensor& rhs_transposed) const;

    static constexpr index_t VEC_SIZE = 8;

    static number_t vector_mult_sum(const number_t *v1, const number_t *v2, const size_t size);
    
    static number_t *get_padded(const number_t *ptr, const size_t size, const size_t divider);

    tensor simple_conv (const tensor& rhs) const;

    tensor im2col(const tensor& rhs) const;
    
    friend class optimization_testing_mat_mul_equity_Test;

    friend class optimization_testing_transpose_test_Test;
    
    friend class bench;

#endif

    bool       is_square() const;

    tensor(): data(), size() {};

    tensor(std::vector<number_t> numbers): data(numbers), size() {};

    void set_tensor_size(const tensor_dim dim);
};


#endif