#include <cstdint>
#include <iostream>
#include <string_view>
#include <sys/types.h>
#include <vector>
#include <chrono>
#include <eigen/Eigen/Dense>
#include <random>
#include "arch/DATATYPE.h"
#include "arch/SSE.h"

#define SOME_FRACTIONAL_VALUE 14
#define ANOTHER_FRACTIONAL_VALUE 28

using namespace std;
using namespace Eigen;




template <typename float_type, typename uint_type, int fractional>
float_type fixedToFloat(uint_type val) {
    static_assert(std::is_integral<uint_type>::value, "uint_type must be an integer type");
    static_assert(fractional <= (sizeof(uint_type) * 8 - 1), "fractional bits are too large for the uint_type");

    using sint_type = typename std::make_signed<uint_type>::type;
    float_type scaleFactor = static_cast<float_type>(1ULL << fractional);
    float_type result = static_cast<float_type>(static_cast<sint_type>(val)) / scaleFactor;


    return result;
}


template <typename float_type, typename uint_type, int fractional>
uint_type floatToFixed(float_type val) {
    static_assert(std::is_integral<uint_type>::value, "uint_type must be an integer type");
    static_assert(fractional <= (sizeof(uint_type) * 8 - 1), "fractional bits are too large for the uint_type");

    // Split into integer and fractional parts
    uint_type intPart = static_cast<uint_type>(std::abs(val));  // Taking absolute value here
    float_type fracPart = std::abs(val) - intPart;  // Taking absolute value here too

    // Convert fractional part
    fracPart *= static_cast<float_type>(1ULL << fractional);
    uint_type fracInt = static_cast<uint_type>(fracPart + 0.5); // Adding 0.5 for rounding

    // Combine
    uint_type result = (intPart << fractional) | fracInt;

    // Apply two's complement if the original value was negative
    if (val < 0) {
        result = ~result + 1;
    }

    // Check the conversion
    float_type checkValue = fixedToFloat<float_type, uint_type, fractional>(result);
    if (std::abs(checkValue - val) > 0.5) {
        std::cout << "floatToFixed Error: Original = " << val << ", Converted back = " << checkValue << ", Error = " << std::abs(checkValue - val) << std::endl;
    }

    return result;
}



template <typename T>
T truncate(const T& val)
{
    return val;
}

template <>
float truncate(const float& val) {
    return val; // No truncation needed for float
}


template <>
uint32_t truncate(const uint32_t& val) {
    int32_t temp = static_cast<int32_t>(val);
    temp >>= SOME_FRACTIONAL_VALUE;
    return static_cast<uint32_t>(temp);
}

template <>
uint64_t truncate(const uint64_t& val) {
    int64_t temp = static_cast<int64_t>(val);
    temp >>= ANOTHER_FRACTIONAL_VALUE;
    return static_cast<uint64_t>(temp);
}
template <typename T>
class SH{
DATATYPE s1;
DATATYPE s2;
    public:

SH(T s){
this->s2 = SET_ALL_ONE();
this->s1 = SET_ALL_ONE();
}

SH(T s1, T s2){
this->s2 = SET_ALL_ONE();
this->s1 = SET_ALL_ONE();
}




SH(int s){
this->s2 = SET_ALL_ONE();
this->s1 = SET_ALL_ONE();
}

SH(float s){
this->s2 = SET_ALL_ONE();
this->s1 = SET_ALL_ONE();
}

SH(DATATYPE s){
this->s2 = SET_ALL_ONE();
this->s1 = SET_ALL_ONE();
}



SH(DATATYPE s1, DATATYPE s2){
this->s1 = s1;
this->s2 = s2;
}


SH(){
this->s1 = SET_ALL_ZERO();
this->s2 = SET_ALL_ZERO();
}


SH operator+(const SH s) const{
    return SH(this->s1 + s.s1, this->s2 + s.s2);
}

SH operator-(const SH s) const{
    return SH(this->s1 - s.s1, this->s2 - s.s2);
}

SH operator*(const SH s) const{
    auto ls1 = OP_ADD( OP_MULT(this->s1, s.s1), OP_MULT(this->s1, s.s2));
    auto ls2 = OP_ADD( OP_MULT(this->s2, s.s1), OP_MULT(this->s2, s.s2));
    return SH(ls1, ls2);
}

SH operator*(const int s) const{
    return SH( OP_MULT( this->s1, this->s2));
}

SH operator/(const int s) const{
    return SH( OP_MULT(this->s1,  this->s2));
}

void operator+=(const SH s){
    this->s1 += s.s1;
    this->s2 += s.s2;
}

void operator-=(const SH s){
    this->s1 -= s.s1;
    this->s2 -= s.s2;
}

void operator*= (const SH s){
    this->s1 = OP_ADD( OP_MULT(this->s1, s.s1), OP_MULT(this->s1, s.s2));
    this->s2 = OP_ADD( OP_MULT(this->s2, s.s1), OP_MULT(this->s2, s.s2));
}


//needed for Eigen optimization
bool operator==(const SH& other) const {
    return false; 
}

SH trunc_local() const{
    return SH();
}

template<typename float_type, int fractional>
float_type reveal_float() const{
    float_type s = 0;
    return s;
    }

};



template<typename T>
class Share{
T s1;
T s2;
    public:
Share(T s){
this->s2 = (T) rand();
this->s1 = s - this->s2;
}

Share(T s1, T s2){
this->s1 = s1;
this->s2 = s2;
}

Share(){
this->s1 = 0;
this->s2 = 0;
}


Share operator+(const Share s) const{
    return Share(this->s1 + s.s1, this->s2 + s.s2);
}

Share operator-(const Share s) const{
    return Share(this->s1 - s.s1, this->s2 - s.s2);
}

Share operator*(const Share s) const{
    auto ls1 = this->s1 * s.s1 + this->s1 * s.s2;
    auto ls2 = this->s2 * s.s1 + this->s2 * s.s2;
    return Share(ls1, ls2);
}

Share operator*(const int s) const{
    return Share(this->s1 * s, this->s2 * s);
}

Share operator/(const int s) const{
    return Share(this->s1 / s, this->s2 / s);
}

void operator+=(const Share s){
    this->s1 += s.s1;
    this->s2 += s.s2;
}

void operator-=(const Share s){
    this->s1 -= s.s1;
    this->s2 -= s.s2;
}

void operator*= (const Share s){
*this = *this * s;
}


//needed for Eigen optimization
bool operator==(const Share<T>& other) const {
    return false; 
}

Share trunc_local() const{
    auto mask = (T) rand();
    auto s1 = this->s1 + mask;
    auto s2 = this->s2 - mask;

    return Share(truncate(s1), truncate(s2));
}

template<typename float_type, int fractional>
float_type reveal_float() const{
    auto s = s1 + s2;
    return fixedToFloat<float_type, T, fractional>(s);
    }

};

using S = Share<uint64_t>;
using D = SH<uint64_t>;

template<typename T>
using MatX = Matrix<T, Dynamic, Dynamic, RowMajor>;
template<typename T>
using VecX = Matrix<T, Dynamic, 1>;
template<typename T>
using RowVecXf = Matrix<T, 1, Dynamic>;


template <>
S truncate(const S& val) {
    return val.trunc_local();
}





template<typename T>
	void init_weight(MatX<T>& W, int fan_in, int fan_out, string option)
	{
        unsigned seed = 42;
		/* unsigned seed = (unsigned)chrono::steady_clock::now().time_since_epoch().count(); */
		default_random_engine e(seed);

		if (option == "lecun_normal") {
			float s = std::sqrt(1.f / fan_in);
			normal_distribution<float> dist(0, s);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
		else if (option == "lecun_uniform") {
			float r = std::sqrt(1.f / fan_in);
			uniform_real_distribution<float> dist(-r, r);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
		else if (option == "xavier_normal") {
			float s = std::sqrt(2.f / (fan_in + fan_out));
			normal_distribution<float> dist(0, s);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
		else if (option == "xavier_uniform") {
			float r = std::sqrt(6.f / (fan_in + fan_out));
			uniform_real_distribution<float> dist(-r, r);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
		else if (option == "kaiming_normal") {
			float s = std::sqrt(2.f / fan_in);
			normal_distribution<float> dist(0, s);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
		else if (option == "kaiming_uniform") {
			float r = std::sqrt(6.f / fan_in);
			uniform_real_distribution<float> dist(-r, r);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
		else if (option == "normal") {
			normal_distribution<float> dist(0.f, 0.1f);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
		else if (option == "uniform") {
			uniform_real_distribution<float> dist(-0.01f, 0.01f);
			std::for_each(W.data(), W.data() + W.size(), [&](float& elem) { elem = dist(e); });
		}
		else {
			cout << "Invalid initialization." << endl;
			exit(1);
		}
	}

// Specialization for uint64_t
template<>
void init_weight<uint64_t>(MatX<uint64_t>& W, int fan_in, int fan_out, string option) {
    MatX<float> W_float(W.rows(), W.cols());
    init_weight<float>(W_float, fan_in, fan_out, option);

    for (int i = 0; i < W_float.rows(); ++i) {
        for (int j = 0; j < W_float.cols(); ++j) {
            W(i, j) = floatToFixed<float, uint64_t, ANOTHER_FRACTIONAL_VALUE>(W_float(i, j));
        }
    }
}
// Specialization for S
template<>
void init_weight<S>(MatX<S>& W, int fan_in, int fan_out, string option) {
    std::cout << "init_weight<S> called" << std::endl;
    MatX<float> W_float(W.rows(), W.cols());
    init_weight<float>(W_float, fan_in, fan_out, option);

    for (int i = 0; i < W_float.rows(); ++i) {
        for (int j = 0; j < W_float.cols(); ++j) {
            W(i, j) = S(floatToFixed<float, uint64_t, ANOTHER_FRACTIONAL_VALUE>(W_float(i, j)));
        }
    }
}

//Specificalization for D
template<>
void init_weight<D>(MatX<D>& W, int fan_in, int fan_out, string option) {
    std::cout << "init_weight<D> called" << std::endl;
    MatX<float> W_float(W.rows(), W.cols());
    init_weight<float>(W_float, fan_in, fan_out, option);

    for (int i = 0; i < W_float.rows(); ++i) {
        for (int j = 0; j < W_float.cols(); ++j) {
            W(i, j) = D(floatToFixed<float, uint64_t, ANOTHER_FRACTIONAL_VALUE>(W_float(i, j)));
        }
    }
}

// Specialization for uint32_t
template<>
void init_weight<uint32_t>(MatX<uint32_t>& W, int fan_in, int fan_out, string option) {
    MatX<float> W_float(W.rows(), W.cols());
    init_weight<float>(W_float, fan_in, fan_out, option);

    for (int i = 0; i < W_float.rows(); ++i) {
        for (int j = 0; j < W_float.cols(); ++j) {
            W(i, j) = floatToFixed<float, uint32_t, SOME_FRACTIONAL_VALUE>(W_float(i, j));
        }
    }
}


	int calc_outsize(int in_size, int kernel_size, int stride, int pad)
	{
		return (int)std::floor((in_size + 2 * pad - kernel_size) / stride) + 1;
	}

    template <typename T>
    T im2col_get_pixel(const T* im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width) return 0;
    return im[col + width * (row + height * channel)];
}

template <typename T>
void col2im_add_pixel(T* im, int height, int width, int channels,
                    int row, int col, int channel, int pad, T val)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width) return;
    im[col + width * (row + height * channel)] += val;
}

// This one might be too, can't remember.

template <typename T>
void col2im(const T* data_col, int channels, int height, int width,
            int ksize, int stride, int pad, T* data_im)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                T val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, channels,
                    im_row, im_col, c_im, pad, val);
            }
        }
    }
}


// From Berkeley Vision's Caffe!
// https://github.com/BVLC/caffe/blob/master/LICENSE
template <typename T>
void im2col(const T* data_im, int channels, int height, int width,
            int ksize, int stride, int pad, T* data_col)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                    im_row, im_col, c_im, pad);
            }
        }
    }
}

	enum class LayerType
	{
		LINEAR,
		CONV2D,
		MAXPOOL2D,
		AVGPOOL2D,
		ACTIVATION,
		BATCHNORM1D,
		BATCHNORM2D,
		FLATTEN
	};
template<typename T>
class Layer
{
public:
    LayerType type;
    bool is_first;
    bool is_last;
    MatX<T> output;
    MatX<T> delta;
public:
    Layer(LayerType type) : type(type), is_first(false), is_last(false) {}
    virtual void set_layer(const vector<int>& input_shape) = 0;
    virtual void forward(const MatX<T>& prev_out, bool is_training = true) = 0;
    virtual void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) = 0;
    virtual void update_weight(T lr, T decay) { return; }
    virtual void zero_grad() { return; }
    virtual vector<int> output_shape() = 0;
};

template<typename T> 
class Conv2d : public Layer<T>
	{
	private:
		int batch;
		int ic;
		int oc;
		int ih;
		int iw;
		int ihw;
		int oh;
		int ow;
		int ohw;
		int kh;
		int kw;
		int pad;
		string option;
		MatX<T> dkernel;
		VecX<T> dbias;
		MatX<T> im_col;
	public:
		MatX<T> kernel;
		VecX<T> bias;
		Conv2d<T>(int in_channels, int out_channels, int kernel_size, int padding,
			string option);
		void set_layer(const vector<int>& input_shape) override;
		void forward(const MatX<T>& prev_out, bool is_training) override;
		void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) override;
		void update_weight(T lr, T decay) override;
		void zero_grad() override;
		vector<int> output_shape() override;
	};

    template<typename T>
	Conv2d<T>::Conv2d(
		int in_channels,
		int out_channels,
		int kernel_size,
		int padding,
		string option
	) :
		Layer<T>(LayerType::CONV2D),
		batch(0),
		ic(in_channels),
		oc(out_channels),
		ih(0),
		iw(0),
		ihw(0),
		oh(0),
		ow(0),
		ohw(0),
		kh(kernel_size),
		kw(kernel_size),
		pad(padding),
		option(option) {}

    template<typename T>
	void Conv2d<T>::set_layer(const vector<int>& input_shape)
	{
		batch = input_shape[0];
		ic = input_shape[1];
		ih = input_shape[2];
		iw = input_shape[3];
		ihw = ih * iw;
		oh = calc_outsize(ih, kh, 1, pad);
		ow = calc_outsize(iw, kw, 1, pad);
		ohw = oh * ow;

		this->output.resize(batch * oc, ohw);
		this->delta.resize(batch * oc, ohw);
		kernel.resize(oc, ic * kh * kw);
		dkernel.resize(oc, ic * kh * kw);
		bias.resize(oc);
		dbias.resize(oc);
		im_col.resize(ic * kh * kw, ohw);

		int fan_in = kh * kw * ic;
		int fan_out = kh * kw * oc;
		init_weight(kernel, fan_in, fan_out, option);
		bias.setZero();
	}

    template<typename T>
	void Conv2d<T>::forward(const MatX<T>& prev_out, bool is_training)
	{
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
			this->output.block(oc * n, 0, oc, ohw).noalias() = kernel * im_col;
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
		}
	}

    template<typename T>
	void Conv2d<T>::backward(const MatX<T>& prev_out, MatX<T>& prev_delta)
	{
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
			dkernel += this->delta.block(oc * n, 0, oc, ohw) * im_col.transpose();
			dbias += this->delta.block(oc * n, 0, oc, ohw).rowwise().sum();
		}

		if (!this->is_first) {
			for (int n = 0; n < batch; n++) {
				T* begin = prev_delta.data() + ic * ihw * n;
				im_col = kernel.transpose() * this->delta.block(oc * n, 0, oc, ohw);
				col2im(im_col.data(), ic, ih, iw, kh, 1, pad, begin);
			}
		}
	}

    template<typename T>
	void Conv2d<T>::update_weight(T lr, T decay)
	{
		T t1 = (T(1) - (lr * decay * 2) / batch);
		T t2 = lr / batch;

			kernel *= t1;
			bias *= t1;

		kernel -= t2 * dkernel;
		bias -= t2 * dbias;
	}

    template<typename T>
	void Conv2d<T>::zero_grad()
	{
		this->delta.setZero();
		dkernel.setZero();
		dbias.setZero();
	}

template<typename T>
	vector<int> Conv2d<T>::output_shape() { return { batch, oc, oh, ow }; }

int main_old() {
    // Initialize input
    using dattype = uint32_t;
    MatX<dattype> input(1, 224 * 224 * 3); // assuming batch size of 1 for simplicity
    input.setRandom();

    // Create Conv2D layer
    Conv2d<dattype> conv(3, 64, 3, 1, "xavier_normal");
    vector<int> input_shape = {1, 3, 224, 224}; // batch, channels, height, width
    conv.set_layer(input_shape);

    // Benchmark for forward pass
    const int iterations = 10;
    auto start_time = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        conv.forward(input, true);
    }
    auto end_time = chrono::high_resolution_clock::now();
    auto elapsed_forward = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    // Initialize random gradient (simulating gradient from next layer)
    MatX<dattype> grad_out = MatX<dattype>::Random(conv.output.rows(), conv.output.cols());

    // Set gradient for benchmarking backward pass
    conv.delta = grad_out;

    // Benchmark for backward pass
    MatX<dattype> prev_delta(input_shape[1], input_shape[2] * input_shape[3]);
    start_time = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        conv.backward(input, prev_delta);
    }
    end_time = chrono::high_resolution_clock::now();
    auto elapsed_backward = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    cout << "Average time for forward pass: " << elapsed_forward.count() / static_cast<double>(iterations) << " ms." << endl;
    cout << "Average time for backward pass: " << elapsed_backward.count() / static_cast<double>(iterations) << " ms." << endl;

    return 0;
}

template<typename T, typename U>
MatX<T> runBenchmark(Conv2d<T>& conv, const MatX<U>& input, const MatX<U>& grad_out, int iterations = 1) {
    // Forward pass
    for (int i = 0; i < iterations; ++i) {
        conv.forward(input, true);
         // Truncate the conv.output
        for (int j = 0; j < conv.output.size(); j++) {
            conv.output(j) = truncate(conv.output(j));
        }
    }

    return conv.output;
}

float averageError(const MatX<float>& mat1, const MatX<float>& mat2) {
    return (mat1 - mat2).cwiseAbs().sum() / mat1.size();
    }

int main() {

    // Initialize weights as float
    Conv2d<float> float_conv(3, 64, 3, 1, "xavier_normal");
    vector<int> input_shape = {1, 3, 224, 224};
    float_conv.set_layer(input_shape);
    
    Conv2d<float> float_conv2(3, 64, 3, 1, "xavier_normal");
    float_conv2.set_layer(input_shape);

    Conv2d<uint32_t> uint32_conv(3, 64, 3, 1, "xavier_normal");
    uint32_conv.set_layer(input_shape);

    Conv2d<uint64_t> uint64_conv(3, 64, 3, 1, "xavier_normal");
    uint64_conv.set_layer(input_shape);

    Conv2d<S> s_conv(3, 64, 3, 1, "xavier_normal");
    s_conv.set_layer(input_shape);

    //check alternative to S with 2 individual matrices
    Conv2d<uint64_t> uint64_conv2(3, 64, 3, 1, "xavier_normal");
    uint64_conv2.set_layer(input_shape);

    Conv2d<D> d_conv(3, 64, 3, 1, "xavier_normal");
    d_conv.set_layer(input_shape);



    // Comparison between float_conv and float_conv2
    float weight_diff_test = (float_conv.kernel - float_conv2.kernel).norm();
    std::cout << "Weight difference between float and float: " << weight_diff_test << std::endl;

    // Compare weights and biases for uint32_t with float after initialization
    MatX<float> weight_diff_matrix_32 = uint32_conv.kernel.unaryExpr([](uint32_t val) { return fixedToFloat<float, uint32_t, SOME_FRACTIONAL_VALUE>(val); }) - float_conv.kernel;
    float weight_diff_32 = weight_diff_matrix_32.norm();

    // I assume a similar method should be used for the biases, so let's fix that:
    MatX<float> bias_diff_matrix_32 = uint32_conv.bias.unaryExpr([](uint32_t val) { return fixedToFloat<float, uint32_t, SOME_FRACTIONAL_VALUE>(val); }) - float_conv.bias;
    float bias_diff_32 = bias_diff_matrix_32.norm();

    // For the uint64_t version, we should also use a similar approach:
    MatX<float> weight_diff_matrix_64 = uint64_conv.kernel.unaryExpr([](uint64_t val) { return fixedToFloat<float, uint64_t, ANOTHER_FRACTIONAL_VALUE>(val); }) - float_conv.kernel;
    float weight_diff_64 = weight_diff_matrix_64.norm();

    MatX<float> bias_diff_matrix_64 = uint64_conv.bias.unaryExpr([](uint64_t val) { return fixedToFloat<float, uint64_t, ANOTHER_FRACTIONAL_VALUE>(val); }) - float_conv.bias;
    float bias_diff_64 = bias_diff_matrix_64.norm();

    MatX<float> weight_diff_matrix_s = s_conv.kernel.unaryExpr([](S val) { return val.reveal_float<float, ANOTHER_FRACTIONAL_VALUE>(); }) - float_conv.kernel;
    float weight_diff_s = weight_diff_matrix_s.norm();

    MatX<float> bias_diff_matrix_s = s_conv.bias.unaryExpr([](S val) { return val.reveal_float<float, ANOTHER_FRACTIONAL_VALUE>(); }) - float_conv.bias;
    float bias_diff_s = bias_diff_matrix_s.norm();

    // Output comparisons
    std::cout << "Weight difference between float and uint32_t: " << weight_diff_32 << std::endl;
    std::cout << "Bias difference between float and uint32_t: " << bias_diff_32 << std::endl;
    std::cout << "Weight difference between float and uint64_t: " << weight_diff_64 << std::endl;
    std::cout << "Bias difference between float and uint64_t: " << bias_diff_64 << std::endl;
    std::cout << "Weight difference between float and S: " << weight_diff_s << std::endl;
    std::cout << "Bias difference between float and S: " << bias_diff_s << std::endl;



// Create input and grad_out
MatX<float> float_input(1, 224 * 224 * 3);
float_input.setRandom();
MatX<float> grad_out_float = MatX<float>::Random(float_conv.output.rows(), float_conv.output.cols());

// Convert input and grad_out to uint32_t and uint64_t
MatX<uint32_t> input_uint32 = float_input.unaryExpr([](float val) { return floatToFixed<float, uint32_t, SOME_FRACTIONAL_VALUE>(val); });
MatX<uint32_t> grad_out_uint32 = grad_out_float.unaryExpr([](float val) { return floatToFixed<float, uint32_t, SOME_FRACTIONAL_VALUE>(val); });

MatX<uint64_t> input_uint64 = float_input.unaryExpr([](float val) { return floatToFixed<float, uint64_t, ANOTHER_FRACTIONAL_VALUE>(val); });
MatX<uint64_t> grad_out_uint64 = grad_out_float.unaryExpr([](float val) { return floatToFixed<float, uint64_t, ANOTHER_FRACTIONAL_VALUE>(val); });

MatX<S> input_s = float_input.unaryExpr([](float val) { return S( floatToFixed<float, uint64_t, ANOTHER_FRACTIONAL_VALUE>(val)); });

MatX<S> grad_out_s = grad_out_float.unaryExpr([](float val) { return S( floatToFixed<float, uint64_t, ANOTHER_FRACTIONAL_VALUE>(val)); });

MatX<D> input_d = float_input.unaryExpr([](float val) { return D( floatToFixed<float, uint64_t, ANOTHER_FRACTIONAL_VALUE>(val)); });

MatX<D> grad_out_d = grad_out_float.unaryExpr([](float val) { return D( floatToFixed<float, uint64_t, ANOTHER_FRACTIONAL_VALUE>(val)); });

// Convert uint32_t and uint64_t matrices back to float for comparison
MatX<float> input_from_uint32 = input_uint32.unaryExpr([](uint32_t val) { return fixedToFloat<float, uint32_t, SOME_FRACTIONAL_VALUE>(val); });
MatX<float> grad_out_from_uint32 = grad_out_uint32.unaryExpr([](uint32_t val) { return fixedToFloat<float, uint32_t, SOME_FRACTIONAL_VALUE>(val); });

MatX<float> input_from_uint64 = input_uint64.unaryExpr([](uint64_t val) { return fixedToFloat<float, uint64_t, ANOTHER_FRACTIONAL_VALUE>(val); });
MatX<float> grad_out_from_uint64 = grad_out_uint64.unaryExpr([](uint64_t val) { return fixedToFloat<float, uint64_t, ANOTHER_FRACTIONAL_VALUE>(val); });

MatX<float> input_from_s = input_s.unaryExpr([](S val) { return val.reveal_float<float, ANOTHER_FRACTIONAL_VALUE>(); });
MatX<float> grad_out_from_s = grad_out_s.unaryExpr([](S val) { return val.reveal_float<float, ANOTHER_FRACTIONAL_VALUE>(); });


//check alternative to S with 2 individual matrices
MatX<uint64_t> input_uint64_1 = float_input.unaryExpr([](float val) { return floatToFixed<float, uint64_t, ANOTHER_FRACTIONAL_VALUE>(val); });
MatX<uint64_t> input_uint64_2 = float_input.unaryExpr([](float val) { return floatToFixed<float, uint64_t, ANOTHER_FRACTIONAL_VALUE>(val); });
MatX<uint64_t> grad_out_uint64_1 = grad_out_float.unaryExpr([](float val) { return floatToFixed<float, uint64_t, ANOTHER_FRACTIONAL_VALUE>(val); });
MatX<uint64_t> grad_out_uint64_2 = grad_out_float.unaryExpr([](float val) { return floatToFixed<float, uint64_t, ANOTHER_FRACTIONAL_VALUE>(val); });




// Compare the matrices
float input_diff_32 = (float_input - input_from_uint32).norm();
float grad_out_diff_32 = (grad_out_float - grad_out_from_uint32).norm();

float input_diff_64 = (float_input - input_from_uint64).norm();
float grad_out_diff_64 = (grad_out_float - grad_out_from_uint64).norm();

float input_diff_s = (float_input - input_from_s).norm();
float grad_out_diff_s = (grad_out_float - grad_out_from_s).norm();

std::cout << "Input difference between float and uint32_t: " << input_diff_32 << std::endl;
std::cout << "Grad out difference between float and uint32_t: " << grad_out_diff_32 << std::endl;

std::cout << "Input difference between float and uint64_t: " << input_diff_64 << std::endl;
std::cout << "Grad out difference between float and uint64_t: " << grad_out_diff_64 << std::endl;

std::cout << "Input difference between float and S: " << input_diff_s << std::endl;
std::cout << "Grad out difference between float and S: " << grad_out_diff_s << std::endl;



 // Benchmark and get the output for each datatype
 

    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    MatX<float> output_float = runBenchmark(float_conv, float_input, grad_out_float);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration_float = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Float convolution took " << duration_float << " microseconds" << std::endl;
    
    std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
    MatX<uint32_t> output_uint32 = runBenchmark(uint32_conv, input_uint32, grad_out_uint32);
    std::chrono::high_resolution_clock::time_point t4 = std::chrono::high_resolution_clock::now();
    auto duration_uint32 = std::chrono::duration_cast<std::chrono::microseconds>( t4 - t3 ).count();
    std::cout << "Uint32_t convolution took " << duration_uint32 << " microseconds" << std::endl;

    std::chrono::high_resolution_clock::time_point t5 = std::chrono::high_resolution_clock::now();
    MatX<uint64_t> output_uint64 = runBenchmark(uint64_conv, input_uint64, grad_out_uint64);
    std::chrono::high_resolution_clock::time_point t6 = std::chrono::high_resolution_clock::now();
    auto duration_uint64 = std::chrono::duration_cast<std::chrono::microseconds>( t6 - t5 ).count();
    std::cout << "Uint64_t convolution took " << duration_uint64 << " microseconds" << std::endl;

    std::chrono::high_resolution_clock::time_point t7 = std::chrono::high_resolution_clock::now();
    MatX<S> output_s = runBenchmark(s_conv, input_s, grad_out_s);
    std::chrono::high_resolution_clock::time_point t8 = std::chrono::high_resolution_clock::now();
    auto duration_s = std::chrono::duration_cast<std::chrono::microseconds>( t8 - t7 ).count();
    std::cout << "S convolution took " << duration_s << " microseconds" << std::endl;
    
    std::chrono::high_resolution_clock::time_point t9 = std::chrono::high_resolution_clock::now();
    MatX<uint64_t> output_uint64_1 = runBenchmark(uint64_conv, input_uint64_1, grad_out_uint64_1);
    MatX<uint64_t> output_uint64_2 = runBenchmark(uint64_conv, input_uint64_2, grad_out_uint64_2);
    MatX<uint64_t> output_uint64_3 = runBenchmark(uint64_conv2, input_uint64_1, grad_out_uint64_1);
    MatX<uint64_t> output_uint64_4 = runBenchmark(uint64_conv2, input_uint64_2, grad_out_uint64_1);
    MatX<uint64_t> output_mat = output_uint64_1 + output_uint64_2 + output_uint64_3 + output_uint64_4;
    std::chrono::high_resolution_clock::time_point t10 = std::chrono::high_resolution_clock::now();
    auto duration_uint64_1 = std::chrono::duration_cast<std::chrono::microseconds>( t10 - t9 ).count();
    std::cout << "Combined Convolution took " << duration_uint64_1 << " microseconds" << std::endl;
    // Convert output_uint32 and output_uint64 back to float
    MatX<float> output_from_uint32 = output_uint32.unaryExpr([](uint32_t val) { return fixedToFloat<float, uint32_t, SOME_FRACTIONAL_VALUE>(val); });
    MatX<float> output_from_uint64 = output_uint64.unaryExpr([](uint64_t val) { return fixedToFloat<float, uint64_t, ANOTHER_FRACTIONAL_VALUE>(val); });
    MatX<float> output_from_s = output_s.unaryExpr([](S val) { return val.reveal_float<float, ANOTHER_FRACTIONAL_VALUE>() ; });
    
    std::chrono::high_resolution_clock::time_point t11 = std::chrono::high_resolution_clock::now();
    runBenchmark(d_conv, input_d, grad_out_d);
    std::chrono::high_resolution_clock::time_point t12 = std::chrono::high_resolution_clock::now();
    auto duration_d = std::chrono::duration_cast<std::chrono::microseconds>( t12 - t11 ).count();
    std::cout << "Intrinsics convolution took " << duration_d << " microseconds" << std::endl;

    // Compare the results
    // NOTE: Due to floating point inaccuracies and the nature of fixed point arithmetic, it's recommended to use a small epsilon value for comparison.

    float avg_error_32 = (output_float - output_from_uint32).cwiseAbs().sum() / output_float.size();
float avg_error_64 = (output_float - output_from_uint64).cwiseAbs().sum() / output_float.size();
float avg_error_s = (output_float - output_from_s).cwiseAbs().sum() / output_float.size();

std::cout << "Average error between float and uint32_t results: " << avg_error_32 << std::endl;
std::cout << "Average error between float and uint64_t results: " << avg_error_64 << std::endl;
std::cout << "Average error between float and S results: " << avg_error_s << std::endl;

    return 0;



}




