#include <iostream>
#include <vector>
#include <chrono>
#include <eigen/Eigen/Dense>
#include <random>


using namespace std;
using namespace Eigen;

typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatXf;
typedef Matrix<float, Dynamic, 1> VecXf;
typedef Matrix<float, 1, Dynamic> RowVecXf;
typedef Matrix<int, Dynamic, Dynamic, RowMajor> MatXi;
typedef Matrix<int, Dynamic, 1> VecXi;


	void init_weight(MatXf& W, int fan_in, int fan_out, string option)
	{
		unsigned seed = (unsigned)chrono::steady_clock::now().time_since_epoch().count();
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

	int calc_outsize(int in_size, int kernel_size, int stride, int pad)
	{
		return (int)std::floor((in_size + 2 * pad - kernel_size) / stride) + 1;
	}

    float im2col_get_pixel(const float* im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width) return 0;
    return im[col + width * (row + height * channel)];
}

void col2im_add_pixel(float* im, int height, int width, int channels,
                    int row, int col, int channel, int pad, float val)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width) return;
    im[col + width * (row + height * channel)] += val;
}

// This one might be too, can't remember.
void col2im(const float* data_col, int channels, int height, int width,
            int ksize, int stride, int pad, float* data_im)
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
                float val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, channels,
                    im_row, im_col, c_im, pad, val);
            }
        }
    }
}


// From Berkeley Vision's Caffe!
// https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col(const float* data_im, int channels, int height, int width,
            int ksize, int stride, int pad, float* data_col)
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

	class Layer
	{
	public:
		LayerType type;
		bool is_first;
		bool is_last;
		MatXf output;
		MatXf delta;
	public:
		Layer(LayerType type) : type(type), is_first(false), is_last(false) {}
		virtual void set_layer(const vector<int>& input_shape) = 0;
		virtual void forward(const MatXf& prev_out, bool is_training = true) = 0;
		virtual void backward(const MatXf& prev_out, MatXf& prev_delta) = 0;
		virtual void update_weight(float lr, float decay) { return; }
		virtual void zero_grad() { return; }
		virtual vector<int> output_shape() = 0;
	};


class Conv2d : public Layer
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
		MatXf dkernel;
		VecXf dbias;
		MatXf im_col;
	public:
		MatXf kernel;
		VecXf bias;
		Conv2d(int in_channels, int out_channels, int kernel_size, int padding,
			string option);
		void set_layer(const vector<int>& input_shape) override;
		void forward(const MatXf& prev_out, bool is_training) override;
		void backward(const MatXf& prev_out, MatXf& prev_delta) override;
		void update_weight(float lr, float decay) override;
		void zero_grad() override;
		vector<int> output_shape() override;
	};

	Conv2d::Conv2d(
		int in_channels,
		int out_channels,
		int kernel_size,
		int padding,
		string option
	) :
		Layer(LayerType::CONV2D),
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

	void Conv2d::set_layer(const vector<int>& input_shape)
	{
		batch = input_shape[0];
		ic = input_shape[1];
		ih = input_shape[2];
		iw = input_shape[3];
		ihw = ih * iw;
		oh = calc_outsize(ih, kh, 1, pad);
		ow = calc_outsize(iw, kw, 1, pad);
		ohw = oh * ow;

		output.resize(batch * oc, ohw);
		delta.resize(batch * oc, ohw);
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

	void Conv2d::forward(const MatXf& prev_out, bool is_training)
	{
		for (int n = 0; n < batch; n++) {
			const float* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
			output.block(oc * n, 0, oc, ohw).noalias() = kernel * im_col;
			output.block(oc * n, 0, oc, ohw).colwise() += bias;
		}
	}

	void Conv2d::backward(const MatXf& prev_out, MatXf& prev_delta)
	{
		for (int n = 0; n < batch; n++) {
			const float* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
			dkernel += delta.block(oc * n, 0, oc, ohw) * im_col.transpose();
			dbias += delta.block(oc * n, 0, oc, ohw).rowwise().sum();
		}

		if (!is_first) {
			for (int n = 0; n < batch; n++) {
				float* begin = prev_delta.data() + ic * ihw * n;
				im_col = kernel.transpose() * delta.block(oc * n, 0, oc, ohw);
				col2im(im_col.data(), ic, ih, iw, kh, 1, pad, begin);
			}
		}
	}

	void Conv2d::update_weight(float lr, float decay)
	{
		float t1 = (1 - (2 * lr * decay) / batch);
		float t2 = lr / batch;

		if (t1 != 1) {
			kernel *= t1;
			bias *= t1;
		}

		kernel -= t2 * dkernel;
		bias -= t2 * dbias;
	}

	void Conv2d::zero_grad()
	{
		delta.setZero();
		dkernel.setZero();
		dbias.setZero();
	}

	vector<int> Conv2d::output_shape() { return { batch, oc, oh, ow }; }

int main() {
    // Initialize input
    MatXf input(1, 224 * 224 * 3); // assuming batch size of 1 for simplicity
    input.setRandom();

    // Create Conv2D layer
    Conv2d conv(3, 64, 3, 1, "xavier_normal");
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
    MatXf grad_out = MatXf::Random(conv.output.rows(), conv.output.cols());

    // Set gradient for benchmarking backward pass
    conv.delta = grad_out;

    // Benchmark for backward pass
    MatXf prev_delta(input_shape[1], input_shape[2] * input_shape[3]);
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




