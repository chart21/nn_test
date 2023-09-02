#include <iostream>
#include <vector>
#include <chrono>
#include <eigen/Eigen/Dense>
#include <random>

using namespace std;
using namespace Eigen;

template<typename T>
using MatX = Matrix<T, Dynamic, Dynamic, RowMajor>;

template<typename T>
using VecX = Matrix<T, Dynamic, 1>;

template<typename T>
void init_weight(MatX<T>& W, int fan_in, int fan_out, string option) {
    unsigned seed = (unsigned)chrono::steady_clock::now().time_since_epoch().count();
    default_random_engine e(seed);

    if (option == "lecun_normal") {
        T s = sqrt(1.0 / fan_in);
        normal_distribution<T> dist(0, s);
        for_each(W.data(), W.data() + W.size(), [&](T& elem) { elem = dist(e); });
    } 
    // Add other initialization methods similar to the one above
    // ...
    else {
        cout << "Invalid initialization." << endl;
        exit(1);
    }
}

// ... [All other functions remain almost the same; replace `float` with `T`]

template<typename T>
class Layer {
    // ... [the class internals, replacing `float` with `T`]
};

template<typename T>
class Conv2d : public Layer<T> {
    // ... [the class internals, replacing `float` with `T`]
};

// Main function example for testing:
int main() {
    // Testing with float:
    MatX<float> input_float(1, 224 * 224 * 3); 
    input_float.setRandom();
    Conv2d<float> conv_float(3, 64, 3, 1, "xavier_normal");
    vector<int> input_shape_float = {1, 3, 224, 224};
    conv_float.set_layer(input_shape_float);
    conv_float.forward(input_float, true);

    // Testing with uint64_t:
    MatX<uint64_t> input_uint(1, 224 * 224 * 3); 
    input_uint.setRandom();
    Conv2d<uint64_t> conv_uint(3, 64, 3, 1, "xavier_normal");
    vector<int> input_shape_uint = {1, 3, 224, 224};
    conv_uint.set_layer(input_shape_uint);
    conv_uint.forward(input_uint, true);

    return 0;
}

