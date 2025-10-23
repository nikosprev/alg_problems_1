
#include <vector>
#include <random>


using Float = float;
using Vec = std::vector<Float>;

class KMeans {
public:
    struct Params {
        size_t k = 8;
        size_t max_iters = 100;
        size_t seed = 123;
    };

    explicit KMeans(const Params &p = Params());

    std::vector<Vec> fit(const Dataset &ds);

private:
    Params params;
    std::mt19937 rng;
};
