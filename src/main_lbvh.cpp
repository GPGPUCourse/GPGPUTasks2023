#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include "cl/nbody_cl.h"
#include "cl/lbvh_cl.h"
#include <libimages/images.h>
#include <functional>
#include <gtest/gtest.h>


////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////


// может понадобиться поменять индекс локально чтобы выбрать GPU если у вас более одного девайса
#define OPENCL_DEVICE_INDEX 0

// TODO включить чтобы начали запускаться тесты
#define ENABLE_TESTING 1

// имеет смысл отключать при оффлайн симуляции больших N, но в итоговом решении стоит оставить
#define EVALUATE_PRECISION 1

// удобно включить при локальном тестировании
#define ENABLE_GUI 1

// сброс картинок симуляции на диск
#define SAVE_IMAGES 0

// TODO на сервер лучше коммитить самую простую конфигурацию. Замеры по времени получатся нерелевантные, но зато быстрее отработает CI
// TODO локально интересны замеры на самой сложной версии, которую получится дождаться
// #define NBODY_INITIAL_STATE_COMPLEXITY 0
// #define NBODY_INITIAL_STATE_COMPLEXITY 1
#define NBODY_INITIAL_STATE_COMPLEXITY 2

// использовать lbvh для построения начального состояния. Нужно на очень больших N (>1000000)
#define ENABLE_LBVH_STATE_INITIALIZATION 1


////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////


struct Color
{
    unsigned char r, g, b;
};

const Color RED{255, 0, 0};
const Color GREEN{0, 255, 0};
const Color BLUE{0, 0, 255};
const Color CYAN{0, 255, 255};
const Color MAGENTA{255, 0, 255};
const Color YELLOW{255, 255, 0};
const Color BLACK{0, 0, 0};
const Color WHITE{255, 255, 255};

const double GRAVITATIONAL_FORCE = 0.0001;

struct DeltaState
{
    std::vector<float> dvx2d;
    std::vector<float> dvy2d;
};

struct State
{

    State() {
    }

    State(int N)
        : pxs(N)
          , pys(N)
          , vxs(N)
          , vys(N)
          , mxs(N)
          , coord_shift(0) {
    }

    std::vector<float> pxs;
    std::vector<float> pys;

    std::vector<float> vxs;
    std::vector<float> vys;

    std::vector<float> mxs;

    int coord_shift;
};

struct Point
{
    int x, y;

    bool operator==(const Point &rhs) const {
        return std::tie(x, y) == std::tie(rhs.x, rhs.y);
    }

    bool operator!=(const Point &rhs) const {
        return !(rhs == *this);
    }

    Point &operator+=(const Point &other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    Point &operator-=(const Point &other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    Point operator-(const Point &other) const {
        Point result = *this;
        result -= other;
        return result;
    }
};

void bresenham(std::vector<Point> &line_points, const Point &from, const Point &to) {
    line_points.clear();

    double x1 = from.x;
    double y1 = from.y;
    double x2 = to.x;
    double y2 = to.y;

    const bool steep = (fabs(y2 - y1) > fabs(x2 - x1));
    if (steep) {
        std::swap(x1, y1);
        std::swap(x2, y2);
    }
    const bool flip = x1 > x2;
    if (flip) {
        std::swap(x1, x2);
        std::swap(y1, y2);
    }

    const double dx = x2 - x1;
    const double dy = fabs(y2 - y1);

    double error = dx / 2.0;
    const int ystep = (y1 < y2) ? 1 : -1;
    int y = (int) y1;

    const int max_x = (int) x2;

    line_points.push_back(from);

    for (int x = (int) x1; x < max_x; x++) {
        Point point = (steep) ? Point{y, x} : Point{x, y};
        if (line_points.back() != point) {
            line_points.push_back(point);
        }

        error -= dy;
        if (error < 0) {
            y += ystep;
            error += dx;
        }
    }

    if (line_points.back() != to) {
        line_points.push_back(to);
    }

    if (line_points.size() > 2 && flip) {
        std::reverse(++line_points.begin(), --line_points.end());
    }
}

//Преобразует 0xbbbb в 0x0b0b0b0b
int spreadBits(int word) {
    word = (word ^ (word << 8)) & 0x00ff00ff;
    word = (word ^ (word << 4)) & 0x0f0f0f0f;
    word = (word ^ (word << 2)) & 0x33333333;
    word = (word ^ (word << 1)) & 0x55555555;
    return word;
}

using morton_t = uint64_t;
const int NBITS_PER_DIM = 16;
const int NBITS = NBITS_PER_DIM /*x dimension*/ + NBITS_PER_DIM /*y dimension*/ + 32 /*index augmentation*/;
//Convert xy coordinate to a 32 bit morton/z order code + 32 bit index augmentation for distinguishing between duplicates
morton_t zOrder(const Point &coord, int i) {
    if (coord.x < 0 || coord.x >= (1 << NBITS_PER_DIM))
        throw std::runtime_error("098245490432590890");
    if (coord.y < 0 || coord.y >= (1 << NBITS_PER_DIM))
        throw std::runtime_error("432764328764237823");

    int x = spreadBits(coord.x);
    int y = spreadBits(coord.y);

    morton_t morton_code = (y << 1) | x;
    return (morton_code << 32) | i;
}

#pragma pack (push, 1)

struct BBox
{

    BBox() {
        clear();
    }

    explicit BBox(const Point &point)
        : BBox() {
        grow(point);
    }

    BBox(float fx, float fy) {
        clear();
        grow(fx, fy);
    }

    void clear() {
        minx = std::numeric_limits<int>::max();
        maxx = std::numeric_limits<int>::lowest();
        miny = minx;
        maxy = maxx;
    }

    bool contains(const Point &point) const {
        return point.x >= minx && point.x <= maxx && point.y >= miny && point.y <= maxy;
    }

    bool contains(float fx, float fy) const {
        int x = fx + 0.5;
        int y = fy + 0.5;
        return x >= minx && x <= maxx && y >= miny && y <= maxy;
    }

    bool empty() const {
        return minx > maxx;
    }

    bool operator==(const BBox &other) const {
        return minx == other.minx && maxx == other.maxx && miny == other.miny && maxy == other.maxy;
    }

    bool operator!=(const BBox &other) const {
        return !(*this == other);
    }

    void grow(const Point &point) {
        minx = std::min(minx, point.x);
        maxx = std::max(maxx, point.x);
        miny = std::min(miny, point.y);
        maxy = std::max(maxy, point.y);
    }

    void grow(float fx, float fy) {
        minx = std::min(minx, int(fx + 0.5));
        maxx = std::max(maxx, int(fx + 0.5));
        miny = std::min(miny, int(fy + 0.5));
        maxy = std::max(maxy, int(fy + 0.5));
    }

    void grow(const BBox &other) {
        grow(other.min());
        grow(other.max());
    }

    int minX() const { return minx; }
    int maxX() const { return maxx; }
    int minY() const { return miny; }
    int maxY() const { return maxy; }

    Point min() const { return Point{minx, miny}; }
    Point max() const { return Point{maxx, maxy}; }

private:
    int minx, maxx;
    int miny, maxy;

};

bool feq(float x, float y) {
    float maxXY = std::max(1e-8f, std::max(std::fabs(x) , std::fabs(y)));
    return std::fabs(x - y) < 1e-3f * maxXY;
}
struct Node
{

    bool hasLeftChild() const { return child_left >= 0; }
    bool hasRightChild() const { return child_right >= 0; }
    bool isLeaf() const { return !hasLeftChild() && !hasRightChild(); }

    bool operator==(const Node &other) const {
        return child_left == other.child_left &&  child_right == other.child_right && bbox ==  other.bbox &&
            feq(mass, other.mass) && feq(cmsx, other.cmsx) && feq(cmsy,other.cmsy);
    }

    bool operator!=(const Node &other) const {
        return !(*this == other);
    }

    int child_left, child_right;
    BBox bbox;

    // used only for nbody
    float mass;
    float cmsx;
    float cmsy;
};
#pragma pack (pop)

morton_t getBits(morton_t morton_code, int bit_index, int prefix_size) {
    return (morton_code >> bit_index) & ((1ull << prefix_size) - 1ull);
}

int getBit(morton_t morton_code, int bit_index) {
    return (morton_code >> bit_index) & 1;
}

// из аугментированного мортоновского кода можно извлечь индекс в изначальном массиве
int getIndex(morton_t morton_code) {
    morton_t mask = 1;
    mask = (mask << 32) - 1;
    return morton_code & mask;
}

// N листьев, N-1 внутренних нод
int LBVHSize(int N) {
    return N + N - 1;
}

using points_mass_functor = std::function<std::tuple<float, float, float>(int)>;

Point makePoint(float x, float y) {
    if (x < 0.f || y < 0.f) {
        throw std::runtime_error("0432959435934534");
    }
    return Point{int(x + 0.5f), int(y + 0.5f)};
}

void initLBVHNode(std::vector<Node> &nodes, int i_node, const std::vector<morton_t> &codes,
                  const points_mass_functor &points_mass_array);
void buildBBoxes(std::vector<Node> &nodes, std::vector<int> &flags, int N, bool use_omp = false);
void buildBBoxesRecursive(std::vector<Node> &nodes, Node &root);
void drawLBVH(images::Image<unsigned char> &canvas, const std::vector<Node> &nodes, int coord_shift = 0);

using interactive_callback_t = std::function<void(const std::vector<float> &, const std::vector<float> &,
                                                  const std::vector<Node> &)>;

// https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation
bool barnesHutCondition(float x, float y, const Node &node) {
    float dx = x - node.cmsx;
    float dy = y - node.cmsy;
    float s = std::max(node.bbox.maxX() - node.bbox.minX(), node.bbox.maxY() - node.bbox.minY());
    float d2 = dx * dx + dy * dy;
    float thresh = 0.5;

    // возвращаем true, если находимся от ноды достаточно далеко, чтобы можно было ее считать примерно точечной

    return s * s < d2 * thresh * thresh;
}

void calculateForce(float x0, float y0, float m0, const std::vector<Node> &nodes, float *force_x, float *force_y) {
    // основная идея ускорения - аггрегировать в узлах дерева веса и центры масс,
    //   и не спускаться внутрь, если точка запроса не пересекает ноду, а заменить на взаимодействие с ее центром масс

    int stack[2 * NBITS_PER_DIM];
    int stack_size = 0;
    stack[stack_size++] = 0;
    while (stack_size) {
        int i_node = stack[--stack_size];
        auto node = nodes[i_node];

        if (node.isLeaf()) {
            continue;
        }

        // если запрос содержится и а левом и в правом ребенке - то они в одном пикселе
        {
            const Node &left = nodes[node.child_left];
            const Node &right = nodes[node.child_right];
            if (left.bbox.contains(x0, y0) && right.bbox.contains(x0, y0)) {
                if (left.bbox != right.bbox) {
                    throw std::runtime_error("42357987645432456547");
                }
                if (left.bbox != BBox(x0, y0)) {
                    throw std::runtime_error("5446456456435656");
                }
                continue;
            }
        }
        for (int i_child : {node.child_left, node.child_right}) {
            const Node &child = nodes[i_child];
            // С точки зрения ббоксов заходить в ребенка, ббокс которого не пересекаем, не нужно (из-за того, что в листьях у нас точки и они не высовываются за свой регион пространства)
            //   Но, с точки зрения физики, замена гравитационного влияния всех точек в регионе на взаимодействие с суммарной массой в центре масс - это точное решение только в однородном поле (например, на поверхности земли)
            //   У нас поле неоднородное, и такая замена - лишь приближение. Чтобы оно было достаточно точным, будем спускаться внутрь ноды, пока она не станет похожа на точечное тело (маленький размер ее ббокса относительно нашего расстояния до центра масс ноды)
            if (!child.bbox.contains(x0, y0) && barnesHutCondition(x0, y0, child)) {
                float dx = child.cmsx - x0;
                float dy = child.cmsy - y0;
                float dr2 = std::max(100.f, dx * dx + dy * dy);

                float dr2_inv = 1.f / dr2;
                float dr_inv = sqrt(dr2_inv);

                float ex = dx * dr_inv;
                float ey = dy * dr_inv;

                float fx = ex * dr2_inv * GRAVITATIONAL_FORCE;
                float fy = ey * dr2_inv * GRAVITATIONAL_FORCE;

                *force_x += child.mass * fx;
                *force_y += child.mass * fy;
            } else {
                stack[stack_size++] = i_child;
                if (stack_size >= 2 * NBITS_PER_DIM) {
                    throw std::runtime_error("0420392384283");
                }
            }
        }
    }
}

void integrate(int i, std::vector<float> &pxs, std::vector<float> &pys, std::vector<float> &vxs,
               std::vector<float> &vys, float *dvx, float *dvy, int coord_shift) {
    vxs[i] += dvx[i];
    vys[i] += dvy[i];
    pxs[i] += vxs[i];
    pys[i] += vys[i];

    // отражаем частицы от границ мира, чтобы не ломался подсчет мортоновского кода
    if (pxs[i] < 1) {
        vxs[i] *= -1;
        pxs[i] += vxs[i];
    }
    if (pys[i] < 1) {
        vys[i] *= -1;
        pys[i] += vys[i];
    }
    if (pxs[i] >= 2 * coord_shift - 1) {
        vxs[i] *= -1;
        pxs[i] += vxs[i];
    }
    if (pys[i] >= 2 * coord_shift - 1) {
        vys[i] *= -1;
        pys[i] += vys[i];
    }
}

// in: initial conditions, out: 2D array for integration
void nbody_cpu_lbvh(DeltaState &delta_state, State &initial_state, int N, int NT,
                    const interactive_callback_t *interactive_callback = nullptr) {
    int NT_interactive = interactive_callback ? 1 : NT;

    delta_state.dvx2d.assign(N * NT_interactive, 0.0);
    delta_state.dvy2d.assign(N * NT_interactive, 0.0);

    std::vector<float> &pxs = initial_state.pxs;
    std::vector<float> &pys = initial_state.pys;

    std::vector<float> &vxs = initial_state.vxs;
    std::vector<float> &vys = initial_state.vys;

    const std::vector<float> &mxs = initial_state.mxs;

    const int tree_size = LBVHSize(N);
    std::vector<Node> nodes(tree_size);
    std::vector<morton_t> codes(N);
    std::vector<int> buffer;

    const points_mass_functor points_mass_array = [&](int i) { return std::make_tuple(pxs[i], pys[i], mxs[i]); };

    for (int t = 0; t < NT; ++t) {
        int t_interactive = interactive_callback ? 0 : t;

        float *dvx = &delta_state.dvx2d[t_interactive * N];
        float *dvy = &delta_state.dvy2d[t_interactive * N];

        // инициализируем мортоновские коды
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            codes[i] = zOrder(makePoint(pxs[i], pys[i]), i);
        }

        // упорядочиваем тела по z-curve
        std::sort(codes.begin(), codes.end());

        // строим LBVH
        #pragma omp parallel for
        for (int i_node = 0; i_node < tree_size; ++i_node) {
            initLBVHNode(nodes, i_node, codes, points_mass_array);
        }

        // инициализируем ббоксы и массы
        buildBBoxes(nodes, buffer, N, false/*omp here can cause stuttering on start of simulation*/);

        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            float x0 = pxs[i];
            float y0 = pys[i];
            float m0 = mxs[i];

            calculateForce(x0, y0, m0, nodes, &dvx[i], &dvy[i]);
        }

        if (interactive_callback) {
            (*interactive_callback)(pxs, pys, nodes);
        }

        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            integrate(i, pxs, pys, vxs, vys, dvx, dvy, initial_state.coord_shift);
        }

        if (interactive_callback) {
            delta_state.dvx2d.assign(N * NT_interactive, 0.0);
            delta_state.dvy2d.assign(N * NT_interactive, 0.0);
        }
    }
}

// in: initial conditions, out: 2D array for integration
void nbody_cpu(DeltaState &delta_state, State &initial_state, int N, int NT,
               const interactive_callback_t *interactive_callback = nullptr) {
    int NT_interactive = interactive_callback ? 1 : NT;

    delta_state.dvx2d.assign(N * NT_interactive, 0.0);
    delta_state.dvy2d.assign(N * NT_interactive, 0.0);

    std::vector<float> &pxs = initial_state.pxs;
    std::vector<float> &pys = initial_state.pys;

    std::vector<float> &vxs = initial_state.vxs;
    std::vector<float> &vys = initial_state.vys;

    const std::vector<float> &mxs = initial_state.mxs;

    for (int t = 0; t < NT; ++t) {
        int t_interactive = interactive_callback ? 0 : t;

        float *dvx = &delta_state.dvx2d[t_interactive * N];
        float *dvy = &delta_state.dvy2d[t_interactive * N];

        // to be kernel 1
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            float x0 = pxs[i];
            float y0 = pys[i];
            float m0 = mxs[i];
            for (int j = 0; j < N; ++j) {

                if (j == i) {
                    continue;
                }

                float x1 = pxs[j];
                float y1 = pys[j];
                float m1 = mxs[j];

                float dx = x1 - x0;
                float dy = y1 - y0;
                float dr2 = std::max(100.f, dx * dx + dy * dy);

                float dr2_inv = 1.f / dr2;
                float dr_inv = std::sqrt(dr2_inv);

                float ex = dx * dr_inv;
                float ey = dy * dr_inv;

                float fx = ex * dr2_inv * GRAVITATIONAL_FORCE;
                float fy = ey * dr2_inv * GRAVITATIONAL_FORCE;

                dvx[i] += m1 * fx;
                dvy[i] += m1 * fy;
            }
        }

        if (interactive_callback) {
            (*interactive_callback)(pxs, pys, {});
        }

        // to be kernel 2
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            vxs[i] += dvx[i];
            vys[i] += dvy[i];
            pxs[i] += vxs[i];
            pys[i] += vys[i];
        }

        if (interactive_callback) {
            delta_state.dvx2d.assign(N * NT_interactive, 0.0);
            delta_state.dvy2d.assign(N * NT_interactive, 0.0);
        }
    }
}

void nbody_gpu_lbvh(DeltaState &delta_state, State &initial_state, int N, int NT,
                    const interactive_callback_t *interactive_callback = nullptr) {
    int tree_size = LBVHSize(N);

    std::vector<Node> nodes(tree_size);

    int NT_interactive = interactive_callback ? 1 : NT;

    delta_state.dvx2d.assign(N * NT_interactive, 0.0);
    delta_state.dvy2d.assign(N * NT_interactive, 0.0);

    std::vector<float> &pxs = initial_state.pxs;
    std::vector<float> &pys = initial_state.pys;

    std::vector<float> &vxs = initial_state.vxs;
    std::vector<float> &vys = initial_state.vys;

    const std::vector<float> &mxs = initial_state.mxs;

    std::vector<float> &dvx2d = delta_state.dvx2d;
    std::vector<float> &dvy2d = delta_state.dvy2d;

    unsigned int workGroupSize = 32;
    unsigned int global_work_size_points = (N + workGroupSize - 1) / workGroupSize * workGroupSize;
    unsigned int global_work_size_nodes = (LBVHSize(N) + workGroupSize - 1) / workGroupSize * workGroupSize;

    ocl::Kernel kernel_generate_morton_codes(lbvh_kernel, lbvh_kernel_length, "generateMortonCodes");
    ocl::Kernel kernel_merge(lbvh_kernel, lbvh_kernel_length, "merge");
    ocl::Kernel kernel_build_lbvh(lbvh_kernel, lbvh_kernel_length, "buidLBVH");
    ocl::Kernel kernel_init_flags(lbvh_kernel, lbvh_kernel_length, "initFlags");
    ocl::Kernel kernel_grow_nodes(lbvh_kernel, lbvh_kernel_length, "growNodes");
    ocl::Kernel kernel_calculate_forces(lbvh_kernel, lbvh_kernel_length, "calculateForces");
    ocl::Kernel kernel_integrate(lbvh_kernel, lbvh_kernel_length, "integrate");

    kernel_generate_morton_codes.compile();
    kernel_merge.compile();
    kernel_build_lbvh.compile();
    kernel_init_flags.compile();
    kernel_grow_nodes.compile();
    kernel_calculate_forces.compile();
    kernel_integrate.compile();

    gpu::gpu_mem_32f pxs_gpu, pys_gpu, vxs_gpu, vys_gpu, mxs_gpu;
    gpu::gpu_mem_32f dvx2d_gpu, dvy2d_gpu;

    gpu::gpu_mem_64u codes_gpu;
    gpu::gpu_mem_64u codes_gpu_buf;
    gpu::gpu_mem_any nodes_gpu;
    gpu::gpu_mem_32i flags_gpu;

    pxs_gpu.resizeN(N);
    pys_gpu.resizeN(N);
    vxs_gpu.resizeN(N);
    vys_gpu.resizeN(N);
    mxs_gpu.resizeN(N);

    dvx2d_gpu.resizeN(N * NT_interactive);
    dvy2d_gpu.resizeN(N * NT_interactive);

    codes_gpu.resizeN(N);
    codes_gpu_buf.resizeN(N);
    nodes_gpu.resize(tree_size * sizeof(Node));
    flags_gpu.resizeN(N);

    pxs_gpu.writeN(pxs.data(), N);
    pys_gpu.writeN(pys.data(), N);
    vxs_gpu.writeN(vxs.data(), N);
    vys_gpu.writeN(vys.data(), N);
    mxs_gpu.writeN(mxs.data(), N);

    dvx2d_gpu.writeN(dvx2d.data(), N * NT_interactive);
    dvy2d_gpu.writeN(dvy2d.data(), N * NT_interactive);

    for (int t = 0; t < NT; ++t) {
        int t_interactive = interactive_callback ? 0 : t;

        // generate morton codes
        kernel_generate_morton_codes.exec(gpu::WorkSize(workGroupSize, global_work_size_points),
                                          pxs_gpu, pys_gpu,
                                          codes_gpu,
                                          N);

        // sort morton codes
        for (unsigned int subn = 1; subn < N; subn *= 2) {
            kernel_merge.exec(gpu::WorkSize(workGroupSize, global_work_size_points), codes_gpu, codes_gpu_buf, N, subn);
            codes_gpu.swap(codes_gpu_buf);
        }

        // build child pointers for lbvh nodes
        kernel_build_lbvh.exec(gpu::WorkSize(workGroupSize, global_work_size_nodes),
                               pxs_gpu, pys_gpu, mxs_gpu,
                               codes_gpu, nodes_gpu,
                               N);

        // propagate bbox and mass info from leaves
        for (int level = 0; level < NBITS; ++level) {
            kernel_init_flags.exec(gpu::WorkSize(workGroupSize, global_work_size_points),
                                   flags_gpu, nodes_gpu,
                                   N, level);

            kernel_grow_nodes.exec(gpu::WorkSize(workGroupSize, global_work_size_points),
                                   flags_gpu, nodes_gpu,
                                   N, level);

            int n_updated;
            flags_gpu.readN(&n_updated, 1, N - 1);

            if (!n_updated)
                break;
        }

        kernel_calculate_forces.exec(gpu::WorkSize(workGroupSize, global_work_size_points),
                                     pxs_gpu, pys_gpu, vxs_gpu, vys_gpu,
                                     mxs_gpu, nodes_gpu,
                                     dvx2d_gpu, dvy2d_gpu,
                                     N, t_interactive);

        if (interactive_callback) {
            pxs_gpu.readN(pxs.data(), N);
            pys_gpu.readN(pys.data(), N);
            nodes_gpu.read(nodes.data(), tree_size * sizeof(Node));
            (*interactive_callback)(pxs, pys, nodes);
        }

        kernel_integrate.exec(gpu::WorkSize(workGroupSize, global_work_size_points),
                              pxs_gpu, pys_gpu, vxs_gpu, vys_gpu,
                              mxs_gpu,
                              dvx2d_gpu, dvy2d_gpu,
                              N, t_interactive, initial_state.coord_shift);

        if (interactive_callback) {
            dvx2d_gpu.writeN(dvx2d.data(), N * NT_interactive);
            dvy2d_gpu.writeN(dvy2d.data(), N * NT_interactive);
        }
    }

    dvx2d_gpu.readN(dvx2d.data(), N * NT_interactive);
    dvy2d_gpu.readN(dvy2d.data(), N * NT_interactive);
}

void nbody_gpu(DeltaState &delta_state, State &initial_state, int N, int NT,
               const interactive_callback_t *interactive_callback = nullptr) {
    int NT_interactive = interactive_callback ? 1 : NT;

    delta_state.dvx2d.assign(N * NT_interactive, 0.0);
    delta_state.dvy2d.assign(N * NT_interactive, 0.0);

    std::vector<float> &pxs = initial_state.pxs;
    std::vector<float> &pys = initial_state.pys;

    std::vector<float> &vxs = initial_state.vxs;
    std::vector<float> &vys = initial_state.vys;

    const std::vector<float> &mxs = initial_state.mxs;

    std::vector<float> &dvx2d = delta_state.dvx2d;
    std::vector<float> &dvy2d = delta_state.dvy2d;

    unsigned int workGroupSize = 32;
    unsigned int global_work_size = (N + workGroupSize - 1) / workGroupSize * workGroupSize;
    ocl::Kernel kernel_calculate_force_global(nbody_kernel, nbody_kernel_length, "nbody_calculate_force_global");
    ocl::Kernel kernel_integrate(nbody_kernel, nbody_kernel_length, "nbody_integrate");
    gpu::gpu_mem_32f pxs_gpu, pys_gpu, vxs_gpu, vys_gpu, mxs_gpu;
    gpu::gpu_mem_32f dvx2d_gpu, dvy2d_gpu;

    kernel_calculate_force_global.compile();
    kernel_integrate.compile();

    pxs_gpu.resizeN(N);
    pys_gpu.resizeN(N);
    vxs_gpu.resizeN(N);
    vys_gpu.resizeN(N);
    mxs_gpu.resizeN(N);

    dvx2d_gpu.resizeN(N * NT_interactive);
    dvy2d_gpu.resizeN(N * NT_interactive);

    pxs_gpu.writeN(pxs.data(), N);
    pys_gpu.writeN(pys.data(), N);
    vxs_gpu.writeN(vxs.data(), N);
    vys_gpu.writeN(vys.data(), N);
    mxs_gpu.writeN(mxs.data(), N);

    dvx2d_gpu.writeN(dvx2d.data(), N * NT_interactive);
    dvy2d_gpu.writeN(dvy2d.data(), N * NT_interactive);

    for (int t = 0; t < NT; ++t) {
        int t_interactive = interactive_callback ? 0 : t;

        kernel_calculate_force_global.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                           pxs_gpu, pys_gpu, vxs_gpu, vys_gpu,
                                           mxs_gpu,
                                           dvx2d_gpu, dvy2d_gpu,
                                           N, t_interactive);

        if (interactive_callback) {
            pxs_gpu.readN(pxs.data(), N);
            pys_gpu.readN(pys.data(), N);
            (*interactive_callback)(pxs, pys, {});
        }

        kernel_integrate.exec(gpu::WorkSize(workGroupSize, global_work_size),
                              pxs_gpu, pys_gpu, vxs_gpu, vys_gpu,
                              mxs_gpu,
                              dvx2d_gpu, dvy2d_gpu,
                              N, t_interactive);

        if (interactive_callback) {
            dvx2d_gpu.writeN(dvx2d.data(), N * NT_interactive);
            dvy2d_gpu.writeN(dvy2d.data(), N * NT_interactive);
        }
    }

    dvx2d_gpu.readN(dvx2d.data(), N * NT_interactive);
    dvy2d_gpu.readN(dvy2d.data(), N * NT_interactive);
}

State makeSimpleState() {
    State state;
    state.pxs = {-25.f, 25.f, 100.f};
    state.pys = {-0.f, 0.f, 0.f};
    state.vxs = {-1.f, -0.f, 0.f};
    state.vys = {0.f, -1.f, 0.f};
    state.mxs = {20.f, 20.f, 50.f};
    return state;
}

State makeRandomState(int N, int minx, int maxx, int miny, int maxy) {
    State result(N);

    int w = maxx - minx;
    int h = maxy - miny;

    for (int i = 0; i < N; ++i) {
        result.pxs[i] = std::rand() % w + minx;
        result.pys[i] = std::rand() % h + miny;

        result.vxs[i] = 0;
        std::rand() % 3 - 1;
        result.vys[i] = 0;
        std::rand() % 3 - 1;

        result.mxs[i] = std::rand() % 20 + 1;
    }

    return result;
}

State makeCircularState() {
    const int N = 1000;

    State result(N);

    for (int i = 0; i < N - 1; ++i) {
        int r = 20;
        // circle sampling
        float angle = 3.14159 * 2 / (N - 1) * i;
        int x = r * std::sin(angle);
        int y = r * std::cos(angle);
        result.pxs[i] = x;
        result.pys[i] = y;
        result.vxs[i] = 0;
        result.vys[i] = 0;
        result.mxs[i] = 0.05;
    }

    result.pxs.back() = 0;
    result.pys.back() = 0;
    result.vxs.back() = 0;
    result.vys.back() = 0;
    result.mxs.back() = 1000000;

    DeltaState delta_state;
    auto tmp = result;
    nbody_cpu(delta_state, tmp, N, 1);

    for (int i = 0; i < N - 1; ++i) {
        float rx = result.pxs[i];
        float ry = result.pys[i];
        float r = std::sqrt(rx * rx + ry * ry);
        {
            rx /= r;
            ry /= r;
        }

        float fx = delta_state.dvx2d[i];
        float fy = delta_state.dvy2d[i];

        float f = fx * rx + fy * ry;
        float v = std::sqrt(std::abs(r * f));

        result.vxs[i] = ry * v;
        result.vys[i] = -rx * v;
    }

    return result;
}

State makeGalacticState(int N, int s0, int s1) {
    State result(N);

    int minx = -s1;
    int maxx = +s1;
    int miny = -s1;
    int maxy = +s1;

    int w = maxx - minx;
    int h = maxy - miny;

    double total_mass = 0;
    for (int i = 0; i < N - 1; ++i) {
        int rr;
        do {
            result.pxs[i] = std::rand() % w + minx;
            result.pys[i] = std::rand() % h + miny;
            rr = result.pxs[i] * result.pxs[i] + result.pys[i] * result.pys[i];
        } while (rr > s1 * s1 || rr < s0 * s0);

        result.vxs[i] = 0;
        result.vys[i] = 0;

        result.mxs[i] = 0.001 * (std::rand() % 10000 + 1);
        total_mass += result.mxs[i];
    }

    // spawn black hole
    result.pxs.back() = 0;
    result.pys.back() = 0;
    result.vxs.back() = 0;
    result.vys.back() = 0;
    result.mxs.back() = 8.0 * total_mass;

    DeltaState delta_state;
    auto tmp = result;
    // для больших N (миллион) дождаться даже одного брутфорс шага проблематично
    #if ENABLE_LBVH_STATE_INITIALIZATION
    {
        tmp.coord_shift = 1 << (NBITS_PER_DIM - 1);
        for (int i = 0; i < N; ++i) {
            tmp.pxs[i] += tmp.coord_shift;
            tmp.pys[i] += tmp.coord_shift;
        }
        nbody_cpu_lbvh(delta_state, tmp, N, 1);
    }
    #else
    nbody_cpu(delta_state, tmp, N, 1);
    #endif

    for (int i = 0; i < N - 1; ++i) {
        float rx = result.pxs[i];
        float ry = result.pys[i];
        float r = std::sqrt(rx * rx + ry * ry);
        {
            rx /= r;
            ry /= r;
        }

        float fx = delta_state.dvx2d[i];
        float fy = delta_state.dvy2d[i];

        float f = fx * rx + fy * ry;
        float v = std::sqrt(std::abs(r * f));

        result.vxs[i] = ry * v;
        result.vys[i] = -rx * v;
    }

    return result;
}

int findSplit(const std::vector<morton_t> &codes, int i_begin, int i_end, int bit_index) {
    // Если биты в начале и в конце совпадают, то этот бит незначащий
    if (getBit(codes[i_begin], bit_index) == getBit(codes[i_end - 1], bit_index)) {
        return -1;
    }

    // наивная версия, линейный поиск, можно использовать для отладки бинпоиска
    // int  linear_i = 0;
    //    for (int i = i_begin + 1; i < i_end; ++i) {
    //        int a = getBit(codes[i-1], bit_index);
    //        int b = getBit(codes[i], bit_index);
    //        if (a < b) {
    //            linear_i = i;
    //        }
    //    }

    int start = i_begin;
    while (i_begin < i_end) {
        int mid = (i_begin + i_end) / 2;
        if (getBit(codes[start], bit_index) != getBit(codes[mid], bit_index))
            i_end = mid;
        else
            i_begin = mid + 1;
    }
    return i_begin;
}

void buildLBVHRecursive(std::vector<Node> &nodes, const std::vector<morton_t> &codes, const std::vector<Point> &points,
                        int i_begin, int i_end, int bit_index) {
    int i_node = nodes.size();
    nodes.emplace_back();

    // leaf
    if (i_begin + 1 == i_end) {
        nodes[i_node].child_left = -1;
        nodes[i_node].child_right = -1;
        auto pt = points[getIndex(codes[i_begin])];
        nodes[i_node].bbox.grow(pt);
        nodes[i_node].mass = 1.f;
        nodes[i_node].cmsx = pt.x;
        nodes[i_node].cmsy = pt.y;
        return;
    }

    for (int i_bit = bit_index; i_bit >= 0; --i_bit) {
        int split = findSplit(codes, i_begin, i_end, i_bit);
        if (split < 0)
            continue;

        nodes[i_node].child_left = nodes.size();
        buildLBVHRecursive(nodes, codes, points, i_begin, split, i_bit - 1);

        nodes[i_node].child_right = nodes.size();
        buildLBVHRecursive(nodes, codes, points, split, i_end, i_bit - 1);

        //        std::cout << "push non leaf" << std::endl;
        return;
    }

    throw std::runtime_error("043242304023: potentially found duplicate morton code");
}

void findRegion(int *i_begin, int *i_end, int *bit_index, const std::vector<morton_t> &codes, int i_node) {
    int N = codes.size();
    if (i_node < 1 || i_node > N - 2) {
        throw std::runtime_error("842384298293482");
    }

    // 1. найдем, какого типа мы граница: левая или правая. Идем от самого старшего бита и паттерн-матчим тройки соседних битов
    //  если нашли (0, 0, 1), то мы правая граница, если нашли (0, 1, 1), то мы левая
    // dir: 1 если мы левая граница и -1 если правая
    int dir = 0;
    int i_bit = NBITS - 1;
    for (; i_bit >= 0; --i_bit) {
        if (getBit(codes[i_node - 1], i_bit) == 0 && getBit(codes[i_node + 1], i_bit) == 1) {
            dir = getBit(codes[i_node], i_bit) == 0 ? -1 : 1;
            break;
        }
    }

    if (dir == 0) {
        throw std::runtime_error("8923482374983");
    }

    // 2. Найдем вторую границу нашей зоны ответственности

    // количество совпадающих бит в префиксе
    int K = NBITS - i_bit;
    morton_t pref0 = getBits(codes[i_node], i_bit, K);

    // граница зоны ответственности - момент, когда префикс перестает совпадать
    int i_node_end = -1;
    // наивная версия, линейный поиск, можно использовать для отладки бинпоиска
    // for (int i = i_node; i >= 0 && i < int(codes.size()); i += dir) {
    //     if (getBits(codes[i], i_bit, K) == pref0) {
    //         i_node_end = i;
    //     } else {
    //         break;
    //     }
    // }
    // if (i_node_end == -1) {
    //     throw std::runtime_error("47248457284332098");
    // }
    int start = i_node, end = dir == 1 ? N : -1;
    while (start + dir != end) {
        int mid = start + (end - start) / 2;
        if (getBits(codes[mid], i_bit, K) == pref0) {
            start = mid;
        } else {
            end = mid;
        }
    }
    i_node_end = start;
    *bit_index = i_bit - 1;

    if (dir > 0) {
        *i_begin = i_node;
        *i_end = i_node_end + 1;
    } else {
        *i_begin = i_node_end;
        *i_end = i_node + 1;
    }
}

void initLBVHNode(std::vector<Node> &nodes, int i_node, const std::vector<morton_t> &codes,
                  const points_mass_functor &points_mass_array) {
    // инициализация ссылок на соседей для нод lbvh
    // если мы лист, то просто инициализируем минус единицами (нет детей), иначе ищем своб зону ответственности и запускаем на ней findSplit
    // можно заполнить пропуски в виде тудушек, можно реализовать с чистого листа самостоятельно, если так проще

    nodes[i_node].bbox.clear();
    nodes[i_node].mass = 0;
    nodes[i_node].cmsx = 0;
    nodes[i_node].cmsy = 0;

    const int N = codes.size();

    // первые N-1 элементов - внутренние ноды, за ними N листьев

    // инициализируем лист
    if (i_node >= N - 1) {
        nodes[i_node].child_left = -1;
        nodes[i_node].child_right = -1;
        int i_point = i_node - (N - 1);

        float center_mass_x, center_mass_y;
        float mass;
        std::tie(center_mass_x, center_mass_y, mass) = points_mass_array(getIndex(codes[i_point]));

        nodes[i_node].bbox.grow(makePoint(center_mass_x, center_mass_y));
        nodes[i_node].cmsx = center_mass_x;
        nodes[i_node].cmsy = center_mass_y;
        nodes[i_node].mass = mass;

        return;
    }

    // инициализируем внутреннюю ноду

    int i_begin = 0, i_end = N, bit_index = NBITS - 1;
    // если рассматриваем не корень, то нужно найти зону ответственности ноды и самый старший бит, с которого надо начинать поиск разреза
    if (i_node) {
        findRegion(&i_begin, &i_end, &bit_index, codes, i_node);
    }

    bool found = false;
    for (int i_bit = bit_index; i_bit >= 0; --i_bit) {
        int split = findSplit(codes, i_begin, i_end, i_bit);
        if (split < 0)
            continue;

        if (split < 1) {
            throw std::runtime_error("043204230042342");
        }


        // проинициализировать nodes[i_node].child_left, nodes[i_node].child_right на основе i_begin, i_end, split
        nodes[i_node].child_left = split - i_begin < 2 ? split - 1 + (N - 1) : split - 1;
        nodes[i_node].child_right = i_end - split < 2 ? split + (N - 1) : split;
        //   не забудьте на N-1 сдвинуть индексы, указывающие на листья

        found = true;
        break;
    }

    if (!found) {
        throw std::runtime_error("54356549645");
    }
}

void buildLBVH(std::vector<Node> &nodes, const std::vector<morton_t> &codes, const std::vector<Point> &points) {
    const int N = codes.size();
    int tree_size = LBVHSize(N);
    nodes.resize(tree_size);

    const points_mass_functor points_mass_array = [&](int i) {
        return std::make_tuple((float) points[i].x, (float) points[i].y, 1.f);
    };

    // можно раскомментировать и будет работать, но для дебага удобнее оставить однопоточную версию
    #pragma omp parallel for
    for (int i_node = 0; i_node < tree_size; ++i_node) {
        initLBVHNode(nodes, i_node, codes, points_mass_array);
    }
}

void printMortonCodes(const std::vector<morton_t> &codes) {
    std::cout << "morton codes: \n";
    for (int bit_index = NBITS - 1; bit_index >= 0; --bit_index) {
        for (int i = 0; i < (int) codes.size(); ++i) {
            int bit = getBit(codes[i], bit_index);
            std::cout << bit << "  ";
        }
        std::cout << "\n";
    }
    std::cout << std::flush;
}

void growNode(Node &root, std::vector<Node> &nodes) {
    const Node &left = nodes[root.child_left];
    const Node &right = nodes[root.child_right];

    root.bbox.grow(left.bbox);
    root.bbox.grow(right.bbox);

    double m0 = left.mass;
    double m1 = right.mass;

    root.mass = m0 + m1;

    if (root.mass <= 1e-8) {
        throw std::runtime_error("04230420340322");
    }

    root.cmsx = (left.cmsx * m0 + right.cmsx * m1) / root.mass;
    root.cmsy = (left.cmsy * m0 + right.cmsy * m1) / root.mass;
}

void buildBBoxesRecursive(std::vector<Node> &nodes, Node &root) {
    if (root.isLeaf())
        return;

    buildBBoxesRecursive(nodes, nodes[root.child_left]);
    buildBBoxesRecursive(nodes, nodes[root.child_right]);

    growNode(root, nodes);
}

void initFlag(std::vector<int> &flags, int i_node, std::vector<Node> &nodes, int level) {
    flags[i_node] = -1;

    Node &node = nodes[i_node];
    if (node.isLeaf()) {
        throw std::runtime_error("9423584385834");
    }

    if (!node.bbox.empty()) {
        return;
    }

    const BBox &left = nodes[node.child_left].bbox;
    const BBox &right = nodes[node.child_right].bbox;

    if (!left.empty() && !right.empty()) {
        flags[i_node] = level;
    }
}

void buildBBoxes(std::vector<Node> &nodes, std::vector<int> &flags, int N, bool use_omp) {
    flags.resize(N - 1);

    // NBITS раз проходимся по всему дереву и инициализируем только те ноды, у которых проинициализированы ббоксы обоих детей
    //   не самый оптимальный вариант (O(NlogN) вместо O(N)), зато легко переложить на GPU
    for (int level = 0; level < NBITS; ++level) {
        // знаем, что листья располагаются после N-1 внутренних нод дерева, а для них уже ббоксы посчитаны -> можно пропустить
        // не сработает для рекурсивно построенного дерева, там такого порядка не вводили

        // чтобы не было гонки в многопоточном режиме (и, по аналогии, потом на видеокарте), в первом проходе отметим ноды, которые нужно обновить, и только на втором проходе обновим
        #pragma omp parallel for if(use_omp)
        for (int i_node = 0; i_node < N - 1; ++i_node) {
            initFlag(flags, i_node, nodes, level);
        }

        int n_updated = 0;
        #pragma omp parallel for if(use_omp) reduction(+:n_updated)
        for (int i_node = 0; i_node < N - 1; ++i_node) {
            // если находимся на нужном уровне (нужный flag), проинициализируем ббокс и центр масс ноды
            if (flags[i_node] != level)
                continue;
            auto &node = nodes[i_node];
            auto &left_node = nodes[node.child_left], &right_node = nodes[node.child_right];
            node.bbox.clear();
            node.bbox.grow(left_node.bbox);
            node.bbox.grow(right_node.bbox);
            node.mass = left_node.mass + right_node.mass;
            node.cmsx = (left_node.cmsx * left_node.mass + right_node.cmsx * right_node.mass) / node.mass;
            node.cmsy = (left_node.cmsy * left_node.mass + right_node.cmsy * right_node.mass) / node.mass;
            ++n_updated;
        }

        //        std::cout << "n updated: " << n_updated << std::endl;

        // если глубина небольшая, то раньше закончим
        if (!n_updated) {
            break;
        }
    }
}

void drawLBVH(images::Image<unsigned char> &canvas, const std::vector<Node> &nodes, int coord_shift) {
    #pragma omp parallel for
    for (int y = 0; y < (int) canvas.height; ++y) {
        for (int x = 0; x < (int) canvas.width; ++x) {

            Point point{x + coord_shift, y + coord_shift};

            int depth = 0;

            int i_node = 0;
            while (true) {
                const Node &node = nodes[i_node];

                if (node.isLeaf()) {
                    break;
                }

                if (!node.bbox.contains(point)) {
                    break;
                }

                const Node &left = nodes[node.child_left];
                const Node &right = nodes[node.child_right];
                bool contains_left = left.bbox.contains(point);
                bool contains_right = right.bbox.contains(point);

                if (contains_left && contains_right) {
                    if (left.bbox != right.bbox) {
                        throw std::runtime_error("0320423949293423");
                    }
                    if (left.bbox != BBox(point)) {
                        throw std::runtime_error("867932433412341");
                    }
                    // left and right children are at the same pixel -> no need to go down further
                    break;
                }

                if (!contains_left && !contains_right) {
                    break;
                }

                i_node = contains_left ? node.child_left : node.child_right;
                ++depth;
            }

            int color = std::min(20, depth) / 30.0 * 255;
            canvas(y, x, 0) = color;
            canvas(y, x, 1) = color;
            canvas(y, x, 2) = color;
        }
    }
}

void checkLBVHInvariants(const std::vector<Node> &nodes, int N) {
    // проверим количество нод в дереве
    if (nodes.size() != N - 1 + N /*N+1 inner nodes + N leaves*/) {
        throw std::runtime_error("4923942304203423: " + std::to_string(nodes.size()) + " vs " + std::to_string(N - 1));
    }

    // у каждой ноды либо нет ни одного ребенка, тогда она лист, либо есть два ребенка
    for (const Node &node : nodes) {
        if (node.hasLeftChild() ^ node.hasRightChild()) {
            throw std::runtime_error("9873208597205982");
        }
    }

    // каждая нода достижима из корня, причем только один раз (только один родитель)
    std::vector<int> used(nodes.size());
    std::vector<int> stack;
    stack.push_back(0);
    int total_visited = 0;
    while (!stack.empty() && total_visited < nodes.size()) {
        int i_node = stack.back();
        stack.pop_back();

        ++used[i_node];
        ++total_visited;

        if (nodes[i_node].hasLeftChild())
            stack.push_back(nodes[i_node].child_left);
        if (nodes[i_node].hasRightChild())
            stack.push_back(nodes[i_node].child_right);
    }
    // стек не пустой -> есть циклы
    if (!stack.empty()) {
        throw std::runtime_error("94959345934534");
    }
    // все ноды достигли по разу
    for (int v : used) {
        if (v != 1) {
            throw std::runtime_error("432543534645654");
        }
    }
}

void nbody(bool interactive, bool evaluate_precision, int nbody_impl_index) {
    //    State initial_state = makeRandomState(N, -200, 200, -200, 200);
    //    State initial_state = makeCircularState();

    #if NBODY_INITIAL_STATE_COMPLEXITY == 2
    // конфигурация из гифки
    State initial_state = makeGalacticState(100000, 10, 300);
    const int canvas_size = 1500;
    #elif NBODY_INITIAL_STATE_COMPLEXITY == 1
    // конфигурация полегче
    State initial_state = makeGalacticState(10000, 10, 100);
    const int canvas_size = 500;
    #else
    // вообще легкая конфигурация жесть
    State initial_state = makeGalacticState(1000, 5, 40);
    const int canvas_size = 150;
    #endif

    const int N = initial_state.pxs.size();


    images::Image<unsigned char> canvas(canvas_size, canvas_size, 3);
    unsigned char zero[3] = {};
    canvas.fill(zero);
    std::vector<Color> colors = {RED, GREEN, BLUE, CYAN, MAGENTA, YELLOW, WHITE};

    //    initial_state.coord_shift = canvas_size / 2;
    initial_state.coord_shift = 1 << (NBITS_PER_DIM - 1);
    for (int i = 0; i < N; ++i) {
        initial_state.pxs[i] += initial_state.coord_shift;
        initial_state.pys[i] += initial_state.coord_shift;
    }

    int NT = interactive ? 1500 : 100;
    DeltaState delta_state;
    State state = initial_state;

    std::shared_ptr<images::ImageWindow> window;
    if (ENABLE_GUI) {
        window = std::make_shared<images::ImageWindow>("nbody");
    }

    using nbody_implementation_t = std::function<void(DeltaState &, State &, int, int, const interactive_callback_t *)>;
    nbody_implementation_t nbody_implementation;
    std::string nbody_impl_name;
    switch (nbody_impl_index) {
        case 0: {
            nbody_implementation = nbody_cpu;
            nbody_impl_name = "nbody_cpu";
            break;
        }
        case 1: {
            nbody_implementation = nbody_gpu;
            nbody_impl_name = "nbody_gpu";
            break;
        }
        case 2: {
            nbody_implementation = nbody_cpu_lbvh;
            nbody_impl_name = "nbody_cpu_lbvh";
            break;
        }
        case 3: {
            nbody_implementation = nbody_gpu_lbvh;
            nbody_impl_name = "nbody_gpu_lbvh";
            break;
        }
        default:
            throw std::runtime_error("95438537459734");
    }

    int i_frame = 0;
    timer tm_framerate;
    interactive_callback_t interactive_callback = [&](const std::vector<float> &pxs, const std::vector<float> &pys,
                                                      const std::vector<Node> &nodes) -> void {
        canvas.fill(zero);

        if (nodes.size()) {
            drawLBVH(canvas, nodes, initial_state.coord_shift - canvas_size / 2);
        }

        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            float x = state.pxs[i] + canvas.width / 2.0f - initial_state.coord_shift;
            float y = state.pys[i] + canvas.height / 2.0f - initial_state.coord_shift;

            if (x < 0 || x >= canvas.width || y < 0 || y >= canvas.height) {
                continue;
            }

            Color c = colors[i % colors.size()];

            #pragma omp critical
            {

                canvas(y, x, 0) = std::min<int>(canvas(y, x, 0) + c.r * 0.2, 255);
                canvas(y, x, 1) = std::min<int>(canvas(y, x, 1) + c.g * 0.2, 255);
                canvas(y, x, 2) = std::min<int>(canvas(y, x, 2) + c.b * 0.2, 255);
            };
        }

        int step = 100;
        if (interactive && ++i_frame % step == 0) {
            std::cout << "simulated " << i_frame << " frames, N: " << N << ", framerate: " << (
                step / tm_framerate.elapsed()) << " fps, method: " << nbody_impl_name << std::endl;
            tm_framerate.restart();
        }

        if (SAVE_IMAGES) {
            std::stringstream ss;
            ss << "export_images/frame" << std::setfill('0') << std::setw(6) << i_frame++ << ".jpg";
            std::string path = ss.str();
            canvas.saveJPEG(path, 90);
        }

        if (ENABLE_GUI) {
            window->display(canvas);
            window->resize(1000, 1000);
            window->wait(5);
        }
    };

    timer tm;
    tm.start();
    tm_framerate.start();

    nbody_implementation(delta_state, state, N, NT, interactive ? &interactive_callback : nullptr);

    std::cout << "simulated " << NT << " frames, N: " << N << ", framerate: " << (NT / tm_framerate.elapsed()) <<
            " fps, method: " << nbody_impl_name << std::endl;

    if (interactive) {
        return;
    }

    if (evaluate_precision) {
        std::cout << "evaluating precision.." << std::endl;
        DeltaState delta_state_tmp;
        State state_tmp = initial_state;
        int NT_test = std::min(20, NT);
        nbody_cpu(delta_state_tmp, state_tmp, N, NT_test, nullptr);
        for (int t = 0; t < NT_test; ++t) {
            float *dvx = &delta_state.dvx2d[t * N];
            float *dvy = &delta_state.dvy2d[t * N];

            float *dvx_tmp = &delta_state_tmp.dvx2d[t * N];
            float *dvy_tmp = &delta_state_tmp.dvy2d[t * N];

            int n_good = 0;
            for (int i = 0; i < N; ++i) {
                double err = 0.1 * std::abs(dvx_tmp[i]);
                if (std::abs(dvx[i] - dvx_tmp[i]) < err)
                    n_good++;
            }
            EXPECT_GE(n_good, 0.9 * N);
        }
    } else {
        std::cout << "skipped precision evaluation" << std::endl;
    }

    state = initial_state;
    for (int t = 0; t < NT; ++t) {
        float *dvx = &delta_state.dvx2d[t * N];
        float *dvy = &delta_state.dvy2d[t * N];

        for (int i = 0; i < N; ++i) {
            state.vxs[i] += dvx[i];
            state.vys[i] += dvy[i];
            state.pxs[i] += state.vxs[i];
            state.pys[i] += state.vys[i];
        }

        interactive_callback(state.pxs, state.pys, {});
    }
}

void checkTreesEqual(const std::vector<Node> &nodes_recursive, const std::vector<Node> &nodes,
                     const Node &root_recursive, const Node &root) {
    EXPECT_EQ(root_recursive.bbox, root.bbox);
    EXPECT_NEAR(root_recursive.mass, root.mass, 0.0001);
    EXPECT_NEAR(root_recursive.cmsx, root.cmsx, 0.0001);
    EXPECT_NEAR(root_recursive.cmsy, root.cmsy, 0.0001);
    EXPECT_EQ(root_recursive.hasLeftChild(), root.hasLeftChild());
    EXPECT_EQ(root_recursive.hasRightChild(), root.hasRightChild());

    if (root_recursive.hasLeftChild() && root.hasLeftChild()) {
        checkTreesEqual(nodes_recursive, nodes, nodes_recursive[root_recursive.child_left], nodes[root.child_left]);
    }
    if (root_recursive.hasRightChild() && root.hasRightChild()) {
        checkTreesEqual(nodes_recursive, nodes, nodes_recursive[root_recursive.child_right], nodes[root.child_right]);
    }
}

TEST(LBVH, CPU) {
    if (!ENABLE_TESTING)
        return;

    std::srand(1);

    images::Image<unsigned char> canvas(500, 500, 3);
    std::vector<Color> colors = {RED, GREEN, BLUE, CYAN, MAGENTA, YELLOW, WHITE};

    std::shared_ptr<images::ImageWindow> window;
    if (ENABLE_GUI) {
        window = std::make_shared<images::ImageWindow>("lbvh_naive");
    }

    auto interactive_callback = [&]() -> void {
        unsigned char zero[3] = {};
        canvas.fill(zero);

        int N = 10000;
        std::vector<Point> points;
        std::vector<morton_t> codes;
        points.reserve(N);
        codes.reserve(N);
        for (int i = 0; i < N; ++i) {

            // circle sampling
            //            float angle = 3.14159 * 2 / N * i;
            //            int x = canvas.width * 0.5 * (1.0 + 0.9 * std::sin(angle));
            //            int y = canvas.height * 0.5 * (1.0 + 0.9 * std::cos(angle));
            //            points.push_back(Point{x, y});

            // random sampling
            points.push_back(Point{int(std::rand() % canvas.width), int(std::rand() % canvas.height)});

            codes.emplace_back(zOrder(points.back(), i));
        }

        std::sort(codes.begin(), codes.end());

        // удобно для дебага, можно распечатывать мортоновские коды в столбики
        //        printMortonCodes(codes);

        // check unique
        for (int i = 1; i < N; ++i) {
            EXPECT_NE(codes[i-1], codes[i]);
        }

        std::vector<Node> nodes;

        EXPECT_NO_THROW(buildLBVH(nodes, codes, points));
        EXPECT_NO_THROW(checkLBVHInvariants(nodes, N));

        std::vector<int> buffer;
        EXPECT_NO_THROW(buildBBoxes(nodes, buffer, N));

        {
            std::vector<Node> nodes_recursive;
            buildLBVHRecursive(nodes_recursive, codes, points, 0, N, NBITS - 1);
            buildBBoxesRecursive(nodes_recursive, nodes_recursive.front());
            EXPECT_NO_THROW(checkTreesEqual(nodes_recursive, nodes, nodes_recursive.front(), nodes.front()));
        }


        if (ENABLE_GUI) {
            drawLBVH(canvas, nodes);

            // draw z-curve
            std::vector<Point> buffer;
            for (int i = 1; i < (int) points.size(); ++i) {
                buffer.clear();
                bresenham(buffer, points[getIndex(codes[i - 1])], points[getIndex(codes[i])]);
                for (const auto &[x, y] : buffer) {
                    canvas(y, x, 0) = 255;
                    canvas(y, x, 1) = 255;
                    canvas(y, x, 2) = 255;
                }
            }

            window->display(canvas);
            window->resize(1000, 1000);
            window->wait(500);
        }
    };

    for (int i = 0; i < 10; ++i) {
        interactive_callback();
    }
}

TEST(LBVH, GPU) {
    if (!ENABLE_TESTING)
        return;

    gpu::Device device = gpu::chooseGPUDevice(OPENCL_DEVICE_INDEX);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    std::srand(1);

    int N = 100000;
    std::vector<float> pxs(N);
    std::vector<float> pys(N);
    std::vector<float> mxs(N);
    std::vector<morton_t> codes(N);
    for (int i = 0; i < N; ++i) {
        pxs[i] = std::rand() % (1 << NBITS_PER_DIM);
        pys[i] = std::rand() % (1 << NBITS_PER_DIM);
        mxs[i] = 100;
    }

    const points_mass_functor points_mass_array = [&](int i) { return std::make_tuple(pxs[i], pys[i], mxs[i]); };

    unsigned int workGroupSize = 128;
    unsigned int global_work_size_points = (N + workGroupSize - 1) / workGroupSize * workGroupSize;
    unsigned int global_work_size_nodes = (LBVHSize(N) + workGroupSize - 1) / workGroupSize * workGroupSize;
    ocl::Kernel kernel_generate_morton_codes(lbvh_kernel, lbvh_kernel_length, "generateMortonCodes");
    kernel_generate_morton_codes.compile();
    gpu::gpu_mem_32f pxs_gpu, pys_gpu, mxs_gpu;
    gpu::shared_device_buffer_typed<morton_t> codes_gpu;

    pxs_gpu.resizeN(N);
    pys_gpu.resizeN(N);
    mxs_gpu.resizeN(N);
    codes_gpu.resizeN(N);

    pxs_gpu.writeN(pxs.data(), N);
    pys_gpu.writeN(pys.data(), N);
    mxs_gpu.writeN(mxs.data(), N);


    // GENERATE MORTON CODES

    kernel_generate_morton_codes.exec(gpu::WorkSize(workGroupSize, global_work_size_points),
                                      pxs_gpu, pys_gpu,
                                      codes_gpu,
                                      N);

    codes_gpu.readN(codes.data(), N);

    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(codes[i], zOrder(makePoint(pxs[i], pys[i]), i));
    }


    // SORT MORTON CODES

    ocl::Kernel kernel_merge(lbvh_kernel, lbvh_kernel_length, "merge");
    kernel_merge.compile();

    gpu::gpu_mem_64f codes_gpu_buf;
    codes_gpu_buf.resizeN(N);
    for (unsigned int subn = 1; subn < N; subn *= 2) {
        kernel_merge.exec(gpu::WorkSize(workGroupSize, global_work_size_points), codes_gpu, codes_gpu_buf, N, subn);
        codes_gpu.swap(codes_gpu_buf);
    }
    std::vector<morton_t> codes_tmp = codes;
    codes_gpu.readN(codes.data(), N);
    {
        std::sort(codes_tmp.begin(), codes_tmp.end());
        for (int i = 1; i < N; ++i) {
            EXPECT_LE(codes_tmp[i-1], codes_tmp[i]);
        }
        for (int i = 1; i < N; ++i) {
            EXPECT_LE(codes[i-1], codes[i]);
        }
        for (int i = 0; i < N; ++i) {
            EXPECT_EQ(codes_tmp[i], codes[i]);
        }
    }


    // BUILD LBVH

    const int tree_size = LBVHSize(N);

    std::vector<Node> nodes(tree_size);
    // init with something just for test
    for (int i = 0; i < tree_size; ++i) {
        Node &n = nodes[i];
        n.mass = std::rand();
        n.cmsx = std::rand();
        n.cmsy = std::rand();
        n.child_right = std::rand();
        n.child_left = std::rand();
        n.bbox.grow(makePoint(std::rand(), std::rand()));
    }
    std::vector<Node> nodes_cpu = nodes;

    gpu::gpu_mem_any nodes_gpu;
    nodes_gpu.resize(tree_size * sizeof(Node));

    {
        nodes_gpu.write(nodes.data(), tree_size * sizeof(Node));
        std::vector<Node> tmp(tree_size);
        nodes_gpu.read(tmp.data(), tree_size * sizeof(Node));

        for (int i = 0; i < tree_size; ++i) {
            EXPECT_EQ(tmp[i], nodes[i]);
        }
    }

    ocl::Kernel kernel_build_lbvh(lbvh_kernel, lbvh_kernel_length, "buidLBVH");
    kernel_build_lbvh.compile();

    kernel_build_lbvh.exec(gpu::WorkSize(workGroupSize, global_work_size_nodes),
                           pxs_gpu, pys_gpu, mxs_gpu,
                           codes_gpu, nodes_gpu,
                           N);


    nodes_gpu.read(nodes.data(), tree_size * sizeof(Node));

    for (int i = 0; i < tree_size; ++i) {
        initLBVHNode(nodes_cpu, i, codes, points_mass_array);
    }

    for (int i = 0; i < tree_size; ++i) {
        EXPECT_EQ(nodes[i], nodes_cpu[i]);
    }


    // BUILD BBOXES AND AGGREGATE MASS INFO

    // аналог buildBBoxes
    {
        gpu::gpu_mem_32i flags_gpu;
        flags_gpu.resizeN(N);

        ocl::Kernel kernel_init_flags(lbvh_kernel, lbvh_kernel_length, "initFlags");
        ocl::Kernel kernel_grow_nodes(lbvh_kernel, lbvh_kernel_length, "growNodes");

        kernel_init_flags.compile();
        kernel_grow_nodes.compile();

        for (int level = 0; level < NBITS; ++level) {

            kernel_init_flags.exec(gpu::WorkSize(workGroupSize, global_work_size_points),
                                   flags_gpu, nodes_gpu,
                                   N, level);

            kernel_grow_nodes.exec(gpu::WorkSize(workGroupSize, global_work_size_points),
                                   flags_gpu, nodes_gpu,
                                   N, level);

            int n_updated;
            flags_gpu.readN(&n_updated, 1, N - 1);

            //            std::cout << "n updated: " << n_updated << std::endl;

            if (!n_updated)
                break;
        }

        nodes_gpu.read(nodes.data(), tree_size * sizeof(Node));

        std::vector<int> flags;
        buildBBoxes(nodes_cpu, flags, N);

        for (int i = 0; i < tree_size; ++i) {
            EXPECT_EQ(nodes[i], nodes_cpu[i]);
        }
    }

    std::vector<float> vxs(N);
    std::vector<float> vys(N);
    std::vector<float> dvx(N);
    std::vector<float> dvy(N);

    gpu::gpu_mem_32f vxs_gpu, vys_gpu;
    gpu::gpu_mem_32f dvx_gpu, dvy_gpu;

    vxs_gpu.resizeN(N);
    vys_gpu.resizeN(N);
    dvx_gpu.resizeN(N);
    dvy_gpu.resizeN(N);

    vxs_gpu.writeN(vxs.data(), N);
    vys_gpu.writeN(vys.data(), N);
    dvx_gpu.writeN(dvx.data(), N);
    dvy_gpu.writeN(dvy.data(), N);

    std::vector<float> pxs_cpu = pxs;
    std::vector<float> pys_cpu = pys;
    std::vector<float> vxs_cpu = vxs;
    std::vector<float> vys_cpu = vys;
    std::vector<float> dvx_cpu = dvx;
    std::vector<float> dvy_cpu = dvy;

    {
        ocl::Kernel kernel_calculate_forces(lbvh_kernel, lbvh_kernel_length, "calculateForces");
        ocl::Kernel kernel_integrate(lbvh_kernel, lbvh_kernel_length, "integrate");

        kernel_calculate_forces.compile();
        kernel_integrate.compile();

        int t = 0;
        int coord_shift = 0;

        kernel_calculate_forces.exec(gpu::WorkSize(workGroupSize, global_work_size_points),
                                     pxs_gpu, pys_gpu, vxs_gpu, vys_gpu,
                                     mxs_gpu, nodes_gpu,
                                     dvx_gpu, dvy_gpu,
                                     N, t);

        kernel_integrate.exec(gpu::WorkSize(workGroupSize, global_work_size_points),
                              pxs_gpu, pys_gpu, vxs_gpu, vys_gpu,
                              mxs_gpu,
                              dvx_gpu, dvy_gpu,
                              N, t, coord_shift);

        pxs_gpu.readN(pxs.data(), N);
        pys_gpu.readN(pys.data(), N);
        vxs_gpu.readN(vxs.data(), N);
        vys_gpu.readN(vys.data(), N);
        dvx_gpu.readN(dvx.data(), N);
        dvy_gpu.readN(dvy.data(), N);
    }

    {
        for (int i = 0; i < N; ++i) {
            float x0 = pxs_cpu[i];
            float y0 = pys_cpu[i];
            float m0 = mxs[i];
            calculateForce(x0, y0, m0, nodes_cpu, &dvx_cpu[i], &dvy_cpu[i]);
        }

        int n_super_good_pxs = 0;
        int n_super_good_pys = 0;
        int n_super_good_vxs = 0;
        int n_super_good_vys = 0;
        int n_super_good_dvx = 0;
        int n_super_good_dvy = 0;
        for (int i = 0; i < N; ++i) {
            integrate(i, pxs_cpu, pys_cpu, vxs_cpu, vys_cpu, dvx_cpu.data(), dvy_cpu.data(), 0);

            double rel_eps_super_good = 1e-3;
            if (std::abs(pxs[i] - pxs_cpu[i]) < rel_eps_super_good * std::abs(pxs_cpu[i]))
                n_super_good_pxs++;
            if (std::abs(pys[i] - pys_cpu[i]) < rel_eps_super_good * std::abs(pys_cpu[i]))
                n_super_good_pys++;
            if (std::abs(vxs[i] - vxs_cpu[i]) < rel_eps_super_good * std::abs(vxs_cpu[i]))
                n_super_good_vxs++;
            if (std::abs(vys[i] - vys_cpu[i]) < rel_eps_super_good * std::abs(vys_cpu[i]))
                n_super_good_vys++;
            if (std::abs(dvx[i] - dvx_cpu[i]) < rel_eps_super_good * std::abs(dvx_cpu[i]))
                n_super_good_dvx++;
            if (std::abs(dvy[i] - dvy_cpu[i]) < rel_eps_super_good * std::abs(dvy_cpu[i]))
                n_super_good_dvy++;

            double rel_eps = 0.5;
            EXPECT_NEAR(pxs[i], pxs_cpu[i], rel_eps * std::abs(pxs_cpu[i]));
            EXPECT_NEAR(pys[i], pys_cpu[i], rel_eps * std::abs(pys_cpu[i]));
            EXPECT_NEAR(vxs[i], vxs_cpu[i], rel_eps * std::abs(vxs_cpu[i]));
            EXPECT_NEAR(vys[i], vys_cpu[i], rel_eps * std::abs(vys_cpu[i]));
            EXPECT_NEAR(dvx[i], dvx_cpu[i], rel_eps * std::abs(dvx_cpu[i]));
            EXPECT_NEAR(dvy[i], dvy_cpu[i], rel_eps * std::abs(dvy_cpu[i]));
        }

        EXPECT_GE(n_super_good_pxs, 0.99 * N);
        EXPECT_GE(n_super_good_pys, 0.99 * N);
        EXPECT_GE(n_super_good_vxs, 0.99 * N);
        EXPECT_GE(n_super_good_vys, 0.99 * N);
        EXPECT_GE(n_super_good_dvx, 0.99 * N);
        EXPECT_GE(n_super_good_dvy, 0.99 * N);
    }
}

TEST(LBVH, Nbody) {
    if (!ENABLE_TESTING)
        return;

    gpu::Device device = gpu::chooseGPUDevice(OPENCL_DEVICE_INDEX);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    bool evaluate_precision = (NBODY_INITIAL_STATE_COMPLEXITY < 2) && EVALUATE_PRECISION;

    #if NBODY_INITIAL_STATE_COMPLEXITY < 2
    nbody(false, evaluate_precision, 0);// cpu naive
    nbody(false, evaluate_precision, 1);// gpu naive
    #endif
    nbody(false, evaluate_precision, 2);// cpu lbvh
    nbody(false, evaluate_precision, 3); // gpu lbvh
}

TEST(LBVH, Nbody_meditation) {
    if (!ENABLE_TESTING)
        return;

    if (!ENABLE_GUI)
        return;

    gpu::Device device = gpu::chooseGPUDevice(OPENCL_DEVICE_INDEX);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    nbody(true, false, 3);// gpu lbvh
}
