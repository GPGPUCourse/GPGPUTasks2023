#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6


#define GRAVITATIONAL_FORCE 0.0001

#define morton_t ulong

#define NBITS_PER_DIM 16
#define NBITS (NBITS_PER_DIM /*x dimension*/ + NBITS_PER_DIM /*y dimension*/ + 32 /*index augmentation*/)

int LBVHSize(int N) {
    return N + N-1;
}

morton_t getBits(morton_t morton_code, int bit_index, int prefix_size)
{
    morton_t one = 1;
    return (morton_code >> bit_index) & ((one << prefix_size) - one);
}

int getBit(morton_t morton_code, int bit_index)
{
    return (morton_code >> bit_index) & 1;
}

int getIndex(morton_t morton_code)
{
    morton_t mask = 1;
    mask = (mask << 32) - 1;
    return morton_code & mask;
}

int spreadBits(int word){
    word = (word ^ (word << 8)) & 0x00ff00ff;
    word = (word ^ (word << 4)) & 0x0f0f0f0f;
    word = (word ^ (word << 2)) & 0x33333333;
    word = (word ^ (word << 1)) & 0x55555555;
    return word;
}

struct __attribute__ ((packed)) BBox {

    int minx, maxx;
    int miny, maxy;

};

void clear(__global struct BBox *self)
{
    self->minx = INT_MAX;
    self->maxx = INT_MIN;
    self->miny = self->minx;
    self->maxy = self->maxx;
}

bool contains(__global const struct BBox *self, float fx, float fy)
{
    int x = fx + 0.5;
    int y = fy + 0.5;
    return x >= self->minx && x <= self->maxx &&
           y >= self->miny && y <= self->maxy;
}

bool empty(__global const struct BBox *self)
{
    return self->minx > self->maxx;
}

struct __attribute__ ((packed)) Node {

    int child_left, child_right;
    struct BBox bbox;

    // used only for nbody
    float mass;
    float cmsx;
    float cmsy;
};

bool hasLeftChild(__global const struct Node *self)
{
    return self->child_left >= 0;
}

bool hasRightChild(__global const struct Node *self)
{
    return self->child_right >= 0;
}

bool isLeaf(__global const struct Node *self)
{
    return !hasLeftChild(self) && !hasRightChild(self);
}

void growPoint(__global struct BBox *self, float fx, float fy)
{
    self->minx = min(self->minx, (int) (fx + 0.5));
    self->maxx = max(self->maxx, (int) (fx + 0.5));
    self->miny = min(self->miny, (int) (fy + 0.5));
    self->maxy = max(self->maxy, (int) (fy + 0.5));
}

void growBBox(__global struct BBox *self, __global const struct BBox *other)
{
    growPoint(self, other->minx, other->miny);
    growPoint(self, other->maxx, other->maxy);
}

bool equals(__global const struct BBox *lhs, __global const struct BBox *rhs)
{
    return lhs->minx == rhs->minx && lhs->maxx == rhs->maxx && lhs->miny == rhs->miny && lhs->maxy == rhs->maxy;
}

bool equalsPoint(__global const struct BBox *lhs, float fx, float fy)
{
    int x = fx + 0.5;
    int y = fy + 0.5;
    return lhs->minx == x && lhs->maxx == x && lhs->miny == y && lhs->maxy == y;
}

morton_t zOrder(float fx, float fy, int i){
    int x = fx + 0.5;
    int y = fy + 0.5;

    // у нас нет эксепшенов, но можно писать коды ошибок просто в консоль, и следить чтобы вывод был пустой

    if (x < 0 || x >= (1 << NBITS_PER_DIM)) {
        printf("098245490432590890\n");
        return 0;
    }
    if (y < 0 || y >= (1 << NBITS_PER_DIM)) {
        printf("432764328764237823\n");
        return 0;
    }

    morton_t morton_code = ((1ll * spreadBits(x)) << 1) + spreadBits(y);
    return (morton_code << 32) | i;
}

__kernel void generateMortonCodes(__global const float *pxs, __global const float *pys,
                                  __global morton_t *codes,
                                  int N)
{
    int gid = get_global_id(0);
    if (gid >= N)
        return;

    codes[gid] = zOrder(pxs[gid], pys[gid], gid);
}

bool mergePathPredicate(morton_t val_mid, morton_t val_cur, bool is_right)
{
    return is_right ? val_mid <= val_cur : val_mid < val_cur;
}

void __kernel merge(__global const morton_t *as, __global morton_t *as_sorted, unsigned int n, unsigned int subarray_size)
{
    const int gid = get_global_id(0);
    if (gid >= n)
        return;

    const int subarray_id = gid / subarray_size;
    const int is_right_subarray = subarray_id & 1;

    const int base_cur = (subarray_id) * subarray_size;
    const int base_other = (subarray_id + 1 - 2 * is_right_subarray) * subarray_size;

    const int j = gid - base_cur;
    const morton_t val_cur = as[gid];

    int i0 = -1;
    int i1 = subarray_size;
    while (i1 - i0 > 1) {
        int mid = (i0 + i1) / 2;
        if (base_other + mid < n && mergePathPredicate(as[base_other + mid], val_cur, is_right_subarray)) {
            i0 = mid;
        } else {
            i1 = mid;
        }
    }
    const int i = i1;

    int idx = min(base_cur, base_other) + j + i;
    as_sorted[idx] = val_cur;
}

__kernel void initLBVHNode(__global struct Node *nodes, __global const morton_t *codes,
                           __global const float *pxs, __global const float *pys, __global const float *mxs,
                           const int N)
{
    const int i_node = get_global_id(0);
    if (i_node >= 2*N-1) {
        return;
    }

    clear(&nodes[i_node].bbox);
    nodes[i_node].mass = 0;
    nodes[i_node].cmsx = 0;
    nodes[i_node].cmsy = 0;

    if (i_node >= N-1) {
        nodes[i_node].child_left = -1;
        nodes[i_node].child_right = -1;
        int i_point = i_node - (N-1);

        i_point = getIndex(codes[i_point]);

        nodes[i_node].bbox.minx = nodes[i_node].bbox.maxx = pxs[i_point] + 0.5;
        nodes[i_node].bbox.miny = nodes[i_node].bbox.maxy = pys[i_point] + 0.5;
        nodes[i_node].cmsx = pxs[i_point];
        nodes[i_node].cmsy = pys[i_point];
        nodes[i_node].mass = mxs[i_point];

        return;
    }

    int i_begin = 0, i_end = N - 1, bit_index = NBITS-1;
    // если рассматриваем не корень, то нужно найти зону ответственности ноды и самый старший бит, с которого надо начинать поиск разреза
    if (i_node) {
        morton_t xor_left = codes[i_node - 1] ^ codes[i_node];
        morton_t xor_right = codes[i_node + 1] ^ codes[i_node];
        for (int i = NBITS - 1; i >= 0; i--) {
            if (xor_left & (1ll << i)) {
                int l = i_node, r = N;
                while (l + 1 < r) {
                    int m = (l + r) / 2;
                    if ((codes[i_node] >> i) == (codes[m] >> i)) l = m;
                    else r = m;
                }
                i_begin = i_node, i_end = l, bit_index = i - 1;
                break;
            }
            if (xor_right & (1ll << i)) {
                int l = -1, r = i_node;
                while (l + 1 < r) {
                    int m = (l + r) / 2;
                    if ((codes[i_node] >> i) == (codes[m] >> i)) r = m;
                    else l = m;
                }
                i_begin = r, i_end = i_node, bit_index = i - 1;
                break;
            }
        }
    }

    for (int i_bit = bit_index; i_bit >= 0; --i_bit) {
        if ((codes[i_begin] & (1ll << i_bit)) != (codes[i_end] & (1ll << i_bit))) {
            int l = i_begin, r = i_end;
            while (l + 1 < r) {
                int m = (l + r) / 2;
                if ((codes[m] & (1ll << i_bit)) == 0) l = m;
                else r = m;
            }
            nodes[i_node].child_left = l, nodes[i_node].child_right = r;
            if (i_begin == l) nodes[i_node].child_left += N - 1;
            if (r == i_end) nodes[i_node].child_right += N - 1;
            return;
        }
    }

    printf("54356549645\n");
}

void initFlag(__global int *flags, int i_node, __global const struct Node *nodes, int level)
{
    flags[i_node] = -1;

    __global const struct Node *node = &nodes[i_node];
    if (isLeaf(node)) {
        printf("9423584385834\n");
        return;
    }

    if (!empty(&node->bbox)) {
        return;
    }

    __global const struct BBox *left = &nodes[node->child_left].bbox;
    __global const struct BBox *right = &nodes[node->child_right].bbox;

    if (!empty(left) && !empty(right)) {
        flags[i_node] = level;
    }
}

__kernel void initFlags(__global int *flags, __global const struct Node *nodes,
                       int N, int level)
{
    int gid = get_global_id(0);

    if (gid == N-1)
        flags[gid] = 0; // use last element as a n_updated counter in next kernel

    if (gid >= N-1) // инициализируем только внутренние ноды
        return;

    initFlag(flags, gid, nodes, level);
}

void growNode(__global struct Node *root, __global struct Node *nodes)
{
    __global const struct Node *left = &nodes[root->child_left];
    __global const struct Node *right = &nodes[root->child_right];

    growBBox(&root->bbox, &left->bbox);
    growBBox(&root->bbox, &right->bbox);

    double m0 = left->mass;
    double m1 = right->mass;

    root->mass = m0 + m1;

    if (root->mass <= 1e-8) {
        printf("04230420340322\n");
//        return;
    }

    root->cmsx = (left->cmsx * m0 + right->cmsx * m1) / root->mass;
    root->cmsy = (left->cmsy * m0 + right->cmsy * m1) / root->mass;
}

__kernel void growNodes(__global int *flags, __global struct Node *nodes,
                        int N, int level)
{
    int gid = get_global_id(0);

    if (gid >= N-1) // инициализируем только внутренние ноды
        return;

    __global struct Node *node = &nodes[gid];
    if (flags[gid] == level) {
        growNode(node, nodes);
        atomic_add(&flags[N-1], 1);
    }
}

// https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation
bool barnesHutCondition(float x, float y, __global const struct Node *node)
{
    float dx = x - node->cmsx;
    float dy = y - node->cmsy;
    float s = max(node->bbox.maxx - node->bbox.minx, node->bbox.maxy - node->bbox.miny);
    float d2 = dx*dx + dy*dy;
    float thresh = 0.5;

    return s * s < d2 * thresh * thresh;
}

bool int3Eq(int x, int y, int z)
{
    return (x == y) && (y == z);
}

void calculateForce(float x0, float y0, float m0, __global const struct Node *nodes, __global float *force_x, __global float *force_y)
{
    int stack[2 * NBITS_PER_DIM];
    int stack_size = 0;
    stack[stack_size++] = 0;
    while (stack_size) {
        const struct Node node = nodes[stack[--stack_size]];

        if (isLeaf(&nodes[stack[stack_size]])) {
            continue;
        }

        // если запрос содержится и а левом и в правом ребенке - то они в одном пикселе
        {
            if (contains(&nodes[node.child_left].bbox, x0, y0) && contains(&nodes[node.child_right].bbox, x0, y0)) {
                if (!equals(&nodes[node.child_left].bbox, &nodes[node.child_right].bbox)) {
                    printf("42357987645432456547\n");
                    return;
                }
                if (!(int3Eq(nodes[node.child_left].bbox.minx, nodes[node.child_left].bbox.maxx, x0 + 0.5) &&
                      int3Eq(nodes[node.child_left].bbox.miny, nodes[node.child_left].bbox.maxy, y0 + 0.5))) {
                    printf("5446456456435656\n");
                    return;
                }
                continue;
            }
        }

        {
            int i_child = node.child_left;
            if (!contains(&nodes[i_child].bbox, x0, y0) && barnesHutCondition(x0, y0, &nodes[i_child])) {
                float x1 = nodes[i_child].cmsx;
                float y1 = nodes[i_child].cmsy;
                float m1 = nodes[i_child].mass;

                float dx = x1 - x0;
                float dy = y1 - y0;
                float dr2 = max(100.f, dx * dx + dy * dy);

                float dr2_inv = 1.f / dr2;
                float dr_inv = sqrt(dr2_inv);

                float ex = dx * dr_inv;
                float ey = dy * dr_inv;

                float fx = ex * dr2_inv * GRAVITATIONAL_FORCE;
                float fy = ey * dr2_inv * GRAVITATIONAL_FORCE;

                *force_x += m1 * fx;
                *force_y += m1 * fy;

            } else {
                stack[stack_size++] = i_child;
                if (stack_size >= 2 * NBITS_PER_DIM) {
                    printf("0420392384283\n");
                    return;
                }
            }
        }

        {
            int i_child = node.child_right;
            if (!contains(&nodes[i_child].bbox, x0, y0) && barnesHutCondition(x0, y0, &nodes[i_child])) {
                float x1 = nodes[i_child].cmsx;
                float y1 = nodes[i_child].cmsy;
                float m1 = nodes[i_child].mass;

                float dx = x1 - x0;
                float dy = y1 - y0;
                float dr2 = max(100.f, dx * dx + dy * dy);

                float dr2_inv = 1.f / dr2;
                float dr_inv = sqrt(dr2_inv);

                float ex = dx * dr_inv;
                float ey = dy * dr_inv;

                float fx = ex * dr2_inv * GRAVITATIONAL_FORCE;
                float fy = ey * dr2_inv * GRAVITATIONAL_FORCE;

                *force_x += m1 * fx;
                *force_y += m1 * fy;

            } else {
                stack[stack_size++] = i_child;
                if (stack_size >= 2 * NBITS_PER_DIM) {
                    printf("0420392384283\n");
                    return;
                }
            }
        }
    }
}

__kernel void calculateForces(
        __global const float *pxs, __global const float *pys,
        __global const float *vxs, __global const float *vys,
        __global const float *mxs,
        __global const struct Node *nodes,
        __global float * dvx2d, __global float * dvy2d,
        const unsigned int N,
        int t)
{
    const unsigned int i_node = get_global_id(0);

    if (i_node > N)
        return;

    dvx2d[i_node], dvy2d[i_node] = 0;
    calculateForce(pxs[i_node], pys[i_node], mxs[i_node], nodes, &dvx2d[i_node], &dvy2d[i_node]);
}

__kernel void integrate(
        __global float * pxs, __global float * pys,
        __global float *vxs, __global float *vys,
        __global const float *mxs,
        __global float * dvx2d, __global float * dvy2d,
        int N,
        int t,
        int coord_shift)
{
    unsigned int i = get_global_id(0);

    if (i >= N)
        return;

    __global float * dvx = dvx2d + t * N;
    __global float * dvy = dvy2d + t * N;

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
