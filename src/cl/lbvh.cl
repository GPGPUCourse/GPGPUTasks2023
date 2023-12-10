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
    word = (word ^ (word << 8 )) & 0x00ff00ff;
    word = (word ^ (word << 4 )) & 0x0f0f0f0f;
    word = (word ^ (word << 2 )) & 0x33333333;
    word = (word ^ (word << 1 )) & 0x55555555;
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
        //return 0;
    }
    if (y < 0 || y >= (1 << NBITS_PER_DIM)) {
        printf("432764328764237823\n");
        //return 0;
    }
    morton_t morton_code = (spreadBits(y) << 1) | spreadBits(x);

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

int findSplit(__global const morton_t *codes, int i_begin, int i_end, int bit_index)
{
    if (getBit(codes[i_begin], bit_index) == getBit(codes[i_end - 1], bit_index)) {
        return -1;
    }

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

void findRegion(int *i_begin, int *i_end, int *bit_index, __global const morton_t *codes, int N, int i_node)
{
    if (i_node < 1 || i_node > N - 2) {
       printf("842384298293482\n");
       return;
    }
    int dir = 0;
    int i_bit = NBITS - 1;
    for (; i_bit >= 0; --i_bit) {
        if (getBit(codes[i_node - 1], i_bit) == 0 && getBit(codes[i_node + 1], i_bit) == 1) {
            dir = getBit(codes[i_node], i_bit) == 0 ? -1 : 1;
            break;
        }
    }

     if (dir == 0) {
        printf("8923482374983\n");
        return;
     }

     int K = NBITS - i_bit;
     morton_t pref0 = getBits(codes[i_node], i_bit, K);


     int start = i_node, end = dir == 1 ? N : -1;
     while (start + dir != end) {
         int mid = start + (end - start) / 2;
         if (getBits(codes[mid], i_bit, K) == pref0) {
             start = mid;
         } else {
             end = mid;
         }
     }
     int i_node_end  = start;
     *bit_index = i_bit - 1;

     if (dir > 0) {
         *i_begin = i_node;
         *i_end = i_node_end + 1;
     } else {
         *i_begin = i_node_end;
         *i_end = i_node + 1;
     }
}


void initLBVHNode(__global struct Node *nodes, int i_node, __global const morton_t *codes, int N, __global const float *pxs, __global const float *pys, __global const float *mxs)
{
    __global struct Node* node = &nodes[i_node];
    clear(&node->bbox);
    node->mass = 0;
    node->cmsx = 0;
    node->cmsy = 0;

    if (i_node >= N - 1) {
        node->child_left = -1;
        node->child_right = -1;
        int i_point = i_node - (N - 1);

        int idx = getIndex(codes[i_point]);
        float center_mass_x = pxs[idx], center_mass_y = pys[idx];
        float mass = mxs[idx];

        growPoint(&node->bbox, center_mass_x, center_mass_y);
        node->cmsx = center_mass_x;
        node->cmsy = center_mass_y;
        node->mass = mass;

        return;
    }

    int i_begin = 0, i_end = N, bit_index = NBITS - 1;
    if (i_node) {
        findRegion(&i_begin, &i_end, &bit_index, codes, N, i_node);
    }

    bool found = false;
    for (int i_bit = bit_index; i_bit >= 0; --i_bit) {
        int split = findSplit(codes, i_begin, i_end, i_bit);
        if (split < 0)
            continue;

        if (split < 1) {
            printf("043204230042342\n");
            return;
        }

        // проинициализировать nodes[i_node].child_left, nodes[i_node].child_right на основе i_begin, i_end, split
        node->child_left = split - i_begin < 2 ? split - 1 + (N - 1) : split - 1;
        node->child_right = i_end - split < 2 ? split + (N - 1) : split;
        //   не забудьте на N-1 сдвинуть индексы, указывающие на листья

        found = true;
        break;
    }

    if (!found) {
        printf("54356549645\n");
    }
}

__kernel void buidLBVH(__global const float *pxs, __global const float *pys, __global const float *mxs,
                       __global const morton_t *codes, __global struct Node *nodes,
                       int N)
{
    int tree_size = LBVHSize(N);
    int i_node = get_global_id(0);
    if (i_node >= tree_size)
    return;

    initLBVHNode(nodes, i_node, codes, N, pxs, pys, mxs);
}

void initFlag(__global int *flags, int i_node, __global const struct Node *nodes, int level)
{
    flags[i_node] = -1;

    __global const struct Node *node = &nodes[i_node];
    if (isLeaf(node)) {
        printf("9423584385834\n");
//        return;
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

void calculateForce(float x0, float y0, float m0, __global const struct Node *nodes, __global float *force_x, __global float *force_y)
{
   // основная идея ускорения - аггрегировать в узлах дерева веса и центры масс,
   //   и не спускаться внутрь, если точка запроса не пересекает ноду, а заменить на взаимодействие с ее центром масс

   int stack[2 * NBITS_PER_DIM];
   int stack_size = 0;
   stack[stack_size++] = 0;
   while (stack_size) {
       int i_node = stack[--stack_size];
       __global const struct Node *node = &nodes[i_node];

       if (isLeaf(node)) {
           continue;
       }

       // если запрос содержится и а левом и в правом ребенке - то они в одном пикселе
       {
           __global const struct Node *left = &nodes[node->child_left];
           __global const struct Node *right = &nodes[node->child_right];
           if (contains(&left->bbox, x0, y0) && contains(&right->bbox, x0, y0)) {
               if (!equals(&left->bbox, &right->bbox)) {
                   printf("42357987645432456547\n");
               }
               if (!equalsPoint(&left->bbox, x0, y0)) {
                  printf("5446456456435656\n");
               }
               continue;
           }
       }
       int idx_children[] = {node->child_left, node->child_right};
       for (int i = 0; i < 2; ++i) {
           int i_child = idx_children[i];
           __global const struct Node *child = &nodes[i_child];
           // С точки зрения ббоксов заходить в ребенка, ббокс которого не пересекаем, не нужно (из-за того, что в листьях у нас точки и они не высовываются за свой регион пространства)
           //   Но, с точки зрения физики, замена гравитационного влияния всех точек в регионе на взаимодействие с суммарной массой в центре масс - это точное решение только в однородном поле (например, на поверхности земли)
           //   У нас поле неоднородное, и такая замена - лишь приближение. Чтобы оно было достаточно точным, будем спускаться внутрь ноды, пока она не станет похожа на точечное тело (маленький размер ее ббокса относительно нашего расстояния до центра масс ноды)
           if (!contains(&child->bbox, x0, y0) && barnesHutCondition(x0, y0, child)) {
               float dx = child->cmsx - x0;
               float dy = child->cmsy - y0;
               float dr2 = max(100.f, dx * dx + dy * dy);

               float dr2_inv = 1.f / dr2;
               float dr_inv = sqrt(dr2_inv);

               float ex = dx * dr_inv;
               float ey = dy * dr_inv;

               float fx = ex * dr2_inv * GRAVITATIONAL_FORCE;
               float fy = ey * dr2_inv * GRAVITATIONAL_FORCE;

               *force_x += child->mass * fx;
               *force_y += child->mass * fy;
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
        int N,
        int t)
{
   int i = get_global_id(0);
   float x0 = pxs[i];
   float y0 = pys[i];
   float m0 = mxs[i];
   __global float *force_x = dvx2d + t * N + i;
   __global float *force_y = dvy2d + t * N + i;
   calculateForce(x0, y0, m0, nodes, force_x, force_y);
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
