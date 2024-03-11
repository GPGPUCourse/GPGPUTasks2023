#line 2

__kernel void matrix_multiplication_basic(global const float *matrixA, global const float *matrixB, __global float *matrixC, unsigned int rowCountA, unsigned int commonDimension, unsigned int columnCountB) {
    size_t columnIndex = get_global_id(0);
    size_t rowIndex = get_global_id(1);

    if (columnIndex >= columnCountB || rowIndex >= rowCountA) {
        return;
    }

    float accumulatedValue = 0.0;
    for (size_t iter = 0; iter < commonDimension; iter++) {
        accumulatedValue += matrixA[rowIndex * commonDimension + iter] * matrixB[iter * columnCountB + columnIndex];
    }
    matrixC[rowIndex * columnCountB + columnIndex] = accumulatedValue;
}


__kernel void matrix_multiplication_local(global const float *matrixA, global const float *matrixB, __global float *resultMatrix, unsigned int rowCountA, unsigned int commonDimension, unsigned int columnCountB) {
    size_t globalColumnIndex = get_global_id(0);
    size_t globalRowIndex = get_global_id(1);
    size_t localColumnIndex = get_local_id(0);
    size_t localRowIndex = get_local_id(1);
    __local float localTileA[TILE_SIZE][TILE_SIZE + 1];
    __local float localTileB[TILE_SIZE][TILE_SIZE + 1];

    float accumulatedValue = 0.0f;
    for (size_t tileIndex = 0; tileIndex * TILE_SIZE < commonDimension; tileIndex++) {
        const size_t offset = tileIndex * TILE_SIZE;
        if (globalRowIndex < rowCountA && (tileIndex * TILE_SIZE + localColumnIndex) < commonDimension) {
            localTileA[localRowIndex][localColumnIndex] = matrixA[globalRowIndex * commonDimension + offset + localColumnIndex];
        }

        if (globalColumnIndex < columnCountB && tileIndex * TILE_SIZE + localRowIndex < commonDimension) {
            localTileB[localRowIndex][localColumnIndex] = matrixB[(offset + localRowIndex) * columnCountB + globalColumnIndex];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        for (size_t innerIndex = 0; innerIndex < TILE_SIZE; innerIndex++) {
            accumulatedValue += localTileA[localRowIndex][innerIndex] * localTileB[innerIndex][localColumnIndex];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (globalRowIndex < rowCountA && globalColumnIndex < columnCountB) {
        resultMatrix[globalRowIndex * columnCountB + globalColumnIndex] = accumulatedValue;
    }
}

__kernel void matrix_multiplication_local_work(global const float *matrixA, global const float *matrixB, __global float *resultMatrix, unsigned int rowCountA, unsigned int commonDimension, unsigned int columnCountB) {
    size_t localRowIndex = get_local_id(0);
    size_t localColumnIndex = get_local_id(1);
    size_t globalRowIndex = get_global_id(0);
    size_t globalColumnIndex = get_group_id(1) * TILE_SIZE + localColumnIndex;

    __local float localTileA[TILE_SIZE][TILE_SIZE + 1];
    __local float localTileB[TILE_SIZE][TILE_SIZE + 1];
    const size_t WORK_STEP = TILE_SIZE / THREAD_WORK;

    float accumulatedValues[THREAD_WORK] = { 0 };

    for (size_t tileIndex = 0; tileIndex * TILE_SIZE < commonDimension; tileIndex++) {
        const size_t offset = tileIndex * TILE_SIZE;
        for (size_t workIndex = 0; workIndex * WORK_STEP < TILE_SIZE; workIndex++) {
            const size_t workOffset = workIndex * WORK_STEP;
            if (globalColumnIndex + workOffset < rowCountA && offset + localRowIndex < commonDimension) {
                localTileA[localColumnIndex + workOffset][localRowIndex] = matrixA[(globalColumnIndex + workOffset) * commonDimension + offset + localRowIndex];
            }

            if (globalRowIndex < columnCountB && offset + localColumnIndex + workOffset < commonDimension) {
                localTileB[localColumnIndex + workOffset][localRowIndex] = matrixB[(offset + localColumnIndex + workOffset) * columnCountB + globalRowIndex];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        const size_t end = (TILE_SIZE < (commonDimension - offset)) ? TILE_SIZE : (commonDimension - offset);
        for (size_t innerIndex = 0; innerIndex < end; innerIndex++) {
            const float elementB = localTileB[innerIndex][localRowIndex];
            for (size_t workIndex = 0; workIndex < THREAD_WORK; workIndex++) {
                if (globalRowIndex < columnCountB && globalColumnIndex + workIndex * WORK_STEP < rowCountA) {
                    accumulatedValues[workIndex] += elementB * localTileA[localColumnIndex + workIndex * WORK_STEP][innerIndex];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (size_t workIndex = 0; workIndex < THREAD_WORK; workIndex++) {
        if (globalRowIndex < columnCountB && globalColumnIndex + WORK_STEP * workIndex < rowCountA) {
            resultMatrix[(globalColumnIndex + WORK_STEP * workIndex) * columnCountB + globalRowIndex] = accumulatedValues[workIndex];
        }
    }
}