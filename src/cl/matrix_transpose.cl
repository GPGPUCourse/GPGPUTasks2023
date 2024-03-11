#line 2

#define TILE_SIZE 16
__kernel void matrix_transpose(global const float *inputMatrix, global float *transposedMatrix, unsigned int rows, unsigned int columns) {
    int columnIndex = get_global_id(0);
    int rowIndex = get_global_id(1);

    __local float localTile[TILE_SIZE][TILE_SIZE];
    int localColumnIndex = get_local_id(0);
    int localRowIndex = get_local_id(1);

    if (columnIndex < columns && rowIndex < rows) {
        localTile[localRowIndex][localColumnIndex] = inputMatrix[rowIndex * columns + columnIndex];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (columnIndex < columns && rowIndex < rows) {
        transposedMatrix[columnIndex * rows + rowIndex] = localTile[localRowIndex][localColumnIndex];
    }
}


__kernel void matrix_transpose_banks(global const float *inputMatrix, global float *transposedMatrix, unsigned int rowCount, unsigned int columnCount)
{
    int columnIndex = get_global_id(0);
    int rowIndex = get_global_id(1);

    __local float localTile[TILE_SIZE][TILE_SIZE + 1];
    int localColumnIndex = get_local_id(0);
    int localRowIndex = get_local_id(1);

    if (columnIndex < columnCount && rowIndex < rowCount) {
        localTile[localRowIndex][localColumnIndex] = inputMatrix[rowIndex * columnCount + columnIndex];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (columnIndex < columnCount && rowIndex < rowCount) {
        transposedMatrix[columnIndex * rowCount + rowIndex] = localTile[localRowIndex][localColumnIndex];
    }
}
