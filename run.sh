#!/bin/sh

./build/matrix_multiplication "matrix_multiplication_naive"
./build/matrix_multiplication "matrix_multiplication_local_memory"
./build/matrix_multiplication "matrix_multiplication_more_work_per_thread"