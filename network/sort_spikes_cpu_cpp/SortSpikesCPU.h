#ifndef SORTSPIKESCPU
#define SORTSPIKESCPU

#include <unistd.h>

#include <cctype>
#include <iostream>

class SortSpikesCPU
{
    public:
    SortSpikesCPU();
    ~SortSpikesCPU();

    void process(
        int64_t input_pointer_addr,
        int64_t input_dim_0,
        int64_t input_dim_1,
        int64_t input_dim_2,
        int64_t input_dim_3,
        int64_t output_pointer_addr,
        int64_t output_dim_0,
        int64_t output_dim_1,
        int64_t output_dim_2,
        int64_t output_dim_3,
        int64_t indices_pointer_addr,
        int64_t indices_dim_0,
        int64_t indices_dim_1,
        int64_t number_of_cpu_processes);

    void count(
        int64_t input_pointer_addr,
        int64_t input_dim_0,
        int64_t input_dim_1,
        int64_t input_dim_2,
        int64_t input_dim_3,
        int64_t output_pointer_addr,
        int64_t output_dim_0,
        int64_t output_dim_1,
        int64_t output_dim_2,
        int64_t indices_pointer_addr,
        int64_t indices_dim_0,
        int64_t indices_dim_1,
        int64_t number_of_cpu_processes);

    private:
    void process_pattern(
        int64_t* input_pointer,
        size_t input_dim_0,
        int64_t* output_pointer,
        size_t output_dim_0,
        size_t output_dim_1,
        size_t output_dim_2,
        int64_t* indices_pointer,
        size_t indices_dim_0,
        size_t indices_dim_1);

    void count_pattern(
        int64_t* input_pointer,
        size_t input_dim_0,
        int64_t* output_pointer,
        size_t output_dim_0,
        size_t output_dim_1,
        int64_t* indices_pointer,
        size_t indices_dim_0,
        size_t indices_dim_1);

};

#endif /* SORTSPIKESCPU */



