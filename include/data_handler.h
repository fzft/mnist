#ifndef TENSOR_DATA_HANDLER_H
#define TENSOR_DATA_HANDLER_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <map>
#include <unordered_set>

#include "data.h"

class DataHandler {
public:
    DataHandler();
    ~DataHandler();

    void read_feature_vector(std::string data_path);
    void read_label(std::string label_path);
    void split_data();
    void count_classes();

    uint32_t convert_to_little_endian(const unsigned char* bytes);
    std::vector<Data*>* get_training_data();
    std::vector<Data*>* get_test_data();
    std::vector<Data*>* get_validation_data();

    int get_num_classes();

private:
    std::vector<Data*>* data_array;
    std::vector<Data*>* training_data;
    std::vector<Data*>* test_data;
    std::vector<Data*>* validation_data;

    int num_classes;
    int feature_vector_size;
    std::map<uint8_t, int> class_map;

    const double TRAINING_DATA_RATIO = 0.75;
    const double TEST_DATA_RATIO = 0.20;
    const double VALIDATION_DATA_RATIO = 0.05;
};

#endif //TENSOR_DATA_HANDLER_H
