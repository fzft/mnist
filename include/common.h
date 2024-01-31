#ifndef MNIST_COMMON_H
#define MNIST_COMMON_H

#include "data_handler.h"
#include "data.h"
#include <vector>

class CommonData {
protected:
    std::vector<Data*>* training_data;
    std::vector<Data*>* test_data;
    std::vector<Data*>* validation_data;

public:
    void set_training_data(std::vector<Data*>* vect);
    void set_test_data(std::vector<Data*>* vect);
    void set_validation_data(std::vector<Data*>* vect);
};

#endif //MNIST_COMMON_H
