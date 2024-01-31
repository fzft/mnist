#include "common.h"

void CommonData::set_training_data(std::vector<Data*>* vect) {
    this->training_data = vect;
}

void CommonData::set_test_data(std::vector<Data*>* vect) {
    this->test_data = vect;
}

void CommonData::set_validation_data(std::vector<Data*>* vect) {
    this->validation_data = vect;
}