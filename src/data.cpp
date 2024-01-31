#include "data.h"

Data::Data() {
    this->feature_vector = new std::vector<uint8_t>();
}

Data::~Data() {
    delete this->feature_vector;
}

void Data::set_feature_vector(std::vector<uint8_t> *feature_vector) {
    this->feature_vector = feature_vector;
}

void Data::append_to_feature_vector(uint8_t feature) {
    this->feature_vector->push_back(feature);
}

void Data::set_label(uint8_t label) {
    this->label = label;
}

void Data::set_enum_label(int enum_label) {
    this->enum_label = enum_label;
}

int Data::get_feature_vector_size() {
    return this->feature_vector->size();
}

uint8_t Data::get_label() {
    return this->label;
}

uint8_t Data::get_enum_label() {
    return this->enum_label;
}

std::vector<uint8_t> *Data::get_feature_vector() {
    return this->feature_vector;
}