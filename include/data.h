#ifndef MINST_DATA_H
#define MINST_DATA_H

#include <vector>

class Data {
public:

    Data();
    ~Data();
    void set_feature_vector(std::vector<uint8_t>* feature_vector);
    void append_to_feature_vector(uint8_t feature);
    void set_label(uint8_t label);
    void set_enum_label(int enum_label);

    int get_feature_vector_size();
    uint8_t get_label();
    uint8_t get_enum_label();

    std::vector<uint8_t>* get_feature_vector();

private:
    std::vector<uint8_t>* feature_vector;
    uint8_t label;
    int enum_label;
};

#endif //MINST_DATA_H
