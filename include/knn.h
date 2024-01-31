#ifndef MNIST_KNN_H
#define MNIST_KNN_H

#include "common.h"

class KNN: public CommonData{
private:
    int k;
    std::vector<Data*>* neighbors;

public:
    KNN(int k);
    KNN();
    ~KNN();

    void find_k_nearest_neighbors(Data* test_data);

    void set_k(int k);

    int predict();
    double calculate_distance(Data* query_data, Data* input);
    double validate_performance();
    double test_performance();




};

#endif //MNIST_KNN_H
