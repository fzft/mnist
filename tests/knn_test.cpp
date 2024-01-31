#include <gtest/gtest.h>
#include "knn.h"
#include "data_handler.h"
#include <iostream>


TEST(KNNTest, BasicAssertions) {
    std::cout << "KNNTest" << std::endl;
    DataHandler *data_handler = new DataHandler();
    data_handler->read_feature_vector("/Users/fangzhenfutao/ClionProjects/mnist/data/train-images-idx3-ubyte");
    data_handler->read_label("/Users/fangzhenfutao/ClionProjects/mnist/data/train-labels-idx1-ubyte");
    data_handler->split_data();
    data_handler->count_classes();

    KNN* knn = new KNN();
    knn->set_test_data(data_handler->get_test_data());
    knn->set_training_data(data_handler->get_training_data());
    knn->set_validation_data(data_handler->get_validation_data());
    knn->set_k(1);

    double performance = knn->validate_performance();
    std::cout << "performance: " << performance << std::endl;
}