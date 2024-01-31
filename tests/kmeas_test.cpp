#include "kmeans.h"
#include <gtest/gtest.h>
#include <iostream>
#include <vector>


TEST(KMEANSTEST, BasicAssertions) {
    std::cout << "KMEANSTEST" << std::endl;
    DataHandler *data_handler = new DataHandler();
    data_handler->read_feature_vector("/Users/fangzhenfutao/ClionProjects/mnist/data/train-images-idx3-ubyte");
    data_handler->read_label("/Users/fangzhenfutao/ClionProjects/mnist/data/train-labels-idx1-ubyte");
    data_handler->split_data();
    data_handler->count_classes();

    double performance = 0;
    double best_performance = 0;

    for (int k = data_handler->get_num_classes(); k < data_handler->get_training_data()->size(); k += data_handler->get_num_classes()) {
        KMeans *kmeans = new KMeans(k);
        kmeans->set_training_data(data_handler->get_training_data());
        kmeans->set_test_data(data_handler->get_test_data());
        kmeans->set_validation_data(data_handler->get_validation_data());
        kmeans->init_clusters();
        kmeans->train();
        performance = kmeans->validate();
        std::cout << "current performance k: " << k << "=" << std::setprecision(3) << performance << std::endl;
        if (performance > best_performance) {
            best_performance = performance;
        }
    }
}