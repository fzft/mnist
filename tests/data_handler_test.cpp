#include "data_handler.h"
#include <gtest/gtest.h>
#include <iostream>

TEST(DataHandlerTest, BasicAssertions) {
    std::cout << "DataHandlerTest" << std::endl;
    DataHandler* data_handler = new DataHandler();
    data_handler->read_feature_vector("/Users/fangzhenfutao/ClionProjects/mnist/data/train-images-idx3-ubyte");
    data_handler->read_label("/Users/fangzhenfutao/ClionProjects/mnist/data/train-labels-idx1-ubyte");
    data_handler->split_data();
    data_handler->count_classes();
}




