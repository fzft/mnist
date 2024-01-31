#include "include/data_handler.h"

int main() {
    DataHandler data_handler;
    data_handler.read_label("./data/train-labels-idx1-ubyte");
    data_handler.read_feature_vector("./data/train-images-idx3-ubyte");
    data_handler.split_data();
    data_handler.count_classes();
}
