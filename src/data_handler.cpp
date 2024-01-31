#include "data_handler.h"

using namespace std;

DataHandler::DataHandler() {
    this->data_array = new std::vector<Data*>();
    this->training_data = new std::vector<Data*>();
    this->test_data = new std::vector<Data*>();
    this->validation_data = new std::vector<Data*>();
}

DataHandler::~DataHandler() {
    delete this->data_array;
    delete this->training_data;
    delete this->test_data;
    delete this->validation_data;
}

void DataHandler::read_feature_vector(std::string data_path) {

    std::cout << "reading feature vector" << std::endl;

    uint32_t header[4]; // magic number, number of images, number of rows, number of columns
    unsigned char bytes[4];
    std::ifstream data_file(data_path, std::ios::binary);
    if (!data_file.is_open()) {
        std::cout << "Failed to open data file." << std::endl;
        exit(1);
    }

    uint32_t magic_number = 0;
    uint32_t num_images = 0;
    uint32_t num_rows = 0;
    uint32_t num_cols = 0;

    data_file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = this->convert_to_little_endian(reinterpret_cast<unsigned char*>(&magic_number));
    if (magic_number != 2051) {
        std::cout << "Invalid magic number, probably not a MNIST data file." << std::endl;
        exit(1);
    }

    std::cout << "magic_number: " << magic_number << std::endl;

    data_file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    num_images = this->convert_to_little_endian(reinterpret_cast<unsigned char*>(&num_images));

    std::cout << "num_images: " << num_images << std::endl;

    data_file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
    num_rows = this->convert_to_little_endian(reinterpret_cast<unsigned char*>(&num_rows));

    std::cout << "num_rows: " << num_rows << std::endl;

    data_file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
    num_cols = this->convert_to_little_endian(reinterpret_cast<unsigned char*>(&num_cols));

    std::cout << "num_cols: " << num_cols << std::endl;

    this->feature_vector_size = num_rows * num_cols;

    for (int i = 0; i < num_images; i++) {
        Data* data = new Data();
        std::vector<uint8_t>* feature_vector = new std::vector<uint8_t>();
        for (int j = 0; j < this->feature_vector_size; j++) {
            uint8_t feature = 0;
            data_file.read(reinterpret_cast<char*>(&feature), sizeof(feature));
            if (!data_file) {
                std::cout << "Failed to read image data." << std::endl;
                exit(1);
            }
            feature_vector->push_back(feature);
        }
        data->set_feature_vector(feature_vector);
        this->data_array->push_back(data);
    }
    std::cout << "successfully read and store feature vector" << num_images << std::endl;
}

void DataHandler::read_label(std::string label_path) {

    std::cout << "reading label" << std::endl;

    uint32_t header[2]; // magic number, number of items
    unsigned char bytes[4];
    std::ifstream label_file(label_path, std::ios::binary);
    if (!label_file.is_open()) {
        std::cout << "Failed to open label file." << std::endl;
        exit(1);
    }

    uint32_t magic_number = 0;
    uint32_t num_items = 0;

    label_file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = this->convert_to_little_endian(reinterpret_cast<unsigned char*>(&magic_number));
    if (magic_number != 2049) {
        std::cout << "Invalid magic number, probably not a MNIST label file." << std::endl;
        exit(1);
    }

    std::cout << "magic_number: " << magic_number << std::endl;

    label_file.read(reinterpret_cast<char*>(&num_items), sizeof(num_items));
    num_items = this->convert_to_little_endian(reinterpret_cast<unsigned char*>(&num_items));

    std::cout << "num_items: " << num_items << std::endl;

    for (int i = 0; i < num_items; i++) {
        uint8_t label = 0;
        label_file.read(reinterpret_cast<char*>(&label), sizeof(label));
        this->data_array->at(i)->set_label(label);
    }
    std::cout << "successfully read and store label" << num_items << std::endl;
}

void DataHandler::split_data() {
    int num_data = this->data_array->size();
    int num_training_data = num_data * this->TRAINING_DATA_RATIO;
    int num_test_data = num_data * this->TEST_DATA_RATIO;
    int num_validation_data = num_data * this->VALIDATION_DATA_RATIO;

    std::unordered_set<int> training_data_index_set;
    std::unordered_set<int> test_data_index_set;
    std::unordered_set<int> validation_data_index_set;

    while (training_data_index_set.size() < num_training_data) {
        int random_index = rand() % num_data;
        if (training_data_index_set.find(random_index) == training_data_index_set.end()) {
            training_data_index_set.insert(random_index);
            this->training_data->push_back(this->data_array->at(random_index));
        }
    }

    while (test_data_index_set.size() < num_test_data) {
        int random_index = rand() % num_data;
        if (training_data_index_set.find(random_index) == training_data_index_set.end() &&
            test_data_index_set.find(random_index) == test_data_index_set.end()) {
            test_data_index_set.insert(random_index);
            this->test_data->push_back(this->data_array->at(random_index));
        }
    }

    while (validation_data_index_set.size() < num_validation_data) {
        int random_index = rand() % num_data;
        if (training_data_index_set.find(random_index) == training_data_index_set.end() &&
            test_data_index_set.find(random_index) == test_data_index_set.end() &&
            validation_data_index_set.find(random_index) == validation_data_index_set.end()) {
            validation_data_index_set.insert(random_index);
            this->validation_data->push_back(this->data_array->at(random_index));
        }
    }
}

void DataHandler::count_classes() {
    for (int i = 0; i < this->data_array->size(); i++) {
        uint8_t label = this->data_array->at(i)->get_label();
        if (this->class_map.find(label) == this->class_map.end()) {
            this->class_map[label] = 1;
        } else {
            this->class_map[label] += 1;
        }
        this->data_array->at(i)->set_enum_label(this->class_map[label]);
    }
    this->num_classes = this->class_map.size();
    cout << "num_classes: " << this->num_classes << endl;
}

uint32_t DataHandler::convert_to_little_endian(const unsigned char *bytes) {
    return (uint32_t)((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3]);
}

std::vector<Data*>* DataHandler::get_training_data() {
    return this->training_data;
}

std::vector<Data*>* DataHandler::get_test_data() {
    return this->test_data;
}

std::vector<Data*>* DataHandler::get_validation_data() {
    return this->validation_data;
}

int DataHandler::get_num_classes() {
    return this->num_classes;
}










