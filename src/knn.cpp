#include "knn.h"
#include <cmath>
#include <map>
#include <iostream>

#define EUCLIED

KNN::KNN(int k) {
    this->k = k;
}

KNN::KNN() {
    // NOTHING TO DO
}

KNN::~KNN() {
    // NOTHING TO DO
}

// O(N)
void KNN::find_k_nearest_neighbors(Data* query_point) {
    std::map<double, Data*>* distance_map = new std::map<double, Data*>();
    for (int i = 0; i < this->training_data->size(); i++) {
        double distance = this->calculate_distance(query_point, this->training_data->at(i));
        distance_map->insert(std::pair<double, Data*>(distance, this->training_data->at(i)));
    }
    for (int i = 0; i < this->k; i++) {
        this->neighbors->push_back(distance_map->begin()->second);
        distance_map->erase(distance_map->begin());
    }
    delete distance_map;
}

void KNN::set_k(int k) {
    this->k = k;
}

double KNN::calculate_distance(Data* query_point, Data* data_point) {
    double distance = 0;
    std::vector<uint8_t>* query_point_feature_vector = query_point->get_feature_vector();
    std::vector<uint8_t>* data_point_feature_vector = data_point->get_feature_vector();
    if (query_point_feature_vector->size() != data_point_feature_vector->size()) {
        std::cout << "ERROR: query_point_feature_vector->size() != data_point_feature_vector->size()" << std::endl;
        exit(1);
    }
#ifdef EUCLIED
    for (int i = 0; i < query_point_feature_vector->size(); i++) {
        distance += pow((query_point_feature_vector->at(i) - data_point_feature_vector->at(i)), 2);
    }
    distance = sqrt(distance);
#elif defined MANHATTAN
    for (int i = 0; i < query_point_feature_vector->size(); i++) {
        distance += abs(query_point_feature_vector->at(i) - data_point_feature_vector->at(i));
    }
#endif
    return distance;
}

int KNN::predict() {
    std::map<uint8_t, int> class_map;
    for (int i = 0; i < this->neighbors->size(); i++) {
        uint8_t label = this->neighbors->at(i)->get_label();
        if (class_map.find(label) == class_map.end()) {
            class_map[label] = 1;
        } else {
            class_map[label] += 1;
        }
    }
    int max_count = 0;
    int max_label = 0;
    for (std::map<uint8_t, int>::iterator it = class_map.begin(); it != class_map.end(); it++) {
        if (it->second > max_count) {
            max_count = it->second;
            max_label = it->first;
        }
    }
    return max_label;
}

double KNN::validate_performance() {
    int correct_count = 0;
    for (int i = 0; i < this->validation_data->size(); i++) {
        this->neighbors = new std::vector<Data*>();
        this->find_k_nearest_neighbors(this->validation_data->at(i));
        int prediction = this->predict();
        if (prediction == this->validation_data->at(i)->get_label()) {
            correct_count++;
            std::cout << "validate performance:" << std::setprecision(3) << (double)correct_count / this->validation_data->size() << std::endl;
        }
        delete this->neighbors;
    }
    return (double)correct_count / this->validation_data->size();
}

double KNN::test_performance() {
    int correct_count = 0;
    for (int i = 0; i < this->test_data->size(); i++) {
        this->neighbors = new std::vector<Data*>();
        this->find_k_nearest_neighbors(this->test_data->at(i));
        int prediction = this->predict();
        if (prediction == this->test_data->at(i)->get_label()) {
            correct_count++;
            std::cout << "test performance:" << std::setprecision(3) << (double)correct_count / this->test_data->size() << std::endl;
        }
        delete this->neighbors;
    }
    return (double)correct_count / this->test_data->size();
}





