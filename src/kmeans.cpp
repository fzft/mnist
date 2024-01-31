#include "kmeans.h"
#include <limits>
#include <vector>

KMeans::KMeans(int k) {
    this->num_clusters = k;
    this->clusters = new std::vector<cluster_t *>();
    this->used_indexes = new std::unordered_set<int>();
}

void KMeans::init_clusters() {
    for (int i = 0; i < num_clusters; i++) {
        int index = rand() % training_data->size();
        while (used_indexes->find(index) != used_indexes->end()) {
            index = rand() % training_data->size();
        }
        used_indexes->insert(index);
        cluster_t *cluster = new cluster_t(training_data->at(index));
        clusters->push_back(cluster);
    }
}

void KMeans::init_clusters_for_each_class() {
    std::unordered_set<int> classes_used ;
    for (int i = 0; i < training_data->size(); i++) {
        if (classes_used.find(training_data->at(i)->get_label()) == classes_used.end()) {
            classes_used.insert(training_data->at(i)->get_label());
            cluster_t *cluster = new cluster_t(training_data->at(i));
            clusters->push_back(cluster);
            used_indexes->insert(i);
        }
    }
}

void KMeans::train() {
    while (used_indexes->size() < training_data->size()) {
        int index = rand() % training_data->size();
        while (used_indexes->find(index) != used_indexes->end()) {
            index = rand() % training_data->size();
        }
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for (int i = 0; i < clusters->size(); i++) {
            double dist = euclidean_distance(clusters->at(i)->centroid, training_data->at(index));
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = i;
            }
        }
        clusters->at(best_cluster)->add_to_cluster(training_data->at(index));
        used_indexes->insert(index);
    }
}

double KMeans::euclidean_distance(std::vector<double> *centroid, Data *point) {
    double distance = 0;
    for (int i = 0; i < centroid->size(); i++) {
        distance += pow(centroid->at(i) - point->get_feature_vector()->at(i), 2);
    }
    return sqrt(distance);
}

double KMeans::validate() {
    int correct = 0;
    for (int i = 0; i < validation_data->size(); i++) {
        double min_distance = euclidean_distance(clusters->at(0)->centroid, validation_data->at(i));
        int min_index = 0;
        for (int j = 1; j < clusters->size(); j++) {
            double distance = euclidean_distance(clusters->at(j)->centroid, validation_data->at(i));
            if (distance < min_distance) {
                min_distance = distance;
                min_index = j;
            }
        }
        if (clusters->at(min_index)->most_freq_class == validation_data->at(i)->get_label()) {
            correct++;
            std::cout << "validate performance:" << std::setprecision(3) << (double)correct / this->validation_data->size() << std::endl;
        }
    }
    return (double) correct / validation_data->size();
}

double KMeans::test() {
    int correct = 0;
    for (int i = 0; i < test_data->size(); i++) {
        double min_distance = euclidean_distance(clusters->at(0)->centroid, test_data->at(i));
        int min_index = 0;
        for (int j = 1; j < clusters->size(); j++) {
            double distance = euclidean_distance(clusters->at(j)->centroid, test_data->at(i));
            if (distance < min_distance) {
                min_distance = distance;
                min_index = j;
            }
        }
        if (clusters->at(min_index)->most_freq_class == test_data->at(i)->get_label()) {
            correct++;
            std::cout << "test performance:" << std::setprecision(3) << (double)correct / this->test_data->size() << std::endl;
        }
    }
    return (double) correct / test_data->size();
}
