#ifndef TENSOR_KMEANS_H
#define TENSOR_KMEANS_H

#include "common.h"
#include <unordered_set>
#include <cmath>
#include "data_handler.h"

typedef struct cluster {
    std::vector<double> *centroid;
    std::vector<Data *> *cluster_points;
    std::map<uint8_t, int> class_counts;
    int most_freq_class;

    cluster(Data *initial_point) {
        centroid = new std::vector<double>();
        cluster_points = new std::vector<Data *>();
        for (auto value: *(initial_point->get_feature_vector())) {
            centroid->push_back(value);
        }
        cluster_points->push_back(initial_point);
        class_counts.insert(std::pair<uint8_t, int>(initial_point->get_label(), 1));
        most_freq_class = initial_point->get_label();
    }

    void add_to_cluster(Data *point) {
        int prev_size = cluster_points->size();
        cluster_points->push_back(point);
        for (int i = 0; i < centroid->size(); i++) {
            centroid->at(i) = (centroid->at(i) * prev_size + point->get_feature_vector()->at(i)) / cluster_points->size();
        }
        if (class_counts.find(point->get_label()) == class_counts.end()) {
            class_counts.insert(std::pair<uint8_t, int>(point->get_label(), 1));
        } else {
            class_counts[point->get_label()]++;
        }
        set_most_freq_class();
    }

    void set_most_freq_class() {
        int max = 0;
        for (auto pair: class_counts) {
            if (pair.second > max) {
                max = pair.second;
                most_freq_class = pair.first;
            }
        }
    }

} cluster_t;

class KMeans : public CommonData {
private:
    int num_clusters;
    std::vector<cluster_t *> *clusters;
    std::unordered_set<int> *used_indexes;
public:
    KMeans(int k);
    void init_clusters();
    void init_clusters_for_each_class();
    void train();
    double euclidean_distance(std::vector<double> *, Data *);
    double validate();
    double test();

};

#endif //TENSOR_KMEANS_H
