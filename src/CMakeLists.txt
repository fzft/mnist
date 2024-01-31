add_library(mnist data.cpp data_handler.cpp knn.cpp common.cpp kmeans.cpp)

target_include_directories(mnist PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)