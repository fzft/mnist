include(FetchContent)
FetchContent_Declare(
        googletest
        SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../lib/googletest
)

FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

add_executable(
        mnist_test
        hello_test.cpp
        data_test.cpp
        knn_test.cpp
        kmeas_test.cpp
        data_handler_test.cpp
       )

target_link_libraries(
        mnist_test
        mnist
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(mnist_test)