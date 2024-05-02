#include "parallel_for.h"
#include <iostream>

void parallel_for(int start, int end, int inc, void *(*functor)(int, void *), void *args, int num_threads) {
    std::vector<std::thread> threads;
    for (int rank = 0; rank < num_threads; ++rank) {
        threads.push_back(std::thread([=] {
            for (int i = start + rank * inc; i < end; i += num_threads * inc) {
                functor(i, args);
                // std::cerr << "Thread " << rank << " is processing element " << i << std::endl;
            }
            }));
    }

    for (auto &thread : threads)
        thread.join();
}