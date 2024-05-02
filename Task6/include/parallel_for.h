#include <thread>
#include <vector>

void parallel_for(int start, int end, int inc, void *(*functor)(int, void *), void *args, int num_threads);