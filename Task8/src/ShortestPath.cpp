#include <iostream>
#include <thread>
#include <map>
#include <queue>
#include <vector>

#include "csv.h"

class DirectedGraph {
protected:
    class Node;
    class Edge;

    class Node {
    public:
        Node() = default;
        Node(const std::string &name) : name(name) {}
        std::string name;
        std::map<Node *, double> distance;
        Edge *finalEdge = nullptr;
    };

    class Edge {
    public:
        Edge() = default;
        Edge(Node *target, double distance, Edge *nextEdge) : target(target), distance(distance), nextEdge{ nextEdge } {}

        Node *target = nullptr;
        double distance = std::numeric_limits<double>::max();
        Edge *nextEdge = nullptr;
    };

    std::map<std::string, Node *> nodes;

    void dijkstra(Node *start) {
        std::priority_queue<std::pair<int, Node *>, std::vector<std::pair<int, Node *>>, std::greater<std::pair<int, Node *>>> pq;
        auto &distance = start->distance;
        distance.clear();

        pq.push({ 0, start });
        distance[start] = 0;

        int count = 0, size = nodes.size();

        while (!pq.empty()) {
            auto [dist, node] = pq.top();
            pq.pop();

            if (distance.find(node) != distance.end() && dist > distance[node])
                continue;

            distance[node] = dist;
            count++;

            if (count == size)
                break;

            for (auto edge = node->finalEdge; edge != nullptr; edge = edge->nextEdge) {
                double newDist = dist + edge->distance;
                auto target = edge->target;
                if (distance.find(target) == distance.end() || newDist < distance[target]) {
                    distance[target] = newDist;
                    target->finalEdge = edge;
                    pq.push({ newDist, target });
                }
            }
        }
    }

public:
    DirectedGraph() = default;
    DirectedGraph(const std::string &csv_path) {
        io::CSVReader<3> in(csv_path);
        in.read_header(io::ignore_extra_column, "source", "target", "distance");
        std::string source; std::string target; double distance;
        while (in.read_row(source, target, distance)) {
            if (nodes.find(source) == nodes.end())
                nodes[source] = new Node(source);

            if (nodes.find(target) == nodes.end())
                nodes[target] = new Node(target);

            nodes[source]->finalEdge = new Edge(nodes[target], distance, nodes[source]->finalEdge);
        }
    }

    enum class Mode {
        PARALLEL,
        SEQUENTIAL
    };

    void solveShortestPath(const int &threadsLimit = 8, Mode mode = Mode::PARALLEL) {
        if (mode == Mode::SEQUENTIAL) {
            for (auto &node : nodes)
                dijkstra(node.second);

            return;
        }

        // std::vector<std::thread> threads;
        // for (auto &node : nodes)
        //     threads.push_back(std::thread(&DirectedGraph::dijkstra, this, node.second));

        // for (auto &thread : threads)
        //     thread.join();

        std::vector<std::thread> threads;
        int threadsCount = 0;
        for (auto &node : nodes) {
            threads.push_back(std::thread(&DirectedGraph::dijkstra, this, node.second));
            threadsCount++;
            if (threadsCount == threadsLimit) {
                for (auto &thread : threads)
                    thread.join();
                threads.clear();
                threadsCount = 0;
            }
        }

        for (auto &thread : threads)
            thread.join();
        threads.clear();
    }
};

class UndirectedGraph : public DirectedGraph {
public:
    UndirectedGraph() = default;
    UndirectedGraph(const std::string &csv_path) {
        io::CSVReader<3> in(csv_path);
        in.read_header(io::ignore_extra_column, "source", "target", "distance");
        std::string source; std::string target; double distance;
        while (in.read_row(source, target, distance)) {
            if (nodes.find(source) == nodes.end())
                nodes[source] = new Node(source);

            if (nodes.find(target) == nodes.end())
                nodes[target] = new Node(target);

            nodes[source]->finalEdge = new Edge(nodes[target], distance, nodes[source]->finalEdge);
            nodes[target]->finalEdge = new Edge(nodes[source], distance, nodes[target]->finalEdge);
        }
    }
};


int main(int argc, char const *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <csv_path> <threads>" << std::endl;
        return 1;
    }

    auto graph = UndirectedGraph(argv[1]);
    int threadsLimit = std::atoi(argv[2]);

    auto beginTime = std::chrono::high_resolution_clock::now();

    graph.solveShortestPath(threadsLimit);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - beginTime).count();
    std::cout << "Parallel Time elapsed: " << duration << " milliseconds" << std::endl;

    beginTime = std::chrono::high_resolution_clock::now();

    graph.solveShortestPath(threadsLimit, UndirectedGraph::Mode::SEQUENTIAL);

    endTime = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - beginTime).count();
    std::cout << "Sequential Time elapsed: " << duration << " milliseconds" << std::endl;

    return 0;
}

