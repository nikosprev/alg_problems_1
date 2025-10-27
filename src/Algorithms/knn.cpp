#include <iostream> 
#include <cstdlib>
#include <vector> 
#include <random>
#include <ctime>
#include <fstream>
#include <queue> 
#include "../../include/Algorithms/knn.hpp"


std::vector<Neighbor> kNN(const std::vector<std::vector<float>> &points ,const std::vector<float> &p , int k){ 
    std::priority_queue<Neighbor>  topK;
    for (auto&  q : points){ 
       double dist_qp =  euclidean_distance( q ,p); 
       Neighbor pq = {&q ,dist_qp}; 
       topK.push(pq); 
       if (topK.size() > k ) { 
        topK.pop() ; 
       }
    }
    
    std::vector<Neighbor> neighbors;
    while (!topK.empty()) {
        neighbors.push_back(topK.top());
        topK.pop();
    }

   return neighbors; 


}

