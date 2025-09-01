#include <bits/stdc++.h>
using namespace std;

struct Point {
    vector<double> features;
    int label;
};

// Euclidean distance
double euclideanDistance(const vector<double>& a, const vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Predict one sample
int predictOne(const vector<Point>& trainData, const vector<double>& query, int k) {
    vector<pair<double,int>> distances;

    // compute all distances
    for (auto& p : trainData) {
        double d = euclideanDistance(query, p.features);
        distances.push_back({d, p.label});
    }

    // sort by distance
    sort(distances.begin(), distances.end());

    // count votes among k nearest
    unordered_map<int,int> freq;
    for (int i = 0; i < k; i++) {
        freq[distances[i].second]++;
    }

    // majority voting
    int bestLabel = -1, maxCount = -1;
    for (auto& f : freq) {
        if (f.second > maxCount) {
            maxCount = f.second;
            bestLabel = f.first;
        }
    }

    return bestLabel;
}

// Predict batch of queries
vector<int> predictBatch(const vector<Point>& trainData, const vector<vector<double>>& queries, int k) {
    vector<int> results;
    for (auto& q : queries) {
        results.push_back(predictOne(trainData, q, k));
    }
    return results;
}

int main() {
    
    vector<Point> dataset = {
        {{1.0, 2.0}, 0},
        {{1.5, 1.8}, 0},
        {{5.0, 8.0}, 1},
        {{6.0, 9.0}, 1},
        {{1.2, 0.8}, 0},
        {{7.0, 7.0}, 1}
    };

    int k = 3;

    // Single query
    vector<double> query = {2.0, 2.0};
    int pred = predictOne(dataset, query, k);
    cout << "Predicted class: " << pred << endl;

    // Batch queries
    vector<vector<double>> queries = {{2.0, 2.0}, {6.5, 8.0}};
    vector<int> results = predictBatch(dataset, queries, k);

    cout << "Batch predictions: ";
    for (int r : results) cout << r << " ";
    cout << endl;

    return 0;
}
