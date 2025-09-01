#include <bits/stdc++.h>
using namespace std;

struct Point {
    vector<double> features;
    int cluster; 
};


double euclideanDistance(const vector<double>& a, const vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}


vector<vector<double>> initializeCentroids(const vector<Point>& data, int k) {
    vector<vector<double>> centroids;
    vector<int> chosen;
    srand(time(0));

    while ((int)centroids.size() < k) {
        int idx = rand() % data.size();
        if (find(chosen.begin(), chosen.end(), idx) == chosen.end()) {
            centroids.push_back(data[idx].features);
            chosen.push_back(idx);
        }
    }
    return centroids;
}


void assignClusters(vector<Point>& data, const vector<vector<double>>& centroids) {
    for (auto& p : data) {
        double bestDist = 1e18;
        int bestCluster = -1;
        for (int i = 0; i < (int)centroids.size(); i++) {
            double d = euclideanDistance(p.features, centroids[i]);
            if (d < bestDist) {
                bestDist = d;
                bestCluster = i;
            }
        }
        p.cluster = bestCluster;
    }
}


vector<vector<double>> updateCentroids(const vector<Point>& data, int k, int dim) {
    vector<vector<double>> newCentroids(k, vector<double>(dim, 0.0));
    vector<int> counts(k, 0);

    for (auto& p : data) {
        counts[p.cluster]++;
        for (int j = 0; j < dim; j++) {
            newCentroids[p.cluster][j] += p.features[j];
        }
    }

    for (int i = 0; i < k; i++) {
        if (counts[i] > 0) {
            for (int j = 0; j < dim; j++) {
                newCentroids[i][j] /= counts[i];
            }
        }
    }

    return newCentroids;
}


void kmeans(vector<Point>& data, int k, int maxIters = 100) {
    int dim = data[0].features.size();
    vector<vector<double>> centroids = initializeCentroids(data, k);

    for (int iter = 0; iter < maxIters; iter++) {
        assignClusters(data, centroids);
        vector<vector<double>> newCentroids = updateCentroids(data, k, dim);

       
        if (newCentroids == centroids) {
            cout << "Converged in " << iter+1 << " iterations.\n";
            break;
        }
        centroids = newCentroids;
    }

    cout << "Final centroids:\n";
    for (int i = 0; i < k; i++) {
        cout << "Cluster " << i << ": ";
        for (auto val : centroids[i]) cout << val << " ";
        cout << endl;
    }
}

int main() {
    
    vector<Point> dataset = {
        {{1.0, 2.0}, -1},
        {{1.5, 1.8}, -1},
        {{5.0, 8.0}, -1},
        {{6.0, 9.0}, -1},
        {{1.2, 0.8}, -1},
        {{7.0, 7.0}, -1}
    };

    int k = 2; 
    kmeans(dataset, k);

   
    cout << "\nCluster assignments:\n";
    for (auto& p : dataset) {
        cout << "(";
        for (auto val : p.features) cout << val << " ";
        cout << ") -> Cluster " << p.cluster << "\n";
    }

    return 0;
}
