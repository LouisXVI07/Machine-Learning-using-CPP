#include <bits/stdc++.h>
using namespace std;
struct Node {
    bool is_leaf;
    int feature_index;
    double threshold;
    double prediction;
    Node* left;
    Node* right;

    Node() : is_leaf(false), feature_index(-1), threshold(0.0), prediction(0.0), left(nullptr), right(nullptr) {}
};
double entropy(const vector<int>& labels) {
    int n = labels.size();
    if (n == 0) return 0.0;
    int count0 = 0, count1 = 0;
    for (int lbl : labels) {
        if (lbl == 0) count0++;
        else count1++;
    }
    double p0 = (double)count0 / n;
    double p1 = (double)count1 / n;

    double ent = 0.0;
    if (p0 > 0) ent -= p0 * log2(p0);
    if (p1 > 0) ent -= p1 * log2(p1);

    return ent;
}
double information_gain(const vector<int>& parent,const vector<int>& left,const vector<int>& right) {
    int n = parent.size();
    if (n == 0) return 0.0;

    double H_parent = entropy(parent);

    double H_left = entropy(left);
    double H_right = entropy(right);

    double w_left = (double)left.size() / n;
    double w_right = (double)right.size() / n;

    double IG = H_parent - (w_left * H_left + w_right * H_right);
    return IG;
}
Node* build_tree(const vector<vector<double>>& X, const vector<int>& y,int depth,int max_depth) {
    Node* node = new Node();

    if (depth >= max_depth || entropy(y) == 0.0) {
        node->is_leaf = true;
        int count1 = count(y.begin(), y.end(), 1);
        node->prediction = (count1 >= (int)y.size() - count1) ? 1.0 : 0.0;
        return node;
    }

    int n_samples = X.size();
    int n_features = X[0].size();

    double best_ig = -1.0;
    int best_feature = -1;
    double best_threshold = 0.0;
    vector<int> best_left, best_right;
    vector<vector<double>> X_left, X_right;

    for (int f = 0; f < n_features; f++) {
        
        vector<double> values;
        for (int i = 0; i < n_samples; i++) values.push_back(X[i][f]);
        sort(values.begin(), values.end());
        values.erase(unique(values.begin(), values.end()), values.end());

        for (double thr : values) {
            vector<int> y_left, y_right;
            vector<vector<double>> Xl, Xr;

            for (int i = 0; i < n_samples; i++) {
                if (X[i][f] <= thr) {
                    y_left.push_back(y[i]);
                    Xl.push_back(X[i]);
                } else {
                    y_right.push_back(y[i]);
                    Xr.push_back(X[i]);
                }
            }

            if (y_left.empty() || y_right.empty()) continue;

            double ig = information_gain(y, y_left, y_right);
            if (ig > best_ig) {
                best_ig = ig;
                best_feature = f;
                best_threshold = thr;
                best_left = y_left;
                best_right = y_right;
                X_left = Xl;
                X_right = Xr;
            }
        }
    }

    
    if (best_ig <= 0) {
        node->is_leaf = true;
        int count1 = count(y.begin(), y.end(), 1);
        node->prediction = (count1 >= (int)y.size() - count1) ? 1.0 : 0.0;
        return node;
    }

    node->feature_index = best_feature;
    node->threshold = best_threshold;
    node->left = build_tree(X_left, best_left, depth + 1, max_depth);
    node->right = build_tree(X_right, best_right, depth + 1, max_depth);

    return node;
}
double predict(Node* node, const vector<double>& sample) {
    if (node->is_leaf) return node->prediction;
    if (sample[node->feature_index] <= node->threshold)
        return predict(node->left, sample);
    else
        return predict(node->right, sample);
}

int main(){
    vector<vector<double>> X = {
        {2.5, 1.5}, {1.0, 2.0}, {3.5, 0.5}, {3.0, 1.0},
        {0.5, 3.0}, {2.0, 2.5}, {3.0, 3.0}, {1.5, 0.5}
    };
    vector<int> y = {0, 0, 1, 1, 0, 0, 1, 0};

    Node* root = build_tree(X, y, 0, 3); // max_depth = 3

    vector<double> test = {3.2, 0.8};
    cout << "Prediction: " << predict(root, test) << endl;

    return 0;
}