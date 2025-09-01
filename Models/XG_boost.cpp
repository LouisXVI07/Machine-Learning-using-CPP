#include <bits/stdc++.h>
using namespace std;

struct TreeNode {
    bool is_leaf;
    double value;
    int feature;
    double threshold;
    TreeNode* left;
    TreeNode* right;

    TreeNode(double val): is_leaf(true), value(val), feature(-1), threshold(0), left(nullptr), right(nullptr) {}
    TreeNode(int feat, double th): is_leaf(false), value(0), feature(feat), threshold(th), left(nullptr), right(nullptr) {}
};

void free_tree(TreeNode* node) {
    if(!node) return;
    free_tree(node->left);
    free_tree(node->right);
    delete node;
}

double predict(TreeNode* node, const vector<double>& x) {
    if(node->is_leaf) return node->value;
    if(x[node->feature] <= node->threshold)
        return predict(node->left, x);
    else
        return predict(node->right, x);
}

TreeNode* build_tree(const vector<vector<double>>& X, const vector<double>& y, int depth) {
    if(depth == 0 || y.size() <= 1) {
        double sum = accumulate(y.begin(), y.end(), 0.0);
        return new TreeNode(sum / y.size());
    }

    int n = X.size();
    int m = X[0].size();
    double best_mse = 1e18;
    int best_feature = -1;
    double best_threshold = 0;
    vector<int> best_left_idx, best_right_idx;

    for(int feature = 0; feature < m; feature++) {
        vector<pair<double,int>> vals;
        for(int i = 0; i < n; i++) vals.push_back({X[i][feature], i});
        sort(vals.begin(), vals.end());

        for(int i = 1; i < n; i++) {
            double threshold = (vals[i-1].first + vals[i].first) / 2.0;
            vector<int> left_idx, right_idx;
            for(int j = 0; j < n; j++) {
                if(X[j][feature] <= threshold) left_idx.push_back(j);
                else right_idx.push_back(j);
            }
            if(left_idx.empty() || right_idx.empty()) continue;

            double left_mean = 0, right_mean = 0;
            for(int id: left_idx) left_mean += y[id];
            for(int id: right_idx) right_mean += y[id];
            left_mean /= left_idx.size();
            right_mean /= right_idx.size();

            double mse = 0;
            for(int id: left_idx) mse += (y[id] - left_mean) * (y[id] - left_mean);
            for(int id: right_idx) mse += (y[id] - right_mean) * (y[id] - right_mean);
            mse /= n;

            if(mse < best_mse) {
                best_mse = mse;
                best_feature = feature;
                best_threshold = threshold;
                best_left_idx = left_idx;
                best_right_idx = right_idx;
            }
        }
    }

    if(best_feature == -1) {
        double sum = accumulate(y.begin(), y.end(), 0.0);
        return new TreeNode(sum / y.size());
    }

    vector<vector<double>> left_X, right_X;
    vector<double> left_y, right_y;
    for(int id: best_left_idx) { left_X.push_back(X[id]); left_y.push_back(y[id]); }
    for(int id: best_right_idx) { right_X.push_back(X[id]); right_y.push_back(y[id]); }

    TreeNode* node = new TreeNode(best_feature, best_threshold);
    node->left = build_tree(left_X, left_y, depth-1);
    node->right = build_tree(right_X, right_y, depth-1);

    return node;
}

vector<double> xg_boost(const vector<vector<double>>& X_train, const vector<double>& y_train, const vector<vector<double>>& X_test, double learning_rate, int max_depth) {
    vector<double> preds_train(y_train.size(),0.0);
    double sum = accumulate(y_train.begin(), y_train.end(), 0.0);
    double mean_y = sum / y_train.size();

    vector<double> error(y_train.size());
    for(int i = 0; i < y_train.size(); i++)
        error[i] = y_train[i] - mean_y;

    int epochs = 100;
    vector<TreeNode*> trees;

    for(int i = 0; i < epochs; i++) {
        TreeNode* root = build_tree(X_train, error, max_depth);
        trees.push_back(root);

        for(int j = 0; j<error.size(); j++)
            preds_train[j] += learning_rate*predict(root, X_train[j]);

        for(int j = 0; j < error.size(); j++)
            error[j] = y_train[j] - (mean_y + preds_train[j]);
    }

    vector<double>preds_test;
    for(auto& x: X_test) {
        double pred = mean_y;
        for(auto* tree: trees)
            pred += learning_rate * predict(tree, x);
        preds_test.push_back(pred);
    }

    for(auto* tree: trees) free_tree(tree);
    return preds_test;
}

int main() {
    vector<vector<double>> X_train = {{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15},{16,17,18,19,20},{21,22,23,24,25}};
    vector<double> y_train = {11,12,13,14,15};

    vector<vector<double>> X_test = {{26,27,28,29,30},{31,32,33,34,35}};
    vector<double> y_pred = xg_boost(X_train, y_train, X_test, 0.1, 3);

    for(double p: y_pred) cout << p << " ";
    cout << endl;
    return 0;
}
