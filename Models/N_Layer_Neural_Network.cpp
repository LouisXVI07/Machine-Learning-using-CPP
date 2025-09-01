#include<bits/stdc++.h>
using namespace std;

struct LayerCache {
    vector<double> z;   
    vector<double> a;   
};

vector<vector<vector<double>>> initialise_weights(int number_of_layers, vector<int> number_of_neurons_per_layer) {
    vector<vector<vector<double>>> weights;
    for (int i = 0; i < number_of_layers - 1; i++) {
        vector<vector<double>> layer_weights;
        for (int j = 0; j < number_of_neurons_per_layer[i + 1]; j++) {
            vector<double> neuron_weights(number_of_neurons_per_layer[i], 0);
            for (int k = 0; k < number_of_neurons_per_layer[i]; k++) {
                neuron_weights[k] = ((double) rand() / RAND_MAX) - 0.5;  
            }
            layer_weights.push_back(neuron_weights);
        }
        weights.push_back(layer_weights);
    }
    return weights;
}

vector<vector<double>> initialise_bias(vector<int> number_of_neurons_per_layer,int number_of_layers) {
    vector<vector<double>> bias;
    for (int i = 0; i < number_of_layers - 1; i++) {
        vector<double> layer_bias(number_of_neurons_per_layer[i + 1], 0);
        for (int j = 0; j < number_of_neurons_per_layer[i + 1]; j++) {
            layer_bias[j] = ((double) rand() / RAND_MAX) - 0.5; 
        }
        bias.push_back(layer_bias);
    }
    return bias;
}

vector<LayerCache> forward_propagation(vector<double> X_train,vector<vector<vector<double>>> &weights,vector<vector<double>> &bias,vector<int> &number_of_neurons_per_layer, vector<string> &activation_function,int number_of_layers){

    vector<LayerCache> caches;
    vector<double> input = X_train;

    for (int i = 0; i < number_of_layers - 1; i++) {
        LayerCache cache;
        cache.z.resize(number_of_neurons_per_layer[i + 1]);
        cache.a.resize(number_of_neurons_per_layer[i + 1]);

        for (int j = 0; j < number_of_neurons_per_layer[i + 1]; j++) {
            cache.z[j] = bias[i][j];
            for (int k = 0; k < number_of_neurons_per_layer[i]; k++) {
                cache.z[j] += input[k] * weights[i][j][k];
            }
            if (activation_function[i] == "sigmoid") {
                cache.a[j] = 1.0 / (1.0 + exp(-cache.z[j]));
            } else if (activation_function[i] == "relu") {
                cache.a[j] = max(0.0, cache.z[j]);
            }
        }
        caches.push_back(cache);
        input = cache.a; 
    }
    return caches;
}

double mean_squared_error(vector<double> preds, vector<double> y_train) {
    if (preds.empty()) { cerr << "Error: empty preds\n"; exit(1); }
    double total = 0.0;
    for (size_t i = 0; i < preds.size(); i++) total += pow(preds[i] - y_train[i], 2);
    return total / preds.size();
}

vector<pair<vector<vector<double>>, vector<double>>> calculate_gradients(vector<LayerCache> &caches,vector<double> &y_train,vector<vector<vector<double>>> &weights,vector<string> &activation_function,vector<double> &X_train){

    int L = caches.size(); 
    vector<pair<vector<vector<double>>, vector<double>>> grads(L);
    int m = y_train.size();

    vector<double> dA(caches[L-1].a.size());
    for (int i = 0; i < dA.size(); i++) {
        dA[i] = (2.0 / m) * (caches[L-1].a[i] - y_train[i]); // scaled properly
    }

    for (int l = L-1; l >= 0; l--) {
        int n_out = caches[l].a.size();
        int n_in  = (l == 0 ? X_train.size() : caches[l-1].a.size());

        vector<double> dZ(n_out);
        for (int i = 0; i < n_out; i++) {
            if (activation_function[l] == "sigmoid")
                dZ[i] = dA[i] * (caches[l].a[i] * (1 - caches[l].a[i]));
            else if (activation_function[l] == "relu")
                dZ[i] = dA[i] * (caches[l].z[i] > 0 ? 1.0 : 0.0);
        }

        vector<vector<double>> dW(n_out, vector<double>(n_in, 0.0));
        vector<double> dB(n_out, 0.0);

        for (int i = 0; i < n_out; i++) {
            for (int j = 0; j < n_in; j++) {
                double a_prev = (l == 0 ? X_train[j] : caches[l-1].a[j]);
                dW[i][j] = dZ[i] * a_prev;
            }
            dB[i] = dZ[i];
        }

        grads[l] = {dW, dB};

        if (l > 0) {
            vector<double> dA_prev(caches[l-1].a.size(), 0.0);
            for (int j = 0; j < caches[l-1].a.size(); j++) {
                for (int i = 0; i < n_out; i++) {
                    dA_prev[j] += weights[l][i][j] * dZ[i];
                }
            }
            dA = dA_prev;
        }
    }
    return grads;
}

void backward_propagation(vector<vector<vector<double>>> &weights,vector<vector<double>> &bias,vector<pair<vector<vector<double>>, vector<double>>> &grads,double learning_rate){
    for (int l = 0; l < weights.size(); l++) {
        for (int i = 0; i < weights[l].size(); i++) {
            for (int j = 0; j < weights[l][i].size(); j++) {
                weights[l][i][j] -= learning_rate * grads[l].first[i][j];
            }
            bias[l][i] -= learning_rate * grads[l].second[i];
        }
    }
}

vector<double> neural_network_predict(
    vector<vector<double>> &X_train,
    vector<vector<double>> &y_train,
    vector<vector<double>> &X_test,
    int number_of_layers,
    vector<int> number_of_neurons_per_layer,
    vector<string> activation_function,
    double learning_rate,
    int epochs){
    vector<vector<vector<double>>> weights = initialise_weights(number_of_layers, number_of_neurons_per_layer);
    vector<vector<double>> bias = initialise_bias(number_of_neurons_per_layer, number_of_layers);

    for (int i = 0; i < epochs; i++) {
        double total_loss_value = 0.0;

        for (int s = 0; s < X_train.size(); s++) {
            vector<LayerCache> preds = forward_propagation(X_train[s], weights, bias, number_of_neurons_per_layer, activation_function, number_of_layers);
            
            if (preds.back().a.size() != y_train[s].size()) {
                cout << "Error: y_train[" << s << "] size (" << y_train[s].size()
                     << ") does not match output layer size (" << preds.back().a.size() << ")" << endl;
                return {};
            }

            total_loss_value += mean_squared_error(preds.back().a, y_train[s]);

            vector<pair<vector<vector<double>>, vector<double>>> gradients = 
                calculate_gradients(preds, y_train[s], weights, activation_function, X_train[s]);

            backward_propagation(weights, bias, gradients, learning_rate);
        }

        if (i % 10 == 0) {
            cout << "Epoch " << i << ", Loss: " << total_loss_value / X_train.size() << endl;
        }
    }

    // Predict for test set
    vector<double> y_pred;
    for (int s = 0; s < X_test.size(); s++) {
        vector<LayerCache> final_preds = forward_propagation(X_test[s], weights, bias, number_of_neurons_per_layer, activation_function, number_of_layers);
        y_pred.push_back(final_preds.back().a[0]); // assume single-output
    }
    return y_pred;
}






int main(){
    srand(time(0));
    vector<vector<double>> X_train;
    vector<vector<double>> y_train;

    for (double x = 0.0; x <= 20.0; x += 0.1) {
        X_train.push_back({x});
        y_train.push_back({sin(x) + 0.5 * cos(2*x)});
    }

    vector<vector<double>> X_test;
    for (double x = 20.1; x <= 25.0; x += 0.1) {
        X_test.push_back({x});
    }

    int number_of_layers = 3;
    vector<int> number_of_neurons_per_layer = {1, 16, 1};
    vector<string> activation_function = {"relu","sigmoid"};
    double learning_rate = 0.05;
    int epochs = 1000;

    if (activation_function.size() != number_of_layers - 1){
        cout << "Error: activation_function size must equal number_of_layers - 1" << endl;
        return 1;
    }
    if (X_train[0].size() != number_of_neurons_per_layer[0]) {
    cerr << "Error: Input size mismatch!" << endl;
    return 1;
    }
    if (y_train[0].size() != number_of_neurons_per_layer.back()) {
        cerr << "Error: Output size mismatch!" << endl;
        return 1;
    }

    vector<double> y_pred = neural_network_predict(X_train,y_train,X_test,number_of_layers,number_of_neurons_per_layer,activation_function,learning_rate,epochs);

    cout << "Prediction: " << y_pred[0] << endl;
}
