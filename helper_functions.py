import numpy as np
import networkx as nx
from networkx.algorithms.similarity import graph_edit_distance
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind
from scipy.special import comb
from networkx.algorithms.approximation import max_clique
from numba import jit, njit #for speeding up slow, not vectorized, python3 computation
import random
import copy
import os
import math
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Activation
from keras.regularizers import l2
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, LeaveOneOut, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from scipy import interp
import scikitplot as skplt
from sklearn.externals.joblib import parallel_backend
from itertools import combinations
import networkx as nx


"""Creates a numpy matrix for a given path to type of subject 
(in this notebook, it's ordered by condition eg MCI, AD, Control)
Note this assumes the data is already processed in the filename format our data is in.
check out the link to the data in the README

If using custom data, we recommend writing a version of this for your own data files, just needing to process
the data into a matrix of (# of people x # timesteps + 1 (to encode an array of 1's of converting, an array of 0's if control) x Features (num electrodes * number of bands))

This function also returns the order in which files are seen

"""
def make_subjects_matrix(num_participants, timesteps, electrode_count, num_bands, path, raw = True):
    count = 0
    subject = 0
    #make 3d array of size number of people x timesteps (+ 1 to encode converter/control) x Features (number of electrodes  x numbands)
    people = np.zeros((num_participants, timesteps + 1, electrode_count * num_bands)) 
    numbers_matrix = []
    order = []
    for filename in sorted(os.listdir(os.path.join(os.getcwd(), path))):
        if "DS_Store" in filename or "Icon" in filename:
            continue
        if raw:
            try:
                electrode_num = int(filename.split("_")[3].split(".")[0])
                if electrode_num > 19 or electrode_num == 1 or electrode_num == 2 or electrode_num == 16 or electrode_num == 17:
                    continue
            except:
                electrode_num = None
        f = open(path + "/" + filename, 'r')
        order.append(filename)
        lines = f.readlines()
        f.close()
        for i in range(len(lines)):
            if raw:
                if i == 0:
                    numbers = np.array(lines[i].split("\t")).astype(float)[250:1000]
                else:
                    numbers += np.array(lines[i].split("\t")).astype(float)[250:1000]
            else:
                if i == 0:
                    numbers = np.array(lines[i].split("\t")).astype(float)
                else:
                    numbers += np.array(lines[i].split("\t")).astype(float)
        numbers = np.true_divide(numbers, i + 1)
        numbers = np.append(numbers, 0) if 'control' in filename else np.append(numbers, 1)
        if len(numbers_matrix) == 0:
            numbers_matrix = numbers.reshape((timesteps + 1, 1))
        else:
            numbers_matrix = np.hstack((numbers_matrix, numbers.reshape((timesteps + 1, 1))))
        if count % (electrode_count*num_bands) == electrode_count*num_bands - 1:
            people[subject] = numbers_matrix
            numbers_matrix = []
            subject += 1
        count += 1
    return people, order

#@jit(parallel=True)
#Averaged epochs by divide factor. We use this to average it to 80ms
def average_epochs(num_participants, people, divide_factor, timesteps, electrode_count, bands):
    new_people = np.zeros((num_participants, int(timesteps/divide_factor + 1), electrode_count * bands))
    for i in range(num_participants):
        for j in range(int(timesteps/divide_factor)):
            start = j * divide_factor
            end = (j+1) * divide_factor
            for k in range(electrode_count * bands):
                total = 0
                for z in range(start, end):
                    total += people[i][z][k]
                new_people[i][j][k] = total/divide_factor
        new_people[i][int(timesteps/divide_factor)] = people[i][timesteps]
    return new_people

#From: https://github.com/rgarcia-herrera/visibility_graph
#Creates visiblity graphs in networkx given a time series

def visibility_graph(series):

    g = nx.Graph()
    
    # convert list of magnitudes into list of tuples that hold the index
    tseries = []
    n = 0
    for magnitude in series:
        tseries.append( (n, magnitude ) )
        n += 1

    # contiguous time points always have visibility
    for n in range(0,len(tseries)-1):
        (ta, ya) = tseries[n]
        (tb, yb) = tseries[n+1]
        g.add_node(ta, mag=ya, label=ta)
        g.add_node(tb, mag=yb, label=tb)
        g.add_edge(ta, tb, weight=1)

    for a,b in combinations(tseries, 2):
        # two points, maybe connect
        (ta, ya) = a
        (tb, yb) = b

        connect = True
        
        # let's see all other points in the series
        for tc, yc in tseries:
            # other points, not a or b
            if tc != ta and tc != tb:
                # does c obstruct?
                if yc > yb + (ya - yb) * ( (tb - tc) / (tb - ta) ):
                    connect = False
                    
        if connect:
            g.add_edge(ta, tb, weight = 1)


    return g

# Creates the VGs given our participants matrix
def make_graphs(num_participants, new_people, timesteps, divide_factor, electrode_count, bands):
    graphs = [[] for x in range(num_participants)]
    for participant in range(num_participants):
        for electrode in range(electrode_count * bands):
            graphs[participant].append(visibility_graph(new_people[participant, 0:int(timesteps/divide_factor), electrode]))
    return graphs

signficance_level = 0.01 #p-value
random_seed = 100 #random seed for reproducibility


# TSP Approximation algorithm
#@jit(parallel=True)
def tsp_approx(G):
    T = nx.minimum_spanning_tree(G)
    dfs = nx.dfs_preorder_nodes(T, 0)
    node_list = []
    for item in dfs:
        node_list.append(item)
    node_list.append(0)
    path = [0]
    for i in range(len(node_list) - 1):
        path.pop()
        path += nx.dijkstra_path(G, node_list[i], node_list[i + 1])
    total_cost = len(path)
    return total_cost

# CCSS
def electrode_connectivity(graphs):
    num_connections = 0
    theta = 0.25
    for i in range(len(graphs)):
        graph1 = graphs[i]
        clustering_sequence_1 = list(nx.clustering(graph1).values())
        for j in range(i+1, len(graphs)):
            graph2 = graphs[j]
            clustering_sequence_2 = list(nx.clustering(graph2).values())
            CCSS = abs(np.cov(clustering_sequence_1, clustering_sequence_2, bias=True)[0][1]/(np.std(clustering_sequence_1) * np.std(clustering_sequence_2)))
            if CCSS >= theta: num_connections += 1
    return num_connections

#CCSS for testing subjects with multiple bands (instead of just a raw signal)
def band_electrode_connectivity(graphs):
    theta = 0.25
    alpha_connections = 0
    beta_connections = 0
    delta_connections = 0
    gamma_connections = 0
    theta_connections = 0
    for i in range(len(graphs)):
        graph1 = graphs[i]
        clustering_sequence_1 = list(nx.clustering(graph1).values())
        for j in range(i+5, len(graphs), 5):
            graph2 = graphs[j]
            clustering_sequence_2 = list(nx.clustering(graph2).values())
            CCSS = abs(np.cov(clustering_sequence_1, clustering_sequence_2, bias=True)[0][1]/(np.std(clustering_sequence_1) * np.std(clustering_sequence_2)))
            if i % 5 == 0:
                if CCSS >= theta: alpha_connections += 1
            elif i % 5 == 1:
                if CCSS >= theta: beta_connections += 1
            elif i % 5 == 2:
                if CCSS >= theta: delta_connections += 1
            elif i % 5 == 3:
                if CCSS >= theta: gamma_connections += 1
            else:
                if CCSS >= theta: theta_connections += 1
    return [alpha_connections, beta_connections, delta_connections, gamma_connections, theta_connections]

#Testing small worldness
#@jit(parallel=True)
def small_world(G):
    num_graphs = 10
    counter = 0
    V = nx.number_of_nodes(G)
    E = nx.number_of_edges(G)
    C_rand = 0
    L_rand = 0
    random_graphs_L = np.zeros(num_graphs)
    while counter < num_graphs:
        erdos_renyi_graph = nx.gnm_random_graph(V, E, random_seed)
        if not nx.is_connected(erdos_renyi_graph):
            c, l, num_subgraphs = 0, 0, 0
            for connected_component in nx.connected_component_subgraphs(erdos_renyi_graph):
                num_subgraphs += 1
                c += nx.average_clustering(connected_component)
                l += nx.average_shortest_path_length(connected_component)
            c /= num_subgraphs
            l /= num_subgraphs
        else:
            c, l = nx.average_clustering(erdos_renyi_graph), nx.average_shortest_path_length(erdos_renyi_graph)
        C_rand += c
        L_rand += l
        counter += 1
    C_g, L_g = nx.average_clustering(G), nx.average_shortest_path_length(G)
    C_rand /= num_graphs
    L_rand /= num_graphs
    return (C_g/C_rand)/(L_g/L_rand)

# Graph Index Complexity
#@jit(parallel=True)
def GIC_func(G):
    largest_eigenvalue = float(max(nx.adjacency_spectrum(G, weight=None)))
    n = nx.number_of_nodes(G)
    c = (largest_eigenvalue - 2*math.cos(math.pi/(n+1)))/(n - 1 - 2*math.cos(math.pi/(n+1)))
    return 4*c*(1-c)

# Performing the actual hypothesis testing
def test_everything(controls, converts, electrode_count, num_bands, raw=True, difference=False):
    features = []
    band_list = []
    elecs = []
    formatted = []
    f = lambda func, x, y : [list(map(func, x)), list(map(func, y))]
    #measure connectivity
    if raw:
        control_conn = []
        convert_conn = []
        for i in range(len(controls)):
            control_conn.append(electrode_connectivity(controls[i]))
        for i in range(len(converts)):
            convert_conn.append(electrode_connectivity(converts[i]))
        stat, p = ttest_ind(control_conn, convert_conn)
        if p < signficance_level:
            features.append("electrode_connectivity")
            elecs.append(float("inf"))
            control_mean = np.mean(control_conn)
            MCI_mean = np.mean(convert_conn)
            direction = 'increase' if MCI_mean > control_mean else 'decrease'
            print_string = ("electrode_connectivity" + " " + direction, p)
            print(print_string)
    else:
        control_alpha_conn, control_beta_conn, control_delta_conn, control_gamma_conn, control_theta_conn = [],[],[],[],[]
        convert_alpha_conn, convert_beta_conn, convert_delta_conn, convert_gamma_conn, convert_theta_conn = [],[],[],[],[]
        for i in range(len(controls)):
            result = band_electrode_connectivity(controls[i])
            control_alpha_conn.append(result[0])
            control_beta_conn.append(result[1])
            control_delta_conn.append(result[2])
            control_gamma_conn.append(result[3])
            control_theta_conn.append(result[4])
        for i in range(len(converts)):
            result = band_electrode_connectivity(converts[i])
            convert_alpha_conn.append(result[0])
            convert_beta_conn.append(result[1])
            convert_delta_conn.append(result[2])
            convert_gamma_conn.append(result[3])
            convert_theta_conn.append(result[4])
        p_array = [0 for i in range(5)]
        stat, p_array[0] = ttest_ind(control_alpha_conn, convert_alpha_conn)
        stat, p_array[1] = ttest_ind(control_beta_conn, convert_beta_conn)
        stat, p_array[2] = ttest_ind(control_delta_conn, convert_delta_conn)
        stat, p_array[3] = ttest_ind(control_gamma_conn, convert_gamma_conn)
        stat, p_array[4] = ttest_ind(control_theta_conn, convert_theta_conn)
        for i in range(len(p_array)):
            if p_array[i] < signficance_level:
                feature = "electrode_connectivity"
                if i % 5 == 0:
                    band_list.append(0)
                    band = 'alpha'
                    control_val = np.mean(control_alpha_conn)
                    convert_val = np.mean(convert_alpha_conn)
                elif i % 5 == 1:
                    band = 'beta'
                    band_list.append(1)
                    control_val = np.mean(control_beta_conn)
                    convert_val = np.mean(convert_beta_conn)
                elif i % 5 == 2:
                    band = 'delta'
                    band_list.append(2)
                    control_val = np.mean(control_delta_conn)
                    convert_val = np.mean(convert_delta_conn)
                elif i % 5 == 3:
                    band = 'gamma'
                    band_list.append(3)
                    control_val = np.mean(control_gamma_conn)
                    convert_val = np.mean(convert_gamma_conn)
                else:
                    band = 'theta'
                    band_list.append(4)
                    control_val = np.mean(control_theta_conn)
                    convert_val = np.mean(convert_theta_conn)
                features.append(feature)
                elecs.append(float("inf"))
                direction = 'increase' if convert_val > control_val else 'decrease'
                print_string = (feature + " " + "band: " + band + " " + direction, p_array[i])
                print(print_string)
    electrode_num = 0
    #everything else
    for i in range(electrode_count * num_bands):
        if raw:
            electrode_num = (electrode_num + 1) % electrode_count
        else:
            if i % num_bands == 0 and i != 0:
                electrode_num = (electrode_num + 1) % electrode_count
        control_vals = [participant[i] for participant in controls]
        converts_vals = [participant[i] for participant in converts]
        if not difference:
            electrodes = {0: 10, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6:18, 7:19, 8:3, 9:4, 10:5, 11:6, 12:7, 13:8, 14:9}
        else:
            electrodes = {0: 3, 1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6:9, 7:10, 8:11, 9:12, 10:13, 11:14, 12:15, 13:18, 14:19}
        bands = {0: 'alpha', 1: 'beta', 2: 'delta', 3: 'gamma', 4: 'theta'}
        tests = {'global_efficiency': f(nx.global_efficiency, control_vals, converts_vals),
                 'clustering_coeff': f(nx.average_clustering, control_vals, converts_vals),
                 'max_clique' : f(lambda x: len(max_clique(x)), control_vals, converts_vals),
                 'TSP' : f(tsp_approx, control_vals, converts_vals),
                 'density': f(nx.density, control_vals, converts_vals),
                 'local_efficiency' : f(nx.local_efficiency, control_vals, converts_vals),
                 'independent_set' : f(lambda x: len(nx.algorithms.approximation.maximum_independent_set(x)), control_vals, converts_vals),
                 'GIC': f(GIC_func, control_vals, converts_vals),
                 'small_worldness' : f(small_world, control_vals, converts_vals),
                 'max_flow' : f(lambda x: nx.maximum_flow_value(x, 0, len(x) - 1, capacity="weight"), control_vals, converts_vals), #same as size of min cut
                 'greedy_coloring' :  f(lambda x: len(set(nx.greedy_color(x).values())), control_vals, converts_vals)
                }
        for k, v in tests.items():
            try:
                stat, p = ttest_ind(v[0], v[1])
            except:
                continue
            if p < signficance_level:
                features.append(k)
                elecs.append(electrode_num)
                control_mean = np.mean(v[0])
                MCI_mean = np.mean(v[1])
                direction = 'increase' if MCI_mean > control_mean else 'decrease'
                if raw:
                    print_string = ("electrode: " + str(electrodes[electrode_num]) + " " + k + " " + direction, p)
                    print(print_string)
                else:
                    band_list.append(i % num_bands)
                    print_string = ("electrode: " + str(electrodes[electrode_num]) + " band: " + bands[i % num_bands] + " " + k + " " + direction, p)
                    print(print_string)
                formatted.append(print_string)
    return features, elecs, band_list, formatted

#convenience function for recalculating electrode connectivity feature
def electrode_connectivity_band(graph, band):
    distances = []
    start = band
    for i in range(start, len(graph)):
        graph1 = graph[i]
        degree_sequence1 = np.array(sorted([d + 1 for n, d in graph1.degree()], reverse=True))
        for j in range(i+5, len(graph), 5):
            graph2 = graph[j]
            degree_sequence2 = np.array(sorted([d + 1 for n, d in graph2.degree()], reverse=True))
            distances.append(np.linalg.norm(degree_sequence1 - degree_sequence2, ord=1))
    return np.average(distances)

#Readys inputs for ML, given list of features
#This is definitely redundant and all computation here could have been done in original hypothesis testing function
def make_ML_input(features, electrodes, bands, graphs, people, num_in_train, num_bands, raw = True):
    i = 0
    mapping = {'global_efficiency': nx.global_efficiency,
                'clustering_coeff': nx.average_clustering,
                'max_clique': lambda x: len(max_clique(x)),
                'TSP': tsp_approx,
                'small_worldness': small_world,
                'GIC': GIC_func,
                'density': nx.density,
                'independent_set' : lambda x: len(nx.algorithms.approximation.maximum_independent_set(x)),
                'local_efficiency' : nx.local_efficiency,
                'max_flow': lambda x: nx.maximum_flow_value(x, 0, len(x) - 1, capacity="weight"),
                'greedy_coloring' : lambda x: len(set(nx.greedy_color(x).values()))
              }
    f = lambda index, number: np.array([mapping[features[index]](participant[electrodes[index]]) for participant in graphs[:number]]).reshape(number, 1)
    band_f = lambda index, number, band: np.array([mapping[features[index]](participant[electrodes[index] * num_bands + band]) for participant in graphs[:number]]).reshape(number, 1)
    g = lambda number : np.array([electrode_connectivity(graphs[i]) for i in range(len(graphs[:number]))]).reshape(number, 1)
    band_g = lambda number, band: np.array([electrode_connectivity_band(graphs[i], band) for i in range(len(graphs[:number]))]).reshape(number, 1)
    i = 0
    X, x_test = None, None
    while i < len(features):
        if i == 0:
            if "connectivity" in features[i]:
                if raw:
                    X = g(num_in_train)
                    x_test = g(len(people) - num_in_train)
                else:
                    X = band_g(num_in_train, bands[i])
                    x_test = band_g(len(people) - num_in_train, bands[i])
            else:
                if raw:
                    X = f(i, num_in_train)
                    x_test = f(i, len(people) - num_in_train)
                else:
                    X = band_f(i, num_in_train, bands[i])
                    x_test = band_f(i, len(people) - num_in_train, bands[i])
        else:
            if "connectivity" in features[i]:
                if raw:
                    X = np.hstack((X, g(num_in_train)))
                    x_test = np.hstack((x_test, g(len(people) - num_in_train)))
                else:
                    X = np.hstack((X, band_g(num_in_train, bands[i])))
                    x_test = np.hstack((x_test, band_g(len(people) - num_in_train, bands[i])))
                    
            else:
                if raw:
                    X = np.hstack((X, f(i, num_in_train)))
                    x_test = np.hstack((x_test, f(i, len(people) - num_in_train)))
                else:
                    X = np.hstack((X, band_f(i, num_in_train, bands[i])))
                    x_test = np.hstack((x_test, band_f(i, len(people) - num_in_train, bands[i])))
        i += 1
    y = people[:num_in_train, people.shape[1] - 1, :]
    y = np.array([1 if np.all(row[0]) else 0 for row in y])
    y_test = people[num_in_train:len(people), people.shape[1] - 1, :]
    y_test = np.array([1 if np.all(row[0]) else 0 for row in y_test])
    return X, y, x_test, y_test

def create_neural_network(feature_reduction = 11):
    model = Sequential()
    model.add(Dense(feature_reduction, input_dim=feature_reduction, kernel_initializer='normal', activation='relu'))
    model.add(Dense(feature_reduction * 2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Performs n_splits number of random shuffling, standardizing, PCA, and ML
# More algorithms than listed in the paper are tested here for wider comparison
def run_ML(num_epochs, X, y, n_splits=5, feature_reduction=11):
    metrics = {
        'Neural Network': {'Class Prediction': [], 'Probabilities': [], 'Precisions': [], 'Recalls': [], 'AUCs': [], 'Accuracies': []},
        #'3-Nearest Neighbor': {'Class Prediction': [], 'Probabilities': [], 'Precisions': [], 'Recalls': [], 'AUCs': [], 'Accuracies': []},
        'SVM': {'Class Prediction': [], 'Probabilities': [], 'Precisions': [], 'Recalls': [], 'AUCs': [], 'Accuracies': []},
        'Logistic Regression': {'Class Prediction': [], 'Probabilities': [], 'Precisions': [], 'Recalls': [], 'AUCs': [], 'Accuracies': []},
        #'Random Forest': {'Class Prediction': [], 'Probabilities': [], 'Precisions': [], 'Recalls': [], 'AUCs': [], 'Accuracies': []},
        'LDA': {'Class Prediction': [], 'Probabilities': [], 'Precisions': [], 'Recalls': [], 'AUCs': [], 'Accuracies': []}
        }
    #sss = ShuffleSplit(n_splits=n_splits, test_size=0.15)
    #sss = LeaveOneOut()
    #sss = KFold(n_splits=n_splits)
    sss = StratifiedKFold(n_splits=n_splits, shuffle=True)
    counter = 1
    all_y_test = []
    for train_index, test_index in sss.split(X, y):
        if counter % 5 == 0: print("Split: %d out of %d" % (counter, n_splits))
        keras.backend.clear_session()
        models = {
        'Neural Network': create_neural_network(),
        #'3-Nearest Neighbor': KNeighborsClassifier(n_neighbors=3),
        'SVM': svm.SVC(probability=True, gamma='auto', kernel='linear'),
        'Logistic Regression': LogisticRegression(solver='liblinear'),
        #'Random Forest': RandomForestClassifier(n_estimators=100),
        'LDA': LinearDiscriminantAnalysis()
        }
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        pca = PCA(n_components=feature_reduction)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        all_y_test.extend(list(y_test))
        for name, model in models.items():
            if name == 'Neural Network':
                model.fit(X_train, y_train, batch_size=8, epochs=num_epochs, verbose=0, validation_split=0)
                y_score = model.predict_proba(X_test)
                metrics[name]['Probabilities'].append(y_score)
                y_preds = model.predict_classes(X_test).flatten()
                metrics[name]['Class Prediction'].append(y_preds)
                scores_for_auc = np.array(y_score)[..., 0].flatten()
            else:
                model.fit(X_train, y_train)
                y_score = model.predict_proba(X_test)
                y_preds = model.predict(X_test)
                metrics[name]['Probabilities'].append(y_score)
                metrics[name]['Class Prediction'].append(y_preds)
                scores_for_auc = np.array(y_score)[..., 1].flatten()
            try:
                auc = roc_auc_score(y_test, scores_for_auc)
                precision, recall, *_ = precision_recall_fscore_support(y_test, y_preds, average='binary')
                #if precision != 0:
                metrics[name]['Precisions'].append(precision)
                #if recall != 0:
                metrics[name]['Recalls'].append(recall)
                #if auc != 0:
                metrics[name]['AUCs'].append(auc)
            except:
                pass
            accuracy = accuracy_score(y_test, y_preds)
            metrics[name]['Accuracies'].append(accuracy)
        counter += 1
    return metrics, all_y_test


# Plots ROC curves
def plot_roc(model_name, y_labels, predictions):
    y_labels = np.array(y_labels).flatten()
    base_fpr = np.linspace(0, 1, 101)
    if model_name == 'Neural Network':
        predictions = np.array(predictions)[:, :, 0].flatten()
        fpr, tpr, _ = roc_curve(y_labels, predictions)
    else:
        predictions = np.array(predictions)[:, :, 1].flatten()
        fpr, tpr, _ = roc_curve(y_labels, predictions)
    print("AUC: %0.2f" % roc_auc_score(y_labels, predictions))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title(model_name)
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()

#Makes the subjects matrix for AD data, which is formatted slightly differently than before
def make_subjects_matrix_AD(num_participants, timesteps, electrode_count, num_bands, path, raw = True):
    count = 0
    subject = 0
    people = np.zeros((num_participants, timesteps + 1, electrode_count * num_bands)) #make 3d array of size number of people x timesteps (+ 1 to encode converter/control) x Features (number of electrodes  x numbands)
    numbers_matrix = []
    order = []
    for filename in sorted(os.listdir(os.path.join(os.getcwd(), path))):
        if "DS_Store" in filename or "Icon" in filename:
            continue
        if raw:
            try:
                electrode_num = int(filename.split("_")[3].split(".")[0])
                if electrode_num > 19 or electrode_num == 1 or electrode_num == 2 or electrode_num == 16 or electrode_num == 17:
                    continue
            except:
                electrode_num = None
        f = open(path + "/" + filename, 'r')
        order.append(filename)
        lines = f.readlines()
        f.close()
        for i in range(len(lines)):
            if raw:
                if i == 0:
                    numbers = np.array(lines[i].split("\t")).astype(float)
                else:
                    numbers += np.array(lines[i].split("\t")).astype(float)
            else:
                if i == 0:
                    numbers = np.array(lines[i].split("\t")).astype(float)
                else:
                    numbers += np.array(lines[i].split("\t")).astype(float)
        numbers = np.true_divide(numbers, i + 1)
        numbers = np.append(numbers, 1)
        if len(numbers_matrix) == 0:
            numbers_matrix = numbers.reshape((timesteps + 1, 1))
        else:
            numbers_matrix = np.hstack((numbers_matrix, numbers.reshape((timesteps + 1, 1))))
        if count % (electrode_count*num_bands) == electrode_count*num_bands - 1:
            people[subject] = numbers_matrix
            numbers_matrix = []
            subject += 1
        count += 1
    return people, order
