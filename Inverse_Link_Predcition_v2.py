import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

def load_edges_list(filename):
    edges_list = pd.read_csv(filename)
    return edges_list

def load_summary_data(filename):
    summary_data = pd.read_csv(filename)
    # Extract last three normalized float values
    features = summary_data.iloc[:, -3:].values
    return features

def link_prediction(graph, features):
    try:
        # Number of nodes in the graph
        num_nodes = len(graph.nodes())
      # Number of features for each node
        num_features_per_node = features.shape[1]  # Assuming features is a 2D array: num_nodes x num_features

        # Correcting the input shape: since we concatenate features of two nodes, input shape is double the number of features per node
        input_shape = num_features_per_node * 2

        # Define the GCN model with the correct input shape
        inputs = layers.Input(shape=(input_shape,))  # Use the corrected input shape
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy')

        # Generate positive samples (existing edges) and negative samples (non-edges)
        edges = list(graph.edges(data=True))
        non_edges = list(nx.non_edges(graph))

        # Create node pair features for classification
        node_features = []
        labels = []
        print("Shape of features array:", features.shape)
        print("Data type of elements in features array:", features.dtype)
# Create a mapping from node labels to integer indices
        node_to_index = {node_label: i for i, node_label in enumerate(graph.nodes())}

        # Collect existing edges features
        node_features = []
        labels = []
        for u, v, data in edges:
            u_index = node_to_index[u]
            v_index = node_to_index[v]
            node_features.append(np.concatenate((features[u_index], features[v_index])))
            labels.append(1)  # 1 for existing edges

        for u, v in non_edges:
            u_index = node_to_index[u]
            v_index = node_to_index[v]
            node_features.append(np.concatenate((features[u_index], features[v_index])))
            labels.append(0)  # 0 for non-edges

        # Convert node_features to numpy array
        node_features = np.array(node_features)
        # Convert labels to numpy array
        labels = np.array(labels)





        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(node_features, labels, test_size=0.2, random_state=42)

        # Train the GCN model
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        # Predict probabilities for test data
        y_pred_proba = model.predict(X_test)

        # Evaluate performance using ROC AUC score
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print("ROC AUC Score:", roc_auc)

        # Predict probabilities for all node pairs in the graph
        all_pred_proba = model.predict(node_features)

        # Add probability scores to the graph as edge attributes
        for i, (u, v, data) in enumerate(edges):  # Only iterate over existing edges
            data['probability'] = all_pred_proba[i][0]

        return graph

    except Exception as e:
        print("Error during link prediction:", e)
        return None

def compute_gradient(loss_function, A, graph, threshold):
    epsilon = 1e-5  # Small perturbation for numerical differentiation
    loss_plus = loss_function(A + epsilon, graph, threshold)
    loss_minus = loss_function(A - epsilon, graph, threshold)
    gradient = (loss_plus - loss_minus) / (2 * epsilon)
    return gradient

def loss_function(A, graph, threshold):
    sparsified_graph = remove_edges_based_on_combined_weight(graph.copy(), A, threshold)
    return sparsified_graph.number_of_edges()

def remove_edges_based_on_combined_weight(graph, A, threshold):
    """
    Remove edges from the graph based on the combined weight.
    """
    try:
        edges_to_remove = []
        for u, v, data in graph.edges(data=True):
            if 'probability' in data and 'weight' in data:
                combined_weight = A * data['probability'] + (1 - A) * data['weight']
                if threshold[0] <= combined_weight <= threshold[1]:
                    edges_to_remove.append((u, v))

        # Remove edges with combined weight within the threshold range
        graph.remove_edges_from(edges_to_remove)
        return graph

    except Exception as e:
        print("Error during edge removal based on combined weight:", e)
        return None


def remove_edges_after_link_prediction(graph, threshold, learning_rate=0.01, num_iterations=100):
    try:
        best_A = 0.0
        best_num_edges = float('inf')
        A = 0.5  # Initialize A

        # Gradient descent optimization
        for _ in range(num_iterations):
            gradient = compute_gradient(loss_function, A, graph, threshold)
            A -= learning_rate * gradient

            # Use the optimized A value for sparsification
            sparsified_graph = remove_edges_based_on_combined_weight(graph.copy(), A, threshold)
            num_edges = sparsified_graph.number_of_edges()
            if num_edges < best_num_edges:
                best_num_edges = num_edges
                best_A = A

        print("Best A value (Gradient Descent):", best_A)
        return remove_edges_based_on_combined_weight(graph.copy(), best_A, threshold)

    except Exception as e:
        print("Error during edge removal:", e)
        return None

if __name__ == "__main__":
    try:
        # Load edges list and summary data
        edges_list_filename = 'edges_list_0.8_Full.csv'
        summary_data_filename = '1M1L3D_summary.csv'

        edges_list = load_edges_list(edges_list_filename)
        summary_data = load_summary_data(summary_data_filename)

        # Create a graph from edges list
        graph = nx.from_pandas_edgelist(edges_list, 'source', 'target')

        # Print number of edges before edge removal
        print("Number of edges before edge removal:", graph.number_of_edges())

        # Perform link prediction
        graph = link_prediction(graph, summary_data)

        if graph:
            # Determine threshold
            threshold = (0.1, 0.2)

            # Remove edges after link prediction using gradient descent
            sparsified_graph = remove_edges_after_link_prediction(graph, threshold)

            # Print number of edges after sparsification
            print("Number of edges after sparsification:", sparsified_graph.number_of_edges())

            # Save final edges list after removing edges
            final_edges_filename = 'final_edges_list.csv'
            nx.write_edgelist(sparsified_graph, final_edges_filename)

            # Save features dataset with only those remaining in the edges list
            remaining_nodes = sparsified_graph.nodes()
            remaining_features = summary_data[[node in remaining_nodes for node in graph.nodes()]]
            remaining_features.to_csv('remaining_features.csv', index=False)
            
            # Plot the final graph
            plt.figure(figsize=(10, 6))
            pos = nx.spring_layout(sparsified_graph)
            nx.draw(sparsified_graph, pos, with_labels=True, node_size=100, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')
            plt.title("Final Graph after removing edges with optimized A value")
            plt.show()
        else:
            print("Error occurred during link prediction.")
    except Exception as e:
        print("Error occurred during example usage:", e)
