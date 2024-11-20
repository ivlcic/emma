import logging
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.semi_supervised import LabelPropagation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve
from argparse import ArgumentParser

from ..core.args import CommonArguments

logger = logging.getLogger('tests.ir_metrics')


def add_args(module_name: str, parser: ArgumentParser) -> None:
    CommonArguments.raw_data_dir(module_name, parser, ('-o', '--data_in_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-i', '--data_out_dir'))


def label_assign_roc(args) -> int:
    # Simulated probabilities for 10 sentences and 7 labels
    # Each value represents the predicted probability of a label being present in a sentence
    sentence_probs = np.random.rand(10, 7)

    # Define true binary labels for each sentence (for evaluation purposes)
    # In practice, these would come from your ground truth data
    true_labels = np.random.randint(0, 2, size=(10, 7))

    # Labels for reference
    labels = ["Animals", "Nature", "Space", "Sports", "Food", "Transport", "Weather"]

    # Function to calculate optimal threshold using ROC curve
    def find_optimal_threshold(y_true, y_prob):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        youden_j = tpr - fpr  # Youden's J statistic
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        return optimal_threshold

    # Find optimal thresholds for each label using ROC curve analysis
    optimal_thresholds = []
    for i in range(len(labels)):
        optimal_threshold = find_optimal_threshold(true_labels[:, i], sentence_probs[:, i])
        optimal_thresholds.append(optimal_threshold)

    # Apply the thresholds to the predicted probabilities to get binary predictions
    binary_predictions = (sentence_probs >= np.array(optimal_thresholds)).astype(int)

    # Aggregate predictions at the document level by counting how many sentences predict each label
    label_counts = binary_predictions.sum(axis=0)

    # Define a document-level threshold (e.g., at least 30% of sentences must predict a label)
    sentence_threshold = int(0.3 * len(sentence_probs))  # 30% of sentences

    # Assign labels to the document based on the count of sentence-level predictions
    document_labels = {labels[i]: label_counts[i] >= sentence_threshold for i in range(len(labels))}

    # Display results
    print("Optimal Thresholds for Each Label:", optimal_thresholds)
    print("Binary Sentence-Level Predictions:\n", binary_predictions)
    print("Label Counts Across Sentences:", label_counts)
    print("Final Document-Level Labels:", document_labels)
    return 0


def label_assign_graph(args):
    # Simulated sentences for an article
    sentences = [
        "The cat sits on the mat.",
        "Dogs are great companions.",
        "The sun rises in the east.",
        "Birds can fly very high.",
        "Fish swim in water.",
        "Horses are fast runners.",
        "There are many stars in the sky.",
        "Trees provide shade and oxygen.",
        "Rabbits can hop quickly.",
        "Insects are vital for the ecosystem."
    ]

    # Labels for reference (7 possible labels)
    labels = ["Animals", "Nature", "Space", "Sports", "Food", "Transport", "Weather"]

    # Simulated ground truth for some sentences (for semi-supervised learning)
    # -1 means that the label is unknown and needs to be propagated
    # This is a multi-label problem, so we will handle each label separately
    initial_labels = np.array([
        [1, 0, 0, 0, 0, 0, 0],  # Sentence 1: Animals
        [1, 0, 0, 0, 0, 0, 0],  # Sentence 2: Animals
        [-1, -1, -1, -1, -1, -1, -1],  # Sentence 3: Unknown
        [1, 0, 0, 0, 0, 0, 0],  # Sentence 4: Animals
        [-1, -1, -1, -1, -1, -1, -1],  # Sentence 5: Unknown
        [1, 0, 0, 0, 0, 0, 0],  # Sentence 6: Animals
        [-1, -1, -1, -1, -1, -1, -1],  # Sentence 7: Unknown
        [0, 1, 0, 0, 0, 0, 0],  # Sentence 8: Nature
        [-1, -1, -1, -1, -1, -1, -1],  # Sentence9 : Unknown
        [0, 1, 0, 0, 0, 0, 0]  # Sentence10 : Nature
    ])

    # Step 2: Vectorize the sentences using TF-IDF to get sentence embeddings
    vectorizer = TfidfVectorizer()
    sentence_embeddings = vectorizer.fit_transform(sentences).toarray()

    # Step 3: Compute pairwise cosine similarity between sentences to construct a graph
    similarity_matrix = cosine_similarity(sentence_embeddings)

    # Step 4: Apply Label Propagation for each label separately
    propagated_labels = np.copy(initial_labels)

    for i in range(len(labels)):
        # Use only one label at a time for propagation (binary classification per label)
        y = initial_labels[:, i]

        # Only propagate if there are unlabeled data points (-1)
        if np.any(y == -1):
            label_propagation_model = LabelPropagation(kernel='rbf', gamma=20)
            label_propagation_model.fit(similarity_matrix, y)

            # Update propagated labels for this specific label
            propagated_labels[:, i] = label_propagation_model.transduction_

    # Step 5: Aggregate sentence-level labels to assign document-level labels
    # We can use majority voting or any other aggregation method

    document_labels = {labels[i]: int(np.sum(propagated_labels[:, i]) > len(sentences) * 0.3) for i in
                       range(len(labels))}

    # Display results
    print("Propagated Labels for Each Sentence:\n", propagated_labels)
    print("\nFinal Document-Level Labels:\n", document_labels)
    return 0
