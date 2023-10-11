import csv
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # make df and skip the headers
    df = pd.read_csv(filename, header=None, skiprows=1)
    
    # months map
    month_mapping = {
        'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5,
        'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11
    }
    
    # VisitorType and Weekend map
    visitor_type_mapping = {'Returning_Visitor': 1, 'New_Visitor': 0, 'Other': 0}
    weekend_mapping = {True: 1, False: 0}
    
    # for each of the first 17 columns, make sure all values fit desired types
    # and store each row as list in evidence
    evidence = df.iloc[:, :17].apply(lambda x: [int(x[0]), float(x[1]), int(x[2]), float(x[3]),
                                                 int(x[4]), float(x[5]), float(x[6]), float(x[7]), float(x[8]),
                                                 float(x[9]), month_mapping[x[10]], int(x[11]), int(x[12]),
                                                 int(x[13]), int(x[14]), visitor_type_mapping[x[15]], weekend_mapping[x[16]]],
                                     axis=1).values.tolist()
    
    # go through last column and mark as 1 if true, 0 otherwise
    labels = df.iloc[:, 17].apply(lambda x: 1 if x == True else 0).values.tolist()
    
    return (evidence,labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    knn_classifier = KNeighborsClassifier(n_neighbors=1)
    knn_classifier.fit(evidence, labels)
    
    return knn_classifier


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # set up our initial variables
    positives = 0
    correct_positives = 0
    negatives = 0
    correct_negatives = 0
    
    # go through each pair of actual label and predicted label, and see 
    # how many positives and negatives we have, and how many we accurately predicted
    for actual_val, pred_val in zip(labels, predictions):
        if actual_val == 1:
            positives += 1
            if pred_val == 1:
                correct_positives += 1
        else:
            negatives += 1
            if pred_val == 0:
                correct_negatives += 1
    
    # calculate our sensitivity and specificity values
    sensitivity = correct_positives / positives
    specificity = correct_negatives / negatives
    
    return (sensitivity, specificity)
            


if __name__ == "__main__":
    main()
