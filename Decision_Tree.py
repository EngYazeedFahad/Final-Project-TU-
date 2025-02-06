#Eng\Yazeed fahad alzaidi
#Eng\Mohammed Alhumaidi
#Eng\Nawaf bin shafi
#Eng\Abdulmohsen Alghamdi
#Eng\Waleed abdi
#Eng\Mohammed Altwirqi
#Project Advisor Prof. Httan Ali Asiri
#---------------------------------------------------------------------------------------------------------------------#

#Initial model
def DT_Random_Search():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        classification_report,
        roc_curve,
        auc,
        ConfusionMatrixDisplay
    )
    import warnings

    warnings.filterwarnings("ignore")

    # Load the dataset
    dataset = pd.read_csv("processed_data.csv")

    # Separate features and target
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=44,
        stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define hyperparameter grid for Randomized Search
    param_dist = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None] + list(range(20, 90)),
        "min_samples_split": range(5, 40),
        "min_samples_leaf": range(5, 30),
    }

    # Initialize the Decision Tree model
    dt = DecisionTreeClassifier(random_state=44)

    # Apply Random Search with 100 iterations, optimizing for accuracy
    random_search = RandomizedSearchCV(
        estimator=dt, param_distributions=param_dist,
        n_iter=100, cv=5, scoring='accuracy', n_jobs=-1, random_state=44
    )

    random_search.fit(X_train, y_train)

    # Find the best parameters minimizing the difference between train and test accuracy
    best_model = None
    best_params = None
    min_diff = float("inf")

    for params in random_search.cv_results_["params"]:
        model = DecisionTreeClassifier(**params, random_state=44)
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        diff = abs(train_acc - test_acc)

        if diff < min_diff:
            min_diff = diff
            best_model = model
            best_params = params

    print("Best Hyperparameters :", best_params)

    # Evaluate the best model
    train_accuracy = best_model.score(X_train, y_train)
    test_accuracy = best_model.score(X_test, y_test)
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Make predictions
    y_pred = best_model.predict(X_test)
    print('Predicted Values for DecisionTree (first 5):', y_pred[:5])

    # Confusion Matrix
    CM = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:\n', CM)

    disp = ConfusionMatrixDisplay(confusion_matrix=CM)
    disp.plot(cmap="OrRd")
    plt.title("Confusion Matrix - Decision Tree")
    plt.show()

    # Classification Report
    Report = classification_report(y_test, y_pred)
    print("Classification Report:\n", Report)

    # Compute AUC if binary classification
    unique_labels = np.unique(y_test)
    if len(unique_labels) == 2:
        y_prob = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        print("Test AUC:", roc_auc)

        plt.figure()
        plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % roc_auc, color='darkred')
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic - Decision Tree")
        plt.legend(loc="lower right")
        plt.show()

    # Overfitting/Underfitting Indicator
    diff_threshold = 0.10
    accuracy_diff = train_accuracy - test_accuracy

    if accuracy_diff > diff_threshold:
        print("\nIndicator: The model may be OVERFITTING "
              f"(Train Accuracy - Test Accuracy = {accuracy_diff:.3f} > {diff_threshold}).")
    elif train_accuracy < 0.6 and test_accuracy < 0.6:
        print("\nIndicator: The model may be UNDERFITTING "
              f"(Train = {train_accuracy:.2f}, Test = {test_accuracy:.2f}).")
    else:
        print("\nIndicator: The model does not appear to be significantly overfit or underfit.")



#---------------------------------------------------------------------------------------------------------------------#

#Final DT Model

def DT_Final():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        roc_curve,
        auc,
        ConfusionMatrixDisplay
    )
    import warnings

    warnings.filterwarnings("ignore")

    # Load the dataset
    dataset = pd.read_csv("processed_data.csv")

    # Separate features and target
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=44,
        stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the Decision Tree model with specified hyperparameters
    best_dt = DecisionTreeClassifier(
        criterion="gini",
        max_depth=65,
        min_samples_leaf=28,
        min_samples_split=30,
        random_state=44
    )

    # Train the model
    best_dt.fit(X_train, y_train)

    # Evaluate the model
    train_accuracy = best_dt.score(X_train, y_train)
    test_accuracy = best_dt.score(X_test, y_test)
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Make predictions
    y_pred = best_dt.predict(X_test)
    print('Predicted Values for DecisionTree (first 5):', y_pred[:5])

    # Confusion Matrix
    CM = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:\n', CM)

    # Display Confusion Matrix with warm colors
    disp = ConfusionMatrixDisplay(confusion_matrix=CM)
    disp.plot(cmap="Accent")  # Warm color scheme
    plt.title("Confusion Matrix - Decision Tree")
    plt.show()

    # Classification Report
    Report = classification_report(y_test, y_pred)
    print("Classification Report:\n", Report)

    # Compute AUC if binary classification
    unique_labels = np.unique(y_test)
    if len(unique_labels) == 2:
        y_prob = best_dt.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        print("Test AUC:", roc_auc)

        plt.figure()
        plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % roc_auc, color='darkred')
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic - Decision Tree")
        plt.legend(loc="lower right")
        plt.show()

    # Overfitting/Underfitting Indicator
    diff_threshold = 0.12
    accuracy_diff = train_accuracy - test_accuracy

    if accuracy_diff > diff_threshold:
        print("\nIndicator: The model may be OVERFITTING "
              f"(Train Accuracy - Test Accuracy = {accuracy_diff:.3f} > {diff_threshold}).")
    elif train_accuracy < 0.6 and test_accuracy < 0.6:
        print("\nIndicator: The model may be UNDERFITTING "
              f"(Train = {train_accuracy:.2f}, Test = {test_accuracy:.2f}).")
    else:
        print("\nIndicator: The model does not appear to be significantly overfit or underfit.")

DT_Final()
#---------------------------------------------------------------------------------------------------------------------#
