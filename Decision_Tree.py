#Eng\Yazeed fahad alzaidi
#Eng\Mohammed Alhumaidi
#Eng\Nawaf bin shafi
#Eng\Abdulmohsen Alghamdi
#Eng\Waleed abdi
#Eng\Mohammed Altwirqi
#Project Advisor Prof. Httan Ali Asiri
#---------------------------------------------------------------------------------------------------------------------#
#Initial model
def DT_Grid_Search():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV
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

    # Define hyperparameter grid
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [None] + list(range(1, 51)),
        "min_samples_split": range(2, 51),
        "min_samples_leaf": range(1, 11),
        "max_features": [None]
    }

    # Initialize the Decision Tree model
    dt = DecisionTreeClassifier(random_state=44)

    # Apply Grid Search for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=dt, param_grid=param_grid,
        cv=5, scoring='accuracy', n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # Best parameters from GridSearchCV
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    # Train the best model
    best_dt = DecisionTreeClassifier(**best_params, random_state=44)
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
    disp.plot(cmap="OrRd")  # Warm color scheme
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
        criterion="entropy",
        max_depth=14,
        min_samples_leaf=1,
        min_samples_split=4,
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

