#Eng\Yazeed fahad alzaidi
#Eng\Mohammed Alhumaidi
#Eng\Nawaf bin shafi
#Eng\Abdulmohsen Alghamdi
#Eng\Waleed abdi
#Eng\Mohammed Altwirqi
#Project Advisor Prof. Httan Ali Asiri
#------------------------------------------------------------------------------------------------------#
#Gradient Boosting
#Initial model
def GD_Random_Search():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.ensemble import GradientBoostingClassifier
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
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [10, 20, 30, 50],
        "min_samples_leaf": [10, 20, 30, 50],
        "subsample": [0.6, 0.7, 0.8]
    }

    # Initialize the Gradient Boosting model
    gb = GradientBoostingClassifier(random_state=44)

    # Apply Random Search with 100 iterations
    random_search = RandomizedSearchCV(
        estimator=gb, param_distributions=param_dist,
        n_iter=100, cv=5, scoring='accuracy', n_jobs=-1, random_state=44
    )
    random_search.fit(X_train, y_train)

    # Get the best model
    best_model = random_search.best_estimator_
    print("Best Hyperparameters:", random_search.best_params_)

    # Evaluate the best model
    train_accuracy = best_model.score(X_train, y_train)
    test_accuracy = best_model.score(X_test, y_test)
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Make predictions
    y_pred = best_model.predict(X_test)
    print('Predicted Values for Gradient Boosting (first 5):', y_pred[:5])

    # Confusion Matrix
    CM = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:\n', CM)

    disp = ConfusionMatrixDisplay(confusion_matrix=CM)
    disp.plot(cmap="OrRd")
    plt.title("Confusion Matrix - Gradient Boosting")
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
        plt.title("Receiver Operating Characteristic - Gradient Boosting")
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


#------------------------------------------------------------#
#Final Model
def GD_Final():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
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

    # Define the Gradient Boosting model with manually set hyperparameters
    best_model = GradientBoostingClassifier(
        subsample=0.8,
        n_estimators=300,
        min_samples_split=50,
        min_samples_leaf=30,
        max_depth=10,
        learning_rate=0.05,
        random_state=44
    )

    # Train the model
    best_model.fit(X_train, y_train)

    # Evaluate the model
    train_accuracy = best_model.score(X_train, y_train)
    test_accuracy = best_model.score(X_test, y_test)
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Make predictions
    y_pred = best_model.predict(X_test)
    print('Predicted Values for Gradient Boosting (first 5):', y_pred[:5])

    # Confusion Matrix
    CM = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:\n', CM)

    disp = ConfusionMatrixDisplay(confusion_matrix=CM)
    disp.plot(cmap="magma")
    plt.title("Confusion Matrix - Gradient Boosting")
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
        plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % roc_auc,
                 color='darkblue')  # Changed to 'darkblue' for GD theme
        plt.plot([0, 1], [0, 1], "g--")  # Changed diagonal reference line to green
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic - Gradient Boosting")
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

GD_Final()