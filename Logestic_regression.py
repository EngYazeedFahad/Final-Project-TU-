#Eng\Yazeed fahad alzaidi
#Eng\Mohammed Alhumaidi
#Eng\Nawaf bin shafi
#Eng\Abdulmohsen Alghamdi
#Eng\Waleed abdi
#Eng\Mohammed Altwirqi
#Project Advisor Prof. Httan Ali Asiri
#---------------------------------------------------------------------------------------------------------------------#
#Logistic Regression Model

def Logistic_Regression():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, roc_curve, auc,
                                 ConfusionMatrixDisplay)

    dataset = pd.read_csv("processed_data.csv")

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=44,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    log_reg = LogisticRegression()

    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000, 10000, 100000]
    }

    grid_search = GridSearchCV(
        estimator=log_reg,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    print("Logistic Regression:")

    print("Best Parameters:", grid_search.best_params_)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    train_accuracy = grid_search.score(X_train, y_train)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy :", test_accuracy)

    CM = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", CM)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    ###############################################################################
    # Display the confusion matrix with proper labels
    class_labels = ["Without Complication", "With Complication"]

    disp = ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Reds)
    plt.title("Confusion Matrix - Logistic Regression")
    plt.show()

    ###############################################################################
    unique_labels = np.unique(y_test)
    if len(unique_labels) == 2:
        y_prob = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        print("Test AUC:", roc_auc)

        plt.figure()
        plt.plot(fpr, tpr, color='yellow', label="ROC curve (AUC = %0.2f)" % roc_auc)  # Changed color to yellow
        plt.plot([0, 1], [0, 1], "r--")  # diagonal line for random classifier
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic - Logistic Regression")
        plt.legend(loc="lower right")
        plt.show()

    ###############################################################################

    train_class_counts = np.bincount(y_train)
    test_class_counts = np.bincount(y_test)
    classes = np.arange(len(train_class_counts))

    # Plotting the class distributions
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    x = np.arange(len(classes))

    plt.bar(x - bar_width / 2, train_class_counts, width=bar_width, label='Train Samples', alpha=0.7)
    plt.bar(x + bar_width / 2, test_class_counts, width=bar_width, label='Test Samples', alpha=0.7)

    plt.xlabel('Class Labels')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Train and Test Sets')
    plt.xticks(classes)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    ###############################################################################

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
Logistic_Regression()
