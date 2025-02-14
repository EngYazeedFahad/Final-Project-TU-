#Eng\Yazeed fahad alzaidi
#Eng\Mohammed Alhumaidi
#Eng\Nawaf bin shafi
#Eng\Abdulmohsen Alghamdi
#Eng\Waleed abdi
#Eng\Mohammed Altwirqi
#Project Advisor Prof. Httan Ali Asiri
#---------------------------------------------------------------------------------------------------------------------#
#KNN Model
def KNN_Grid_Search():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                                 roc_curve, auc, ConfusionMatrixDisplay)

    # Load dataset
    dataset = pd.read_csv("processed_data.csv")

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Split dataset
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

    # Define KNN model
    knn_model = KNeighborsClassifier()

    # Define hyperparameter grid
    param_grid = {
        'n_neighbors': [5, 7, 9, 11],
        'weights': ['uniform'],
        'metric': ['euclidean', 'manhattan']
    }

    # Grid Search for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=knn_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Best parameters and best model
    print("Best Parameters:", grid_search.best_params_)

    best_knn_model = grid_search.best_estimator_

    y_pred = best_knn_model.predict(X_test)

    # Accuracy scores
    train_accuracy = grid_search.score(X_train, y_train)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Confusion Matrix
    CM = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", CM)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    disp = ConfusionMatrixDisplay(confusion_matrix=CM)
    disp.plot(cmap=plt.cm.tab20c)
    plt.title("Confusion Matrix - KNN")
    plt.show()

    # ROC Curve
    unique_labels = np.unique(y_test)
    if len(unique_labels) == 2:
        y_prob = best_knn_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        print("Test AUC:", roc_auc)

        plt.figure()
        plt.plot(fpr, tpr, color='blue', label="ROC curve (AUC = %0.2f)" % roc_auc)
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic - KNN")
        plt.legend(loc="lower right")
        plt.show()

    # Class distribution
    train_class_counts = np.bincount(y_train)
    test_class_counts = np.bincount(y_test)
    classes = np.arange(len(train_class_counts))

    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    x = np.arange(len(classes))

    plt.bar(x - bar_width / 2, train_class_counts, width=bar_width, label='Train Samples', alpha=0.7, color='navy')
    plt.bar(x + bar_width / 2, test_class_counts, width=bar_width, label='Test Samples', alpha=0.7, color='dodgerblue')

    plt.xlabel('Class Labels')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Train and Test Sets - KNN')
    plt.xticks(classes)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Overfitting/Underfitting check
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
KNN_Grid_Search()



