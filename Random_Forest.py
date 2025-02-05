#Eng\Yazeed fahad alzaidi
#Eng\Mohammed Alhumaidi
#Eng\Nawaf bin shafi
#Eng\Abdulmohsen Alghamdi
#Eng\Waleed abdi
#Eng\Mohammed Altwirqi
#Project Advisor Prof. Httan Ali Asiri
#------------------------------------------------------------------------------------------------------#
#Initial model
def RF_Random_Search():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                                 roc_curve, auc, ConfusionMatrixDisplay)


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

    # Initialize Random Forest model
    rf_model = RandomForestClassifier(random_state=44)

    # Define hyperparameter grid
    param_dist = {
         'criterion': ['gini', 'entropy'],
        'n_estimators': [100,250,300],
        'max_depth': [40,50,70,90],
        'min_samples_split': [28,29,30],
        'min_samples_leaf': [1, 2,3,4,5,6,7,8,9],
        'bootstrap': [True,False]
    }

    # Randomized search
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_dist,
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        verbose=1,
        n_iter=20,
        random_state=44
    )

    random_search.fit(X_train, y_train)

    # Best parameters and best model
    print("Best Parameters:", random_search.best_params_)


    best_model = random_search.best_estimator_

    y_pred = best_model.predict(X_test)

    # Accuracy scores
    train_accuracy = random_search.score(X_train, y_train)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy :", test_accuracy)

    # Confusion Matrix
    CM = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", CM)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    disp = ConfusionMatrixDisplay(confusion_matrix=CM)
    disp.plot(cmap=plt.cm.hot)
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve
    unique_labels = np.unique(y_test)
    if len(unique_labels) == 2:
        y_prob = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        print("Test AUC:", roc_auc)

        plt.figure()
        plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % roc_auc)
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.show()

    # Class distribution
    train_class_counts = np.bincount(y_train)
    test_class_counts = np.bincount(y_test)
    classes = np.arange(len(train_class_counts))

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



#------------------------------------------------------------------------------------------------------#
#Final RF Model
def RF_Final():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
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

    # Manually set hyperparameters
    # Best Parameters: {'n_estimators': 300, 'min_samples_split': 28, 'min_samples_leaf': 2, 'max_depth': 50, 'criterion': 'entropy', 'bootstrap': False}
    rf_model = RandomForestClassifier(
        criterion="entropy",
        n_estimators=250,  # Set desired number of trees
        max_depth=40,  # Set max depth of trees
        min_samples_split=29,  # Set min samples for split
        min_samples_leaf=1,  # Set min samples per leaf
        bootstrap=False,  # Enable or disable bootstrap
        random_state=44
    )

    # Train the model
    rf_model.fit(X_train, y_train)

    # Predictions
    y_pred = rf_model.predict(X_test)

    # Accuracy scores
    train_accuracy = rf_model.score(X_train, y_train)
    test_accuracy = accuracy_score(y_test, y_pred)

    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy :", test_accuracy)

    # Confusion Matrix
    CM = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix For Random Forest:\n", CM)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Change the color map to forest-like colors
    disp = ConfusionMatrixDisplay(confusion_matrix=CM)
    disp.plot(cmap=plt.cm.terrain)  # Forest-like color palette
    plt.title("Confusion Matrix For Random Forest")
    plt.show()

    # ROC Curve
    unique_labels = np.unique(y_test)
    if len(unique_labels) == 2:
        y_prob = rf_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        print("Test AUC:", roc_auc)

        plt.figure()
        plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % roc_auc, color='#228B22')  # Forest green color
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.show()

    # Class distribution
    train_class_counts = np.bincount(y_train)
    test_class_counts = np.bincount(y_test)
    classes = np.arange(len(train_class_counts))

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

RF_Final()
