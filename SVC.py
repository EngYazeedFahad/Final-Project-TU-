#Eng\Yazeed fahad alzaidi
#Eng\Mohammed Alhumaidi
#Eng\Nawaf bin shafi
#Eng\Abdulmohsen Alghamdi
#Eng\Waleed abdi
#Eng\Mohammed Altwirqi
#Project Advisor Prof. Httan Ali Asiri
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
#Support Vector Classifier

#This is the initial code to find the best hyperparameter
def initial():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
    )
    import warnings
    from sklearn.exceptions import ConvergenceWarning

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

    svc = SVC(max_iter=-1)

    param_grid = [
        {
            'kernel': ['linear'],
            'C': [0.01, 0.1, 1, 10, 100],
            'gamma': ['scale', 'auto']
        },
        {
            'kernel': ['rbf'],
            'C': [0.01, 0.1, 1, 10, 100],
            'gamma': ['scale', 'auto']
        },
        {
            'kernel': ['poly'],
            'C': [0.01, 0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
            'degree': [2, 3, 4]
        }
    ]

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=svc,
        param_grid=param_grid,
        scoring='accuracy',  # You can choose other metrics like 'f1', 'roc_auc', etc.
        cv=5,  # 5-fold cross-validation
        n_jobs=-1,  # Use all available cores
        verbose=2  # Verbosity level
    )

    # Perform grid search on the training data
    grid_search.fit(X_train, y_train)

    # Display the best parameters and best score
    print("SVC:")
    print("Best Parameters:", grid_search.best_params_)

    # Retrieve the best estimator
    best_svc = grid_search.best_estimator_

    # Evaluate the best model on the training and testing sets
    train_score = best_svc.score(X_train, y_train)
    test_score = best_svc.score(X_test, y_test)

    print("Train Score with Best Parameters:", train_score)
    print("Test Score with Best Parameters:", test_score)

    # Make predictions on the test set
    y_pred = best_svc.predict(X_test)

    # Confusion Matrix
    CM = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:\n', CM)

    # Define class labels
    labels = ["Without Complication", "With Complication"]

    # Plot the confusion matrix using seaborn
    sns.heatmap(
        CM, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - SVC')
    plt.show()

    # Classification Report
    Report = classification_report(
        y_test, y_pred,
        target_names=labels
    )
    print("Classification Report:\n", Report)

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#



#This is the second code to find the best hyperparameter for RBF kernel
def Gamma_And_C_parameter():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
    )
    from sklearn.exceptions import ConvergenceWarning
    import warnings

    # Suppress warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Load the dataset
    dataset = pd.read_csv("processed_data.csv")

    # Separate features and target
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=44, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the parameter grid
    param_grid = {
        'gamma': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Gamma values to test
        'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Optional: Test different C values
    }

    # Initialize the SVC model
    svc = SVC(kernel='rbf', random_state=44)

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=svc,
        param_grid=param_grid,
        scoring='accuracy',  # Optimize for accuracy
        cv=5,  # 5-fold cross-validation
        verbose=1,  # To see progress
        n_jobs=-1  # Use all processors
    )

    # Fit GridSearchCV to the data
    grid_search.fit(X_train, y_train)

    # Extract the best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Evaluate the best model on the test set
    test_score = best_model.score(X_test, y_test)
    print("SVC")
    # Print the results
    print("Best Parameters:", best_params)

    print("Test Score with Best Parameters:", test_score)

    # Train Confusion Matrix
    y_pred = best_model.predict(X_test)
    CM = confusion_matrix(y_test, y_pred)

    # Define class labels
    labels = ["Without Complication", "With Complication"]

    # Plot the confusion matrix using seaborn
    sns.heatmap(
        CM, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Best Model)')
    plt.show()

    # Classification Report
    Report = classification_report(
        y_test, y_pred,
        target_names=labels
    )
    print("Classification Report (Best Model):\n", Report)


#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#

#This is the final code after find the best hyperparameter

def Final_Model():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.svm import SVC
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
    from sklearn.exceptions import ConvergenceWarning

    # Suppress warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

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

    kernel = 'rbf'
    C = 2
    gamma = 4
    degree = 2

    # Initialize the SVC model with the chosen hyperparameters
    svc = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, probability=True, random_state=44)

    # Train the model
    svc.fit(X_train, y_train)

    # Evaluate the model on the training and testing sets
    train_accuracy = svc.score(X_train, y_train)
    test_accuracy = svc.score(X_test, y_test)
    print("SVC:")
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Make predictions on the test set
    y_pred = svc.predict(X_test)

    # Confusion Matrix
    CM = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:\n', CM)

    # Define class labels
    labels = ["Without Complication", "With Complication"]

    # Display Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=labels)
    disp.plot(cmap=plt.cm.Greens)
    plt.title("Confusion Matrix - SVC")
    plt.show()

    # Classification Report
    Report = classification_report(y_test, y_pred, target_names=labels)
    print("Classification Report:\n", Report)

    # Compute AUC if binary classification
    unique_labels = np.unique(y_test)
    if len(unique_labels) == 2:
        y_prob = svc.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        print("Test AUC:", roc_auc)

        plt.figure()
        plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % roc_auc, color='purple')  # Changed color to purple
        plt.plot([0, 1], [0, 1], "r--")  # Diagonal line for random classifier
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic - SVC")
        plt.legend(loc="lower right")
        plt.show()

    # Class Distribution in Train and Test Sets
    train_class_counts = np.bincount(y_train)
    test_class_counts = np.bincount(y_test)
    classes = np.arange(len(train_class_counts))

    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    x = np.arange(len(classes))

    # Set custom colors
    train_color = "royalblue"
    test_color = "darkorange"

    plt.bar(x - bar_width / 2, train_class_counts, width=bar_width, label='Train Samples', color=train_color, alpha=0.7)
    plt.bar(x + bar_width / 2, test_class_counts, width=bar_width, label='Test Samples', color=test_color, alpha=0.7)

    plt.xlabel('Class Labels')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Train and Test Sets')
    plt.xticks(classes, labels)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
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
Final_Model()






