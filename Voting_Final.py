#Eng\Yazeed fahad alzaidi
#Eng\Mohammed Alhumaidi
#Eng\Nawaf bin shafi
#Eng\Abdulmohsen Alghamdi
#Eng\Waleed abdi
#Eng\Mohammed Altwirqi
#Project Advisor Prof. Httan Ali Asiri
#---------------------------------------------------------------------------------------------------------------------#
#Voting Classifier Model

def Final():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.svm import SVC
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

    # Define the classifiers with the best hyperparameters
    rf = RandomForestClassifier(
        n_estimators=250, criterion="entropy", max_depth=40,
        min_samples_leaf=1, min_samples_split=29, bootstrap=False, random_state=44
    )

    gb = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=10,
        min_samples_leaf=30, min_samples_split=50, subsample=0.8, random_state=44
    )

    svc = SVC(kernel="rbf", C=2, gamma=4, degree=2, probability=True, random_state=44)

    # Define the Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[('RandomForest', rf), ('GradientBoosting', gb), ('SVC', svc)],
        voting='soft'
    )

    # Train the ensemble model
    voting_clf.fit(X_train, y_train)

    # Evaluate the model
    train_accuracy = voting_clf.score(X_train, y_train)
    test_accuracy = voting_clf.score(X_test, y_test)
    print("Voting Classifier:")
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Make predictions
    y_pred = voting_clf.predict(X_test)

    # Confusion Matrix
    CM = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:\n', CM)

    # Define class labels
    labels = ["Without Complication", "With Complication"]

    # Display Confusion Matrix with 'YlGnBu' color
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=labels)
    disp.plot(cmap="YlGnBu")  # Updated color scheme
    plt.title("Confusion Matrix - Voting Classifier")
    plt.show()

    # Classification Report
    Report = classification_report(y_test, y_pred, target_names=labels)
    print("Classification Report:\n", Report)

    # Compute AUC if binary classification
    unique_labels = np.unique(y_test)
    if len(unique_labels) == 2:
        y_prob = voting_clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        print("Test AUC:", roc_auc)

        plt.figure()
        plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % roc_auc, color='blue')  # Updated color
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic - Voting Classifier")
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

Final()
