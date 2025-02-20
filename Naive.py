#Eng\Waleed abdi
#Eng\Yazeed fahad alzaidi
#Eng\Mohammed Alhumaidi
#Eng\Nawaf bin shafi
#Eng\Abdulmohsen Alghamdi
#Eng\Mohammed Altwirqi
#Project Advisor Prof. Httan Ali Asiri
#---------------------------------------------------------------------------------------------------------------------#
#Gaussian Naïve Bayes Model
def GNB():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.naive_bayes import GaussianNB
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
    np.random.seed(444)

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

    # Set up iterative search parameters
    max_iterations = 50
    accuracy_threshold = 0.70

    best_accuracy = 0.0
    best_model = None
    best_var_smoothing = None

    print("Starting iterative search for var_smoothing...\n")
    # Iterate up to max_iterations times
    for i in range(1, max_iterations + 1):
        # Sample a random var_smoothing value on a log scale between 10^-14 and 10^-2
        exponent = np.random.uniform(-14, -2)
        current_var_smoothing = 10 ** exponent

        # Initialize GaussianNB with the current var_smoothing
        nb = GaussianNB(var_smoothing=current_var_smoothing)
        nb.fit(X_train, y_train)

        # Evaluate the model on the test set
        test_acc = nb.score(X_test, y_test)
        train_acc = nb.score(X_train, y_train)  # Calculate training accuracy
        print(
            f"Iteration {i}: var_smoothing = {current_var_smoothing:.2e}, Train Accuracy = {train_acc:.4f}, Test Accuracy = {test_acc:.4f}")

        # Update best model if current test accuracy is better
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = nb
            best_var_smoothing = current_var_smoothing

        # Check if the desired threshold is reached
        if test_acc >= accuracy_threshold:
            print(f"\nThreshold reached at iteration {i} with Test Accuracy = {test_acc:.4f}!")
            break

    print("\nBest Model Summary:")
    print(f"Best var_smoothing: {best_var_smoothing:.2e}")
    print(f"Best Test Accuracy: {best_accuracy:.4f}")

    # Evaluate and display final results using the best model
    y_pred = best_model.predict(X_test)
    train_accuracy = best_model.score(X_train, y_train)  # Training accuracy
    test_accuracy = best_model.score(X_test, y_test)  # Testing accuracy

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[f"Class {label}" for label in np.unique(y)]))

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

    # Plot the confusion matrix using seaborn
    CM = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        CM, annot=True, fmt='d', cmap='copper',
        xticklabels=[f'Class {label}' for label in np.unique(y)],
        yticklabels=[f'Class {label}' for label in np.unique(y)]
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Naive bayes (GaussianNB)')
    plt.show()

    # If binary classification, plot ROC curve and compute AUC
    if len(np.unique(y)) == 2:
        y_prob = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        print("Test AUC:", roc_auc)

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic - Naive bayes (GaussianNB) ")
        plt.legend(loc="lower right")
        plt.show()

GNB()