import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import(
    accuracy_score,recall_score,precision_score, f1_score,classification_report, confusion_matrix, ConfusionMatrixDisplay
)

def evaluate_model(model,X_train, y_train, X_test, y_test):

    train_preds= np.rint(model.predict(X_train)) # np.rint = round to the nearest INT
    test_preds= np.rint(model.predict(X_test))

    # Class report
    train_report = classification_report(y_train,train_preds)
    test_report = classification_report(y_test,test_preds)

    # Confusion matrix 
    cm_train = confusion_matrix(y_train,train_preds)
    cm_test = confusion_matrix(y_test,test_preds)

    # Formal all figures in dark mode
    plt.style.use('dark_background')

    # plot train summary and confusion matrix
    fig, axes = plt.subplots(1,2,figsize=(14,5))

    #First axes
    axes[0].text(0.01,0.05,str(train_report),{'fontsize':12}, fontproperties='monospace')
    axes[0].axis('off')

    # Second Axes
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train)
    disp_train.plot(ax=axes[1],cmap='Blues')
    axes[1].set_title('Confusion Matrix-Training Set ')

    # plot test summary and confusion matrix
    fig, axes = plt.subplots(1,2,figsize=(14,5))

    #First axes
    axes[0].text(0.01,0.05,str(test_report),{'fontsize':12}, fontproperties='monospace')
    axes[0].axis('off')

    # Second Axes
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_test)
    disp_train.plot(ax=axes[1],cmap='Purples')
    axes[1].set_title('Confusion Matrix-Test Set ')

    plt.show()

    return train_report, test_report

from sklearn.pipeline import Pipeline

# Create ML pipeline
def train_and_predict_model(X_train,y_train,X_test,preprocessor,model):

    # Combine preprocessing pipeline and model
    model_pipe =Pipeline(steps=[
        ('preprocessor',preprocessor),
        ('classifier',model),
    ])
 
    # Fit pipeline
    model_pipe.fit(X_train,y_train)
    
    # Create predictions
    train_preds = model_pipe.predict(X_train)
    test_preds=model_pipe.predict(X_test)

    return train_preds,test_preds

import joblib
import os

# Save model
def save_model(model, model_path):
    try:
        # Save the model using joblib
        joblib.dump(model, model_path)
        
        # Confirm save if successful
        if os.path.exists(model_path):
            print(f'Model saved successfully to: {model_path}')
        else:
            print(f'Failed to save model: {model_path}')
        
    except Exception as error:
        print(f'Error saving model to {model_path}: {error}')