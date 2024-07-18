from sklearn.metrics import f1_score, roc_curve, auc, recall_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class SVC_Train_standard:
 

    
    def train_SVC(df):
        X = df.drop('results', axis=1).copy()
        X_cleaned = X.dropna()
        #print(X)
        y = df['results'].copy()
        y_cleaned = y[X_cleaned.index]
        
        
        X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)  
        #print(X_train)
        #print(X_train_scaled)      
        
        model = SVC(probability=True)

        model.fit(X_train_scaled, y_train)

        

        #model score
        model_score = model.score(X_test_scaled , y_test)
        
        # Calculate F1-score
        y_pred = model.predict(X_test_scaled )
        f1 = f1_score(y_test, y_pred)

        #calculate sensitivity
        sensitivity = recall_score(y_test, y_pred)

        # Calculate AUC-ROC score
        y_prob = model.predict_proba(X_test_scaled )[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        # Calculate AUC
        roc_auc = auc(fpr, tpr)

        #calculate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
       
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

        #calculate specificity
        tn, fp, fn, tp = conf_matrix.ravel()
        specificity = tn / (tn + fp)



      
        
        
        #print("Principal Components:\n", pca.components_)
        print("F1-score:", f1)
        print("AUC-ROC Score:", roc_auc)
        print("Sensitivity (Recall): ",sensitivity)
        print("Specificity: ",specificity)
        print("Model Score :", model_score)
        #print(np.mean(scores))
        
    def train_PCA(df):

        

        pca = PCA()
        pca.fit_transform(df)

        eigenvalues = pca.explained_variance_

        # Plot the scree plot
        plt.plot(1, len(eigenvalues) + 1, eigenvalues, marker = "o")
        plt.xlabel('Principal Component Number')
        plt.ylabel('Eigenvalue')
        plt.title('Eigenvalues against Priciple Components')
        plt.show()


        
        
        
        



    def main(data_frame):
        
        SVC_Train_standard.train_SVC(data_frame)
        #SVC_Train.train_PCA(data_frame)