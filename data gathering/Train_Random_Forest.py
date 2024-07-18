from sklearn.metrics import f1_score, roc_curve, auc, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Train_Random_Forest:

   def train_random_forrest(df):
      X = df.drop('results', axis=1).copy()
      X_cleaned = X.dropna()
      #print(X)
      y = df['results'].copy()
      y_cleaned = y[X_cleaned.index]
        
        
      X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

        
      model = RandomForestClassifier()

      model.fit(X_train, y_train)

        

         #model score
      model_score = model.score(X_test, y_test)
        
         # Calculate F1-score
      y_pred = model.predict(X_test)
      f1 = f1_score(y_test, y_pred)

         #calculate sensitivity
      sensitivity = recall_score(y_test, y_pred)

        # Calculate AUC-ROC score
      y_prob = model.predict_proba(X_test)[:, 1]
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


   def main(data_frame):
        
      Train_Random_Forest.train_random_forrest(data_frame)