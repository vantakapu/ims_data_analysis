import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report
from clustering import cluster_result

from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import streamlit as st







# Define the training class
class training:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.lda = LinearDiscriminantAnalysis()

    def calculate_medians(self):
        unique_labels = np.unique(self.y)
        medians = []
        for label in unique_labels:
            group_data = self.x[self.y == label]
            median = np.median(group_data, axis=0)
            medians.append(median)
        return np.array(medians)


    def pooled_covariance(self):
        unique_labels = np.unique(self.y)
        cov_matrices = []
        for label in unique_labels:
            group_data = self.x[self.y == label]
            # Ensure there's more than one sample for the class to compute covariance
            if len(group_data) > 1:
                cov_matrix = np.cov(group_data, rowvar=False)
                cov_matrices.append(cov_matrix)
        # If there are valid covariance matrices to average
        if cov_matrices:
            pooled_cov = np.average(cov_matrices, axis=0)
            return pooled_cov
        else:
            return None


    def parameters(self):
        self.lda.fit(self.x, self.y)
        y_pred = self.lda.predict(self.x)
        cm = confusion_matrix(self.y, y_pred, labels=np.unique(self.y))
        # Compute specificity
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        specificity = TN / (TN + FP)
        
        columns = ['True-group-' + str(i) for i in np.unique(self.y)]
        indices = ['Predicted-group-' + str(i) for i in np.unique(self.y)]
        cm_df = pd.DataFrame(cm, columns=columns, index=indices)
        class_report = classification_report(self.y, y_pred, output_dict=True, target_names=columns)
        
        # Rename recall as sensitivity and add specificity
        for key, value in class_report.items():
            if key in columns:
                value["sensitivity(recall)"] = value.pop("recall")
                value["specificity"] = specificity[int(key.split('-')[-1]) - 1]
        
        report_df = pd.DataFrame(class_report).transpose()
        means = self.lda.means_
        means_df = pd.DataFrame(means)
        covariance = self.pooled_covariance()
        covariance_df = pd.DataFrame(covariance)
        x_lda = self.lda.fit_transform(self.x, self.y)
        Eigen_values = self.lda.scalings_
        Eigen_values_df = pd.DataFrame(Eigen_values)
        coefficients = self.lda.coef_
        coefficients_df = pd.DataFrame(coefficients)
        x_proj = np.matmul(np.transpose(Eigen_values), np.transpose(self.x))
        x_proj_diff = x_lda - np.transpose(x_proj)
        x_proj_mean = np.mean(x_proj_diff, 0)
        median_df = pd.DataFrame(self.calculate_medians())
        
        return y_pred, means_df, covariance_df, report_df, cm_df, x_lda, Eigen_values_df, coefficients_df, x_proj_mean, median_df
    
    
    
    def plot_confusion_matrix(self):
        _, _, _, _, cm_df, _, _, _, _, _ = self.parameters()
        plt.figure(figsize=(10,7))
        sns.heatmap(cm_df, annot=True, cmap="YlGnBu", fmt='g')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()








# Define the plotting class
class plotting:
    def __init__(self, train_data, labels_data, x_coord_3d=1, y_coord_3d=2, z_coord_3d=3, x_2d=1, y_2d=2):
        self.train_data = train_data
        self.labels = labels_data
        self.x_3d = x_coord_3d
        self.y_3d = y_coord_3d
        self.z_3d = z_coord_3d
        self.x_2d = x_2d
        self.y_2d = y_2d

    def plots3d(self):
        l = training(self.train_data, self.labels)
        _, _, _, _, _, x_lda, _, _, _, _ = l.parameters()
        plt.close("all")
        plt.ioff()
        fig = plt.figure(clear=True)
        ax = fig.add_subplot(projection='3d')
        sc = ax.scatter(x_lda[:, self.x_3d - 1], x_lda[:, self.y_3d - 1], x_lda[:, self.z_3d - 1], c=self.labels, cmap='jet', alpha=0.7)
        ax.set_xlabel('Discriminant function ' + str(self.x_3d))
        ax.set_ylabel('Discriminant function ' + str(self.y_3d))
        ax.set_zlabel('Discriminant function ' + str(self.z_3d))
        plt.legend(*sc.legend_elements(), bbox_to_anchor=(1, 0.9), loc=2)
        plt.show()

    def plots2d(self):
        l = training(self.train_data, self.labels)
        _, _, _, _, _, x_lda, _, _, _, _ = l.parameters()
        plt.close("all")
        plt.ioff()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sc = ax.scatter(x_lda[:, self.x_2d - 1], x_lda[:, self.y_2d - 1], c=self.labels, cmap='jet', alpha=0.7)
        ax.set_xlabel('Discriminant function ' + str(self.x_2d))
        ax.set_ylabel('Discriminant function ' + str(self.y_2d))
        plt.legend(*sc.legend_elements(), bbox_to_anchor=(1, 0.9), loc=2)
        plt.show()




if __name__ == '__main__':
    ''' Evaluation of Classification '''

    data = cluster_result
    data


    # for col in data.columns[3:-1]:
    #     data[col] = data[col].str.replace(',', '.').astype(float)
    X = data.drop(['File', 'group',], axis=1).values
    X.shape
    y = data['group'].values
    y




    
    
    
    # Use the training class
    train_instance = training(X, y)
    y_pred, means_df, covariance_df, report_df, cm_df, x_lda, Eigen_values_df, coefficients_df, x_proj_mean, median_df = train_instance.parameters()







    # Use the plotting class for visualization (if required)
    plot_instance = plotting(X, y)
    plot_instance.plots2d()
    train_instance.plot_confusion_matrix()  # Plot the confusion matrix





    ''' Cross-fold verification '''

    lda = LinearDiscriminantAnalysis()
    scores = cross_val_score(lda, X, y, cv=3)  # using 5-fold cross validation
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))







