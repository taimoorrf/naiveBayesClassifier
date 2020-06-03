import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from PIL import Image
import os

class Classifier():
    train_data = []
    test_data = []
    prior_normal = 0
    prior_abnormal = 0
    normal_rows = []
    abnormal_rows = []
    normal_rows_final = []
    abnormal_rows_final = []
    test_normal = []
    test_abnormal = []
    pass_normal_counts = []
    pass_abnormal_counts = []
    fail_normal_counts = []
    fail_abnormal_counts = []
    normal_frequency = 0
    abnormal_frequency = 0
    y_true = None
    
    def train(self, filename):
        #Reading file and training
        file = open(filename, 'r')
        lines = file.readlines()
        patients_record = []
        for line in lines:
            patients_record.append(line)
        
        # self.train_data = patients_record
        for i in range(len(patients_record)):
            if patients_record[i][0] == '1':
                self.prior_normal = self.prior_normal + 1
                self.normal_rows.append(patients_record[i])
            elif patients_record[i][0] == '0':
                self.prior_abnormal = self.prior_abnormal + 1
                self.abnormal_rows.append(patients_record[i])
        self.prior_normal = self.prior_normal / len(patients_record)
        self.prior_abnormal = self.prior_abnormal / len(patients_record)
        ## Separating normal rows    
        for i in range(len(self.normal_rows)):
            temp_np = []
            for j in range(len(self.normal_rows[i])):
                if self.normal_rows[i][j] == ',' or self.normal_rows[i][j] == '\n': 
                    continue
                else:
                    temp_np.append(int(self.normal_rows[i][j]))
            self.normal_rows_final.append(temp_np)
        ## Separating abnormal_rows
        for i in range(len(self.abnormal_rows)):
            temp_np = []
            for j in range(len(self.abnormal_rows[i])):
                if self.abnormal_rows[i][j] == ',' or self.abnormal_rows[i][j] == '\n': 
                    continue
                else:
                    temp_np.append(int(self.abnormal_rows[i][j]))
            self.abnormal_rows_final.append(temp_np)
            
        self.normal_rows_final = np.array(self.normal_rows_final)
        self.abnormal_rows_final = np.array(self.abnormal_rows_final)
        self.normal_rows_final = self.normal_rows_final[:, 1:]      ## Removing first column and keeping only data of tests
        self.abnormal_rows_final = self.abnormal_rows_final[:, 1:]  ## Removing first column and keeping only data of tests
        
        for i in range(len((self.normal_rows_final).T)):
            col = self.normal_rows_final[:, i]
            self.pass_normal_counts.append(sum(col))
            self.fail_normal_counts.append(len(self.normal_rows_final) - sum(col))
        
        for i in range(len((self.abnormal_rows_final).T)):
            col = self.abnormal_rows_final[:, i]
            self.pass_abnormal_counts.append(sum(col))
            self.fail_abnormal_counts.append(len(self.abnormal_rows_final) - sum(col))

        
        self.normal_frequency = len(self.normal_rows)
        self.abnormal_frequency = len(self.abnormal_rows)
        
        

        
    
    def accuracy(self, y_pred, y_true):
        length = len(y_pred)
        same = 0
        for i in range(length):
            if y_pred[i] == y_true[i]:
                same = same + 1
        return (same/length) * 100
    
    
    def likelihood(self, row):
        likelihood_normal = 0
        likelihood_abnormal = 0
        normal_probabilities = []
        abnormal_probabilities = []
        
        for i in range(len(row)):
            if row[i] == 1:
                normal_probabilities.append((self.pass_normal_counts[i])/self.normal_frequency)
                abnormal_probabilities.append(self.pass_abnormal_counts[i]/self.abnormal_frequency)

            elif row[i] == 0:
                normal_probabilities.append((self.fail_normal_counts[i])/self.normal_frequency)
                abnormal_probabilities.append((self.fail_abnormal_counts[i])/self.abnormal_frequency)

        likelihood_normal = np.product(normal_probabilities)
        likelihood_abnormal = np.product(abnormal_probabilities)

        return (likelihood_normal, likelihood_abnormal)

    def test(self, filename):
        ## Reading file
        file = open(filename, 'r')
        lines = file.readlines()
        patients_record = []
        for line in lines:
            patients_record.append(line)
        
        for i in range(len(patients_record)):
            temp_np = []
            for j in range(len(patients_record[i])):
                if patients_record[i][j] == ',' or patients_record[i][j] == '\n': 
                    continue
                else:
                    temp_np.append(int(patients_record[i][j]))
            self.test_data.append(temp_np)
            
        self.test_data = np.array(self.test_data)
        self.y_true = np.empty((len(self.test_data)))
        self.y_true = (self.test_data[ :, 0]).T                 ## Converting our y_true to row form
        
        
        y_pred = []
        for i in range(len(self.test_data)):
            likelihood_normal, likelihood_abnormal = self.likelihood(self.test_data[i, 1:])
            if likelihood_normal * self.prior_normal > likelihood_abnormal * self.prior_abnormal: 
                y_pred.append(1)
            else:
                y_pred.append(0)

        return self.accuracy(y_pred, self.y_true)

def main():
    classifier = Classifier()
    
    print('Training...')
    classifier.train(sys.argv[1])
    print('Training Done\n')
    
    print('Testing')
    accuracy = classifier.test(sys.argv[2])
    print('Accuracy:', str(accuracy) + '%')

main()