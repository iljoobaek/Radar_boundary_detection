import os, sys, copy, math
import tkinter as tk
from tkinter import filedialog

ground_truth = {}
def read_ground_truth():
    with open(ground_truth_path, "r") as f:
        for line in f:
            ground_truth[int(line.split(',')[0])] = float(line.split(',')[1])
    return

test_data = {}
def read_test_data():
    with open(test_path, "r") as f:
        for line in f:
            test_data[int(line.split(',')[0])] = float(line.split(',')[1])
    return


# True positive — actual = 1, predicted = 1
# False positive — actual = 0, predicted = 1
# False negative — actual = 1, predicted = 0
# True negative — actual = 0, predicted = 0
def evaluate_test_data(criteria = 0.5):
    evaluation_result = []
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for i in range(len(ground_truth)):
        try:
            if ground_truth[i] == -1 and test_data[i] != -1:
                false_positive += 1
                evaluation_result.append(0)
            
            if ground_truth[i] == -1 and test_data[i] == -1:
                true_negative += 1
                evaluation_result.append(1)

            if ground_truth[i] != -1 and test_data[i] == -1:
                false_negative += 1
                evaluation_result.append(0)
            
            if ground_truth[i] != -1 and test_data[i] != -1:
                true_positive += 1
                if criteria > (ground_truth[i] - test_data[i]):
                    evaluation_result.append(1)
        except:
            continue    

    sum = 0
    for result in evaluation_result:
        sum += result
    
    print("=====\n   total: " + str(len(ground_truth)) + " frames")
    print("   criteria: " + str(criteria) + " meters difference")
    print("   success: " + str(sum) + " frames" )

    print("\n   Precision: How many selected items are relative?")
    print("   Precicion: %.2f" % (100 * float(true_positive) / (true_positive + false_positive)) + "%")
    print("\n   Recall: How many relevant items are selected?")
    print("   Recall: %.2f" % (100 * float(true_positive) / (true_positive + false_negative)) + "%")
    
if __name__ == "__main__":
    logpath = ""
    root = tk.Tk()
    root.withdraw()
    ground_truth_path = filedialog.askopenfilename()
    read_ground_truth()
    test_path = ground_truth_path.split("ground_truth_")[0] + "test_" + ground_truth_path.split("ground_truth_")[1]
    #test_path = filedialog.askopenfilename()
    read_test_data()
    root.destroy()

    print("=====\nground_truth: " + os.path.basename(ground_truth_path))
    print("ground_truth len: " + str(len(ground_truth)))
    print("\ntest_data: " + os.path.basename(test_path))
    print("test_data len: " + str(len(test_data)))

    evaluate_test_data()