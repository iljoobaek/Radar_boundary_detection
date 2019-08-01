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

temporary_data = {}
def read_temporary():
    with open(temporary_path, "r") as f:
        for line in f:
            temporary_data[int(line.split(',')[0])] = float(line.split(',')[1])

# True positive — actual = 1, predicted = 1
# False positive — actual = 0, predicted = 1
# False negative — actual = 1, predicted = 0
# True negative — actual = 0, predicted = 0
def evaluate_test_data(criteria = 0.2):
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
                if criteria > (ground_truth[i] - test_data[i]):
                    true_positive += 1
                    evaluation_result.append(1)
                else:
                    false_positive += 1
                    evaluation_result.append(0)
        except:
            continue    

    sum = 0
    for result in evaluation_result:
        sum += result
    
    print("=====\n   total: " + str(len(ground_truth)) + " frames")
    print("   criteria: " + str(criteria) + " meters difference")
    print("   success: " + str(sum) + " frames" )

    print("\n   Precision: %.2f" % (100 * float(true_positive) / (true_positive + false_positive)) + "%")
    print("\n   Recall: %.2f" % (100 * float(true_positive) / (true_positive + false_negative)) + "%")
    
if __name__ == "__main__":
    
    criteria = 0.5
    if len(sys.argv) == 2:
        ground_truth_path = sys.argv[1]
    else:
        root = tk.Tk()
        root.withdraw()
        ground_truth_path = filedialog.askopenfilename()
        root.destroy()
    read_ground_truth()
    test_path = ground_truth_path.split("ground_truth_")[0] + "test_" + ground_truth_path.split("ground_truth_")[1]
    read_test_data()

    temporary_path = ground_truth_path.split("ground_truth_")[0] + "temporary_ground_truth_" + ground_truth_path.split("ground_truth_")[1]
    try:
        read_temporary()
    except:
        pass
        #print("no temporary for this ground truth")

    print("=====\nground_truth: " + os.path.basename(ground_truth_path))
    print("len: " + str(len(ground_truth)))
    print("\ntest_data: " + os.path.basename(test_path))
    print("len: " + str(len(test_data)))

    evaluate_test_data(criteria)