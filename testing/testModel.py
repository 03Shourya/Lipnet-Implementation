import os 
from model import predict_and_evaluate

def main(folder_path):
    accuracies = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            _,_,accuracy =predict_and_evaluate(file_path)
            accuracies.append(accuracy)
            print(f"{filename}: {accuracy:.4f}")
    
    if accuracies:
        mean_accuracy = sum(accuracies) / len(accuracies)
        print(f"\nMean Accuracy: {mean_accuracy:.4f}")
    else:
        print("No files found in the folder.")

# Example usage
folder = "data/s1"  # Replace with your folder path
main(folder)