from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import json

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def process_data(model_name: str, path: str):
    with open(path, "r") as f:
        data = json.load(f)

    for dataset_name, outer_list in data.items():
        y_true = []
        y_scores = []
        print(dataset_name)
        for inner_list in outer_list:
            for item in inner_list:
                if isinstance(item, str):
                    continue
                label = 1 if item["actual_label"] == "same" else 0
                score = cosine_similarity(item["emb1"], item["emb2"])

                y_true.append(label)
                y_scores.append(score)

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--")  # random baseline

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} {dataset_name} Dataset ROC Curve")
        plt.legend()
        plt.savefig(f"roc_curves/{model_name}/{model_name}_{dataset_name}_ROC_Curve.png")
        plt.close()

if __name__ == "__main__":
    process_data("DLib", "dlib_results.json")
    process_data("ArcFace", "arcface_results.json")
    process_data("FaceNet", "facenet_results.json")
