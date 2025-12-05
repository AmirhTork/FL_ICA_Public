import torch
from sklearn.metrics import f1_score, accuracy_score

def evaluate_model(model, features, labels):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(features, dtype=torch.float32)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        acc = accuracy_score(labels, preds)
        f1_w = f1_score(labels, preds, average='weighted')
        return {'accuracy': acc, 'f1_weighted': f1_w, 'preds': preds, 'logits': outputs.detach(), 'labels': labels}
 
