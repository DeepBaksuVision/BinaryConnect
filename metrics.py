import torch


def confusion_matrix(out, target):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(target)):
        # True Positive, 실제 True(1)인 정답을 True로 예측(정답)
        if target[i] == out[i] == 1:
            tp += 1
        # False Positive, 실제는 False지만 정답을 True(1)라고 예측(오답)
        if out[i] == 1 and target[i] != out[i]:
            fp += 1
        # True Negative, 실제는 False를 False라고 예측(정답)
        if target[i] == out[i] == 0:
            tn += 1
        # False Negative, 실제는 True인 정답을 False라고 예측(오답)
        if target[i] == 0 and target[i] != out[i]:
            fn += 1

    return tp, fp, tn, fn


def accuracy(out, target):
    with torch.no_grad():
        _, predicted = torch.max(out, 1)
        acc = (predicted == target).sum() / target.size(0)
    return acc


def precision(out, target):
    with torch.no_grad():
        tp, fp = confusion_matrix(out, target)[0], confusion_matrix(out, target)[1]
    return tp / (tp + fp)


def recall(out, target):
    with torch.no_grad():
        tp, fn = confusion_matrix(out, target)[0], confusion_matrix(out, target)[3]
    return tp / (tp + fn)


