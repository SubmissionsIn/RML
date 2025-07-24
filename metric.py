import numpy
import scipy.io
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch
import pandas as pd 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score


def classification_evaluation(y_true, y_predict):
    accuracy=classification_report(y_true, y_predict, output_dict=True, zero_division=1)['accuracy']
    s=classification_report(y_true, y_predict, output_dict=True, zero_division=1)['weighted avg']
    precision=s['precision']
    recall=s['recall']
    f1_score=s['f1-score']
    # kappa=cohen_kappa_score(y_true, y_predict)
    return accuracy, precision, recall, f1_score #, kappa


def scale_normalize_matrix(input_matrix, min_value=0, max_value=1):
    min_val = input_matrix.min()
    max_val = input_matrix.max()
    input_range = max_val - min_val
    scaled_matrix = (input_matrix - min_val) / input_range * (max_value - min_value) + min_value
    return scaled_matrix


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur


def inference(loader, model, device, view, classification=False, label=0):
    model.eval()
    # soft_vector = []

    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
            # print(xs[v].shape)

        # if classification:
        #     label_tmp = torch.ones(xs[v].shape) * label
        #     xs[v] = label_tmp.to(device)

        # xs_all = torch.cat(xs, dim=1)

        with torch.no_grad():
            # _, h, z, q = model.forward(xs_all)
            h, z, q, scores, hs = model.forward(xs)
        z = z.cpu().detach().numpy()
        h = h.cpu().detach().numpy()
        q = q.cpu().detach().numpy()
        # soft_vector.extend(q.cpu().detach().numpy())
    # total_pred = np.argmax(np.array(soft_vector), axis=1)
    q = q.argmax(1)
    # print(scores.shape)
    # print(scores[0])
    y = y.numpy()
    y = y.flatten()
    return y, h, z, q, hs, xs


def valid(model, device, dataset, view, data_size, class_num, eval_q=False, eval_z=False, eval_x=False):
    test_loader = DataLoader(dataset, batch_size=data_size, shuffle=False)
    labels_vector, h, z, q, hs, xs = inference(test_loader, model, device, view, classification=False, label=0)

    if eval_x == True:
        metric1 = []
        metric2 = []
        metric3 = []
        metric4 = []
        for v in range(len(xs)):
            xs[v] = xs[v].cpu().detach().numpy()
        for i in range(5):
            kmeans = KMeans(n_clusters=class_num)
            x_pred = kmeans.fit_predict(np.concatenate(xs, axis=1))
            nmi_x, ari_x, acc_x, pur_x = evaluate(labels_vector, x_pred)
            print('ACC_x = {:.4f} NMI_x = {:.4f} ARI_x = {:.4f} PUR_x = {:.4f}'.format(acc_x, nmi_x, ari_x, pur_x))
            metric1.append(acc_x)
            metric2.append(nmi_x)
            metric3.append(ari_x)
            metric4.append(pur_x)
        print('%.3f' % np.mean(metric1), '± %.3f' % np.std(metric1), metric1)
        print('%.3f' % np.mean(metric2), '± %.3f' % np.std(metric2), metric2)
        print('%.3f' % np.mean(metric3), '± %.3f' % np.std(metric3), metric3)
        print('%.3f' % np.mean(metric4), '± %.3f' % np.std(metric4), metric4)

    if eval_z == True:
        kmeans = KMeans(n_clusters=class_num)
        for v in range(len(hs)):
            hs[v] = hs[v].cpu().detach().numpy()
            z_pred = kmeans.fit_predict(hs[v])
            nmi_z, ari_z, acc_z, pur_z = evaluate(labels_vector, z_pred)
            print('ACC_v = {:.4f} NMI_v = {:.4f} ARI_v = {:.4f} PUR_v = {:.4f}'.format(acc_z, nmi_z, ari_z, pur_z))

        # h_pred = kmeans.fit_predict(np.concatenate(hs, axis=1))
        # nmi_h, ari_h, acc_h, pur_h = evaluate(labels_vector, h_pred)
        # print('ACC_h = {:.4f} NMI_h = {:.4f} ARI_h = {:.4f} PUR_h = {:.4f}'.format(acc_h, nmi_h, ari_h, pur_h))

        z_pred = kmeans.fit_predict(z)
        nmi_z, ari_z, acc_z, pur_z = evaluate(labels_vector, z_pred)
        print('ACC_z = {:.4f} NMI_z = {:.4f} ARI_z = {:.4f} PUR_z = {:.4f}'.format(acc_z, nmi_z, ari_z, pur_z))
        return acc_z, nmi_z, ari_z, pur_z

    if eval_q == True:
        accuracy, precision, recall, f1_score = classification_evaluation(labels_vector, q)
        print('ACC_q = {:.4f} Precision_q = {:.4f} F1-score_q = {:.4f} Recall_q = {:.4f}'.format(accuracy, precision, f1_score, recall))
        return accuracy, precision, f1_score, recall
