import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

from utils.meta2vec import get_features
from utils.split_data import split_data


def run_logistic_regression(hetero_data, epoch):

    hetero_data, test_data, train_data, val_data = split_data(hetero_data)
    train_data = get_features(train_data)
    val_data['gene'].x = train_data['gene'].x
    test_data['gene'].x = train_data['gene'].x
    val_data['disease'].x = train_data['disease'].x
    test_data['disease'].x = train_data['disease'].x

    X_train_pos = (train_data['gene'].x[train_data[('gene', 'associate', 'disease')].pos_edge_label_index[0]] * train_data['disease'].x[
        train_data[('gene', 'associate', 'disease')].pos_edge_label_index[1]]).sum(dim=1)
    X_train_neg = (train_data['gene'].x[train_data[('gene', 'associate', 'disease')].neg_edge_label_index[0]] * train_data['disease'].x[
        train_data[('gene', 'associate', 'disease')].neg_edge_label_index[1]]).sum(dim=1)
    X_train = torch.cat([X_train_pos, X_train_neg], dim=0).detach().cpu().numpy().reshape(-1, 1)

    pos_y_train = X_train_pos.new_ones(train_data[('gene', 'associate', 'disease')].pos_edge_label_index.size(1))
    neg_y_train = X_train_pos.new_zeros(train_data[('gene', 'associate', 'disease')].pos_edge_label_index.size(1))
    y_train = torch.cat([pos_y_train, neg_y_train], dim=0).detach().cpu().numpy()

    X_val_pos = (val_data['gene'].x[val_data[('gene', 'associate', 'disease')].pos_edge_label_index[0]] * val_data['disease'].x[
        val_data[('gene', 'associate', 'disease')].pos_edge_label_index[1]]).sum(dim=1)
    X_val_neg = (val_data['gene'].x[val_data[('gene', 'associate', 'disease')].neg_edge_label_index[0]] * val_data['disease'].x[
        val_data[('gene', 'associate', 'disease')].neg_edge_label_index[1]]).sum(dim=1)
    X_val = torch.cat([X_val_pos, X_val_neg], dim=0).detach().cpu().numpy().reshape(-1, 1)

    pos_y_val = X_val_pos.new_ones(val_data[('gene', 'associate', 'disease')].pos_edge_label_index.size(1))
    neg_y_val = X_val_pos.new_zeros(val_data[('gene', 'associate', 'disease')].pos_edge_label_index.size(1))
    y_val = torch.cat([pos_y_val, neg_y_val], dim=0).detach().cpu().numpy()

    X_test_pos = (test_data['gene'].x[test_data[('gene', 'associate', 'disease')].pos_edge_label_index[0]] * test_data['disease'].x[
        test_data[('gene', 'associate', 'disease')].pos_edge_label_index[1]]).sum(dim=1)
    X_test_neg = (test_data['gene'].x[test_data[('gene', 'associate', 'disease')].neg_edge_label_index[0]] * test_data['disease'].x[
        test_data[('gene', 'associate', 'disease')].neg_edge_label_index[1]]).sum(dim=1)
    X_test = torch.cat([X_test_pos, X_test_neg], dim=0).detach().cpu().numpy().reshape(-1, 1)

    pos_y_test = X_test_pos.new_ones(test_data[('gene', 'associate', 'disease')].pos_edge_label_index.size(1))
    neg_y_test = X_test_pos.new_zeros(test_data[('gene', 'associate', 'disease')].pos_edge_label_index.size(1))
    y_test = torch.cat([pos_y_test, neg_y_test], dim=0).detach().cpu().numpy()

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Validation predictions
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]  # Probability of class 1
    y_val_pred = model.predict(X_val)

    # Evaluation
    roc_auc = roc_auc_score(y_val, y_val_pred_proba)
    accuracy = accuracy_score(y_val, y_val_pred)
    ap = average_precision_score(y_val, y_val_pred)

    print(f"Validation ROC-AUC: {roc_auc:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {ap:.4f}")


    # Test predictions
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = model.predict(X_test)

    # Evaluation
    roc_auc_test = roc_auc_score(y_test, y_test_pred_proba)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    ap_test = average_precision_score(y_test, y_test_pred)

    print(f"Test ROC-AUC: {roc_auc_test:.4f}")
    print(f"Test Accuracy: {accuracy_test:.4f}")
    print(f"Test Precision: {ap_test:.4f}")
