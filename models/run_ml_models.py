import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from utils.meta2vec import get_features
from utils.split_data import split_data


def run_ml_models(hetero_data, gene_sum, disease_sum):

    hetero_data, test_data, train_data, val_data = split_data(hetero_data)
    train_data = get_features(train_data)

    gene_bio_embed = np.stack(gene_sum['Embedding'].to_list())
    disease_bio_embed = np.stack(disease_sum['Embedding'].to_list())

    # Using PCA to reduce dimension
    pca = PCA(n_components=128)
    gene_bio_embed = pca.fit_transform(gene_bio_embed)
    disease_bio_embed = pca.fit_transform(disease_bio_embed)

    additional_gene_features = torch.tensor(gene_bio_embed, dtype=train_data['gene'].x.dtype)
    additional_disease_features = torch.tensor(disease_bio_embed, dtype=train_data['disease'].x.dtype)

    # # Check dimensions
    # assert additional_gene_features.size(0) == train_data['gene'].x.size(0), "Mismatch in number of gene nodes"
    # assert additional_disease_features.size(0) == train_data['disease'].x.size(0), "Mismatch in number of disease nodes"

    # Concatenate features along the feature dimension
    train_data['gene'].x = torch.cat([train_data['gene'].x, additional_gene_features], dim=1)
    train_data['disease'].x = torch.cat([train_data['disease'].x, additional_disease_features], dim=1)

    val_data['gene'].x = train_data['gene'].x
    test_data['gene'].x = train_data['gene'].x
    val_data['disease'].x = train_data['disease'].x
    test_data['disease'].x = train_data['disease'].x

    X_train_pos = torch.cat([train_data['gene'].x[train_data[('gene', 'associate', 'disease')].pos_edge_label_index[0]],
                             train_data['disease'].x[train_data[('gene', 'associate', 'disease')].pos_edge_label_index[1]]],
                            dim=1)
    X_train_neg = torch.cat([train_data['gene'].x[train_data[('gene', 'associate', 'disease')].neg_edge_label_index[0]],
                             train_data['disease'].x[train_data[('gene', 'associate', 'disease')].neg_edge_label_index[1]]],
                            dim=1)
    # X_train = torch.cat([X_train_pos, X_train_neg], dim=0).detach().cpu().numpy().reshape(-1, 1)
    X_train = torch.cat([X_train_pos, X_train_neg], dim=0).detach().cpu().numpy()

    pos_y_train = X_train_pos.new_ones(train_data[('gene', 'associate', 'disease')].pos_edge_label_index.size(1))
    neg_y_train = X_train_pos.new_zeros(train_data[('gene', 'associate', 'disease')].pos_edge_label_index.size(1))
    y_train = torch.cat([pos_y_train, neg_y_train], dim=0).detach().cpu().numpy()

    # X_val_pos = (val_data['gene'].x[val_data[('gene', 'associate', 'disease')].pos_edge_label_index[0]] * val_data['disease'].x[
    #     val_data[('gene', 'associate', 'disease')].pos_edge_label_index[1]]).sum(dim=1)
    # X_val_neg = (val_data['gene'].x[val_data[('gene', 'associate', 'disease')].neg_edge_label_index[0]] * val_data['disease'].x[
    #     val_data[('gene', 'associate', 'disease')].neg_edge_label_index[1]]).sum(dim=1)
    # X_val = torch.cat([X_val_pos, X_val_neg], dim=0).detach().cpu().numpy().reshape(-1, 1)
    #
    # pos_y_val = X_val_pos.new_ones(val_data[('gene', 'associate', 'disease')].pos_edge_label_index.size(1))
    # neg_y_val = X_val_pos.new_zeros(val_data[('gene', 'associate', 'disease')].pos_edge_label_index.size(1))
    # y_val = torch.cat([pos_y_val, neg_y_val], dim=0).detach().cpu().numpy()

    X_test_pos = torch.cat([test_data['gene'].x[test_data[('gene', 'associate', 'disease')].pos_edge_label_index[0]],
                            test_data['disease'].x[test_data[('gene', 'associate', 'disease')].pos_edge_label_index[1]]],
                           dim=1)
    X_test_neg = torch.cat([test_data['gene'].x[test_data[('gene', 'associate', 'disease')].neg_edge_label_index[0]],
                             test_data['disease'].x[test_data[('gene', 'associate', 'disease')].neg_edge_label_index[1]]],
                           dim=1)
    # X_test = torch.cat([X_test_pos, X_test_neg], dim=0).detach().cpu().numpy().reshape(-1, 1)
    X_test = torch.cat([X_test_pos, X_test_neg], dim=0).detach().cpu().numpy()

    pos_y_test = X_test_pos.new_ones(test_data[('gene', 'associate', 'disease')].pos_edge_label_index.size(1))
    neg_y_test = X_test_pos.new_zeros(test_data[('gene', 'associate', 'disease')].pos_edge_label_index.size(1))
    y_test = torch.cat([pos_y_test, neg_y_test], dim=0).detach().cpu().numpy()

    # Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]

    print("LR Accuracy:", accuracy_score(y_test, y_test_pred))
    print("LR AUC:", roc_auc_score(y_test, y_test_pred_proba))
    print("LR Precision:", average_precision_score(y_test, y_test_pred))


    # Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=128)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

    print("RF Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("RF AUC:", roc_auc_score(y_test, y_proba_rf))
    print("RF Precision:", average_precision_score(y_test, y_pred_rf))


    # XGBoost Classifier
    gb_model = GradientBoostingClassifier(n_estimators=256, learning_rate=0.01)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    y_proba_gb = gb_model.predict_proba(X_test)[:, 1]

    print("XG Accuracy:", accuracy_score(y_test, y_pred_gb))
    print("XG AUC:", roc_auc_score(y_test, y_proba_gb))
    print("XG Precision:", average_precision_score(y_test, y_pred_gb))

    # # Validation predictions
    # y_val_pred_proba = model.predict_proba(X_val)[:, 1]  # Probability of class 1
    # y_val_pred = model.predict(X_val)

    # # Validation
    # roc_auc = roc_auc_score(y_val, y_val_pred_proba)
    # accuracy = accuracy_score(y_val, y_val_pred)
    # ap = average_precision_score(y_val, y_val_pred)
    #
    # print(f"Validation ROC-AUC: {roc_auc:.4f}")
    # print(f"Validation Accuracy: {accuracy:.4f}")
    # print(f"Validation Precision: {ap:.4f}")
