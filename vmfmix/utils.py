# -*- coding: utf-8 -*-
import numpy as np

from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.special import iv


def cluster_acc(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    total = 0
    for i in range(len(ind[0])):
        total += w[ind[0][i], ind[1][i]]
    return total * 1.0 / Y_pred.size, w


def predict(x, mu, k, pi, n_cluster):

    yita_c = np.exp(np.log(pi[np.newaxis, :]) + blmm_pdfs_log(x, k, mu, n_cluster))

    yita = yita_c
    return np.argmax(yita, axis=1)


def blmm_pdfs_log(x, ks, mus, n_cluster):

    VMF = []
    for c in range(n_cluster):
        VMF.append(blmm_pdf_log(x, mus[c:c+1, :], ks[c]).reshape(-1, 1))
    return np.concatenate(VMF, 1)


def blmm_pdf_log(x, mu, k):

    D = x.shape[len(x.shape) - 1]
    pdf = (D / 2 - 1) * np.log(k) - (D / 2) * np.log(2 * np.pi) - np.log(iv(D / 2 - 1, k)) + x.dot(mu.T * k)
    return pdf


def calculate_mix(a, b, K):

    lambda_bar = a / (a + b)
    pi = np.zeros(K,)
    for i in range(K):
        temp_temp = 1
        for j in range(i):
            temp = 1 - (lambda_bar[j])
            temp_temp = temp_temp * temp
        pi[i] = lambda_bar[i] * temp_temp
    return pi


def d_besseli(nu, kk):

    try:
        bes = iv(nu + 1, kk) / (iv(nu, kk) + np.exp(-700)) + nu / kk
        assert (min(np.isfinite(bes)))
    except:
        bes = np.sqrt(1 + (nu**2) / (kk**2))

    return bes


def d_besseli_low(nu, kk):

    try:
        bes = iv(nu + 1, kk) / (iv(nu, kk) + np.exp(-700)) + nu / kk
        assert (min(np.isfinite(bes)))
    except:
        bes = kk / (nu + 1 + np.sqrt(kk**2 + (nu + 1)**2)) + nu / kk

    return bes


def caculate_pi(model, N, T):

    resp = np.zeros((N, T))
    resp[np.arange(N), model.labels_] = 1
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    pi = (nk / N)[np.newaxis, :]
    return pi


def log_normalize(v):
    ''' return log(sum(exp(v)))'''
    log_max = 100.0
    if len(v.shape) == 1:
        max_val = np.max(v)
        log_shift = log_max - np.log(len(v)+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift))
        log_norm = np.log(tot) - log_shift
        v = v - log_norm
    else:
        max_val = np.max(v, 1)
        log_shift = log_max - np.log(v.shape[1]+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift[:, np.newaxis]), 1)

        log_norm = np.log(tot) - log_shift
        v = v - log_norm[:, np.newaxis]

    return v, log_norm

