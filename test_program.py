import pickle
import sys
import timeit
import math
import numpy as np
import torch
from sklearn.metrics import mean_squared_error,r2_score
from scipy.stats import pearsonr
from tqdm import tqdm
class Tester(object):
    def __init__(self, device):
        self.device = device
    def test(self, model, dataset):
        test_loder = dataset.get_batches(batch_size=1)
        n_batches = dataset.get_counter(batch_size=1)
        N = 0
        SAE = 0  # sum absolute error.
        testY, testPredict = [], []
        for i, batch in tqdm(enumerate(test_loder), total=n_batches, unit="batch"):
            N = N  + 1
            test_sentences = batch[0]
            test_tp = torch.tensor(batch[1]).to(self.device)
            test_ph = torch.tensor(batch[2]).to(self.device)
            test_og = batch[3]
            test_es = batch[4]
            test_km = torch.FloatTensor(batch[5]).to(self.device)
            test_diff_fingerprints = torch.FloatTensor(batch[6]).to(self.device)
            test_all_fingerprints = torch.FloatTensor(batch[7]).to(self.device)
            test_reaction_vectors = torch.FloatTensor(batch[8]).to(self.device)
            cv_km = test_km+1e-6
            model.eval()
            with torch.no_grad():
                pv_km =  model(test_sentences, test_tp, test_ph, test_og, test_es, test_diff_fingerprints, test_all_fingerprints, test_reaction_vectors)
            correct_values = cv_km.to('cpu').data.numpy()
            predicted_values = pv_km[0].to('cpu').data.numpy()
            correct_values = math.log10(correct_values)
            SAE += np.abs(predicted_values-correct_values)
            testY.append(correct_values)
            testPredict.append(predicted_values)
        MAE = SAE / N  # mean absolute error.
        rmse = np.sqrt(mean_squared_error(testY, testPredict))
        corr = self.pearson_correlation_coefficient(data1=testY, data2=testPredict)
        r2 = r2_score(testY, testPredict)
        return MAE, rmse, r2, corr

    def save_MAEs(self, MAEs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, MAEs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

    def pearson_correlation_coefficient(self, data1, data2):
        data1 = np.array(data1)
        data2 = np.array(data2)
        corr, _ = pearsonr(data1, data2)
        return corr
