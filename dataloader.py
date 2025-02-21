import torch.utils.data as Data
import torch
import random
class MyDataset(Data.Dataset):
    def __init__(self, datasample, label1, label2, label3, label4, ec,diff_fingerprints, all_fingerprints, reaction_feature):
      self.datasample  = datasample
      self.diff_fingerprints = diff_fingerprints
      self.all_fingerprints = all_fingerprints
      self.dict = {'-':0}
      self.label1 = label1
      self.label2 = label2
      self.label3 = label3
      self.label4 = label4
      self.reaction_feature = reaction_feature
      self.ec = ec
      self.re_list = []
      self.label1_list = []
      self.label2_list = []
      self.label3_list = []
      self.label4_list = []
      self.ec_list = []
      self.reaction_list = []
    def get_batches(self, batch_size):
        combined_data = list(zip(self.datasample, self.diff_fingerprints, self.all_fingerprints, self.label1, self.label2, self.label3, self.label4, self.ec, self.reaction_feature))
        random.shuffle(combined_data)
        n_batches = (len(self.datasample) + (batch_size - 1)) // batch_size
        for idx in range(n_batches):
            self.re_list = []
            self.diff_fingerprints_list = []
            self.all_fingerprints_list = []
            self.label1_list = []
            self.label2_list = []
            self.label3_list = []
            self.label4_list = []
            self.ec_list = []
            self.reaction_list = []
            for i in range(batch_size):
                index = idx * batch_size + i
                if index >= len(self.datasample):
                    break
                re, diff_fingerprints, all_fingerprints, l1, l2, l3, l4, ec, rf = combined_data[index]
                self.re_list.append(re)
                self.diff_fingerprints_list.append(diff_fingerprints)
                self.all_fingerprints_list.append(all_fingerprints)
                self.label1_list.append(float(self.dict[l1]) if l1 == '-' else float(l1))
                self.label2_list.append(float(self.dict[l2]) if l2 == '-' else float(l2))
                self.label3_list.append(l3)
                self.label4_list.append(("protein", l4[:400]))
                self.ec_list.append(ec)
                self.reaction_list.append(rf)
            yield self.re_list, self.label1_list, self.label2_list, self.label3_list, self.label4_list, self.ec_list, self.diff_fingerprints_list, self.all_fingerprints_list, self.reaction_list

    def get_counter(self, batch_size):
        return (len(self.datasample) + (batch_size - 1)) // batch_size
