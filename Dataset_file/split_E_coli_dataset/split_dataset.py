import pandas as pd
import pickle
# 读取数据
file_path = '/data/stu1/lyl_pycharm_project/Km_pre/Data_process/Process_dataset_Ecoil/E_coli_concat_contain_reaction_fingerprints_reaction_vector.pkl'  # 替换为你的路径
df = pd.read_pickle(file_path)

# 按照 'wildtype' 和 'mutant' 拆分 DataFrame
df_wildtype = df[df['Enzyme Variant'].str.contains('wildtype', case=False, na=False)]
df_mutant = df[df['Enzyme Variant'].str.contains('mutant', case=False, na=False)]

# 重新设置索引
df_wildtype_reset = df_wildtype.reset_index(drop=True)
df_mutant_reset = df_mutant.reset_index(drop=True)
# -----
df_wildtype_reset['Enzyme type'] = 'wildtype'
df_mutant_reset['Enzyme type'] = 'mutant'
# 计算训练集和测试集的行数
train_num = int(len(df) * 0.8)
test_num = len(df) - train_num

# 计算wildtype和mutant的比例
wildtype_ratio = len(df_wildtype_reset) / len(df)
mutant_ratio = len(df_mutant_reset) / len(df)

# 根据比例从每个DataFrame中随机抽样出测试集
wildtype_test_num = int(test_num * wildtype_ratio)
mutant_test_num = test_num - wildtype_test_num  # 剩余部分分配给mutant

# 从wildtype和mutant中分别抽取测试集
df_wildtype_test = df_wildtype_reset.sample(n=wildtype_test_num, random_state=42)
df_mutant_test = df_mutant_reset.sample(n=mutant_test_num, random_state=42)

# 将剩余的部分作为训练集
df_wildtype_train = df_wildtype_reset.drop(df_wildtype_test.index)
df_mutant_train = df_mutant_reset.drop(df_mutant_test.index)

# 合并训练集和测试集
train_set = pd.concat([df_wildtype_train, df_mutant_train], ignore_index=True)
test_set = pd.concat([df_wildtype_test, df_mutant_test], ignore_index=True)

# 保存训练集和测试集
# train_set.to_pickle('E_coli_train_set.pkl')
# test_set.to_pickle('E_coli_test_set.pkl')
with open('E_coli_train_set_protocol4.pkl', 'wb') as f:
    pickle.dump(train_set, f, protocol=4)  #
with open('E_coli_test_set_protocol4.pkl', 'wb') as f:
    pickle.dump(test_set, f, protocol=4)  #
# 可选：查看训练集和测试集的前几行
print("Train Set (with reset index):")
print(train_set.head())

print("Test Set (with reset index):")
print(test_set.head())
