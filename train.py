import pandas as pd
from ERFPPK import *
from dataloader import *
import os
from test_program import *

def fix_seed(seed):  # 确保在整个代码中，所有的随机数生成器都受到相同的种子的影响，从而使实验结果可重复。
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
if torch.cuda.is_available():
    device = 'cuda'
    print('Device is {}'.format(device))
else:
    device = 'cpu'
    print('Device is {}'.format(device))
fix_seed(3407)
# data
# data
file_path_1 = '/data/stu1/lyl_pycharm_project/Km_pre/Data_process/split_dataset/split_E_coli_dataset/E_coli_train_set.pkl'
file_path_2 = '/data/stu1/lyl_pycharm_project/Km_pre/Data_process/split_dataset/split_E_coli_dataset/E_coli_test_set.pkl'
file_path_3 = '/data/stu1/lyl_pycharm_project/Km_pre/Data_process/split_dataset/split_Human_dataset/Human_train_set.pkl'
file_path_4 = '/data/stu1/lyl_pycharm_project/Km_pre/Data_process/split_dataset/split_Human_dataset/Human_test_set.pkl' # 请替换为你的 CSV 文件路径
#dict
E_coli_train_set = pd.read_pickle(file_path_1)
E_coli_test_set = pd.read_pickle(file_path_2)
Human_train_set = pd.read_pickle(file_path_3)
Human_test_set = pd.read_pickle(file_path_4)


# data_split
train_df, test_df =pd.concat([E_coli_train_set, Human_train_set], ignore_index=True), pd.concat([E_coli_test_set, Human_test_set], ignore_index=True)

##

# Train_dataset
# Index(['rxn_smiles', 'ec', 'source', 'label1', 'label2', 'label3', 'label4'], dtype='object')
train_sentences = train_df["R_smiles"]
train_diff_fingerprints = train_df['differ_reaction_vector']
train_all_fingerprints = train_df['all_reaction_vector']
train_tp = train_df["Temperature"]
train_ph = train_df["PH"]
train_og = train_df["Organism"]
train_es = train_df["E_sequence"]
train_km = train_df["Km"]
train_rv = train_df["Reaction_vector"]
# Test_dataset
test_sentences = test_df["R_smiles"]
test_diff_fingerprints = test_df['differ_reaction_vector']
test_all_fingerprints = test_df['all_reaction_vector']
test_tp = test_df["Temperature"]
test_ph = test_df["PH"]
test_og = test_df["Organism"]
test_es = test_df["E_sequence"]
test_km = test_df["Km"]
test_rv = test_df["Reaction_vector"]
#My dataset
train_dataset = MyDataset(train_sentences,  train_tp, train_ph, train_og, train_es, train_km, train_diff_fingerprints, train_all_fingerprints, train_rv)
test_dataset = MyDataset(test_sentences,  test_tp, test_ph, test_og, test_es, test_km, test_diff_fingerprints, test_all_fingerprints, test_rv)


# model
model = ERFPPK(dim=256, device=device).to(device)
num_parameters_group2 = 0
count = 0

# 设置优化器
optimal_function = torch.optim.Adam(model.parameters(), lr=0.0002)

loss_function = nn.MSELoss(reduction='mean')
"""Set a model."""
print('Training...')
MAEs = ('Epoch\tTime(sec)\tRMSE_train\tR2_train\tMAE_dev\tMAE_test\tRMSE_dev\tRMSE_test\tR2_dev\tR2_test\tPearson_dev\tPearson_test')
print(MAEs)
# -------
trainCorrect = []
trainPredict = []
patience = 5  # Number of epochs with no improvement after which training will be stopped
best_loss = np.inf  # Initialize the best loss as infinity
patience_counter = 0  # Counter for how many epochs since last improvement
test_program = Tester(device)
start = timeit.default_timer()
best_p_test = float('-inf')
for epoch in range(1, 50):
    # if epoch==0:
    #     continue

    counter = 0
    loss_total = 0
    model.train()  # Set the model to training mode
    trainloader = train_dataset.get_batches(batch_size=6)
    n_batches = train_dataset.get_counter(batch_size=6)
    for i, batch in tqdm(enumerate(trainloader), total=n_batches, unit="batch"):
        train_sentences = batch[0]
        train_tp = torch.tensor(batch[1]).to(device)
        train_ph = torch.tensor(batch[2]).to(device)
        train_og = batch[3]
        train_es = batch[4]
        train_km = torch.FloatTensor(batch[5]).to(device)
        train_diff_fingerprints = torch.FloatTensor(batch[6]).to(device)
        train_all_fingerprints = torch.FloatTensor(batch[7]).to(device)
        train_reaction_vectors = torch.FloatTensor(batch[8]).to(device)
        # ---
        cv_km = torch.log10(train_km + 1e-6)
        counter += 1
        model.zero_grad()  # Clear the gradients

        pv_km = model(train_sentences, train_tp, train_ph, train_og, train_es, train_diff_fingerprints,
                      train_all_fingerprints, train_reaction_vectors).reshape(-1) # Forward pass
        loss = loss_function(pv_km, cv_km)  # Calculate loss
        loss.backward()  # Backward pass
        optimal_function.step()  # Update weights

        correct_values = cv_km.to('cpu').data.numpy()
        predicted_values = pv_km.to('cpu').data.numpy()
        loss_total += loss.to('cpu').data.numpy()
        trainCorrect.extend(correct_values)
        trainPredict.extend(predicted_values)
    rmse_train = np.sqrt(mean_squared_error(trainCorrect, trainPredict))
    r2_train = r2_score(trainCorrect, trainPredict)
    # MAE_dev, RMSE_dev, R2_dev, p_dev = Test.test(model=k_cat_predict, dataset=dataset_dev)
    MAE_dev, RMSE_dev, R2_dev, p_dev = 0, 0, 0, 0
    MAE_test, RMSE_test, R2_test, p_test = test_program.test(model, test_dataset)
    end = timeit.default_timer()
    time = end - start
    MAEs = [epoch, time, rmse_train, r2_train, MAE_dev,
            MAE_test, RMSE_dev, RMSE_test, R2_dev, R2_test, p_dev, p_test]
    print('\t'.join(map(str, MAEs)))
    if p_test > best_p_test:
        best_p_test = p_test
        model_info = {
            'km_predict':model.state_dict(),
            'MAE_test':MAE_test,
            'RMSE_test':RMSE_test,
            'Pearson:':best_p_test,
            'R_2': R2_test}
        torch.save(model_info, f'Km_parameter.pth')
    MAEs = [epoch, time, rmse_train, r2_train, MAE_dev,
            MAE_test, RMSE_dev, RMSE_test, R2_dev, R2_test, p_dev, p_test]
    print('\t'.join(map(str, MAEs)))

