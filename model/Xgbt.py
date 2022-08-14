from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score
import pandas as pd
import xgboost as xgb

NTHREAD=5

#Processing data
mol1 = pd.read_csv('../final_data/OncologyScreen/drug_f64.csv', header=None,low_memory=False)
mol3 = pd.read_csv('../final_data/OncologyScreen/cell_f64.csv', header=None,low_memory=False)
tag = pd.read_csv('../final_data/OncologyScreen/drug_combination_processed.csv', header=None,low_memory=False)
tag1 = pd.read_csv('../final_data/OncologyScreen/drug_combination_processed.csv')

drug1 = []
drug2 = []
cell = []
res = []
last_res = []
last_resY = []

c1 = mol1.shape[1]
c2 = mol1.shape[1]
c3 = mol3.shape[1]
sum = c1 + c2 + c3

for i in range(len(mol3[0])):
    tmp = []
    for j in range(0, c3):
        tmp.append(mol3[j][i])
    cell.append(tmp)

for i in range(len(mol1[0])):
    tmp = []
    for j in range(0, c1):
        tmp.append(mol1[j][i])
    drug1.append(tmp)
for i in range(len(mol1[0])):
    tmp = []
    for j in range(0, c2):
        tmp.append(mol1[j][i])
    drug2.append(tmp)

for i in range(1, len(tag[0])):
    tmp = [tag[7][i],tag[8][i],tag[9][i],tag[10][i]]
    res.append(tmp)

for i in range(len(res)):
    last_resY.append(int(res[i][3]))
    ttmp = []
    cell_tmp = cell[(int(res[i][2])) - 1]
    drug_tm1 = drug1[(int(res[i][0])) - 1]
    drug_tm2 = drug2[(int(res[i][1])) - 1]

    for k in range(len(drug_tm1)):
        ttmp.append(drug_tm1[k])

    for k in range(len(drug_tm2)):
        ttmp.append(drug_tm2[k])

    for k in range(len(cell_tmp)):
        ttmp.append(cell_tmp[k])

    last_res.append(ttmp)

def getBech_all():
    datax = []
    datay = []
    for i in range(len(last_resY)):
        datay.append([last_resY[i]])
        datax.append(last_res[i])
    return datax, datay

xx,yy = getBech_all()
df0=pd.DataFrame(xx)
print("data input success,len = ", sum)


#xgbt
# load data
fe = df0
df = tag1
target = df['synergistic'].values
X = fe.values
Y = target

#Here, xgbt uses the sklearn interface.
# Training OncologyScreen datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)
model = xgb.XGBClassifier(
            booster='gbtree',
            objective='multi:softmax',
            num_class=2,
            gamma=0.1,
            max_depth=7,
            min_child_weight=1,
            reg_lambda =0.01,
            subsample=0.9,
            colsample_bytree=0.9,
            learning_rate =0.1,
            seed = 30,
            nthread=4,
            random_state = 2,
            use_label_encoder=False,
)

'''
#Training DrugCombDB datasets
model = xgb.XGBClassifier(
    booster='gbtree',
    objective='multi:softmax',
    num_class=2,
    gamma=0.1,
    max_depth=9,
    min_child_weight=6,
    reg_lambda =0.1,
    subsample=0.9,
    colsample_bytree=0.8,
    learning_rate =0.1,
    seed = 30,
    nthread=10,
    random_state = 2,
    use_label_encoder=False,
)
'''

#load model
#model = pickle.load((open("xgbtOncology64.dat","rb")))
model.fit(X_train,Y_train)
#save model
#pickle.dump(model,open("xgbtOncology64_gai.dat","wb"))

y_pred = model.predict(X_test)
models = xgb.XGBClassifier(use_label_encoder=False)
kfold = KFold(n_splits=5, random_state=39,shuffle=True)
results = cross_val_score(models, X, Y, cv=kfold)
accuracy = cross_val_score(models, X, Y, scoring='accuracy', cv=kfold)
precision = cross_val_score(models, X, Y, scoring='precision', cv=kfold)
recall = cross_val_score(models, X, Y, scoring='recall', cv=kfold)
f1_score = cross_val_score(models, X, Y, scoring='f1', cv=kfold)
auc_roc = cross_val_score(models, X, Y, scoring='roc_auc', cv=kfold)
auc_pr = cross_val_score(models,X, Y,scoring='average_precision',cv=kfold)
print('Accuracy:%.6f:' % accuracy.mean(),
    'Recall:%.6f:'% recall.mean(),
    'AUC-ROC:%.6f:'%auc_roc.mean(),
    'AUC-PR:%.6f:'%auc_pr.mean(),
    'Precesion:%.6f:'%precision.mean(),
    'F1-score:%.6f:'%f1_score.mean()
      )

