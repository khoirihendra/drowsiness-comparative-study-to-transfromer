import numpy as np

# Load dataset yang sudah diekstrak
data = np.load('/kaggle/working/output/extracted_features/uta_rldd_features_seq30.npz')
subjects = data['subjects']
folds = data['folds']

for test_fold in range(1, 6):
    val_fold = (test_fold % 5) + 1
    
    test_sub = set(subjects[folds == test_fold])
    val_sub = set(subjects[folds == val_fold])
    train_sub = set(subjects[(folds != test_fold) & (folds != val_fold)])
    
    # Cek apakah ada irisan subjek
    leakage_train_test = train_sub.intersection(test_sub)
    leakage_val_test = val_sub.intersection(test_sub)
    
    print(f"Fold {test_fold} -> Train Subjects: {len(train_sub)} | Test Subjects: {len(test_sub)}")
    print(f"  └─ Overlap Train & Test: {len(leakage_train_test)} (Identity Leakage: {len(leakage_train_test) > 0})")
    print(f"  └─ Overlap Val & Test  : {len(leakage_val_test)} (Identity Leakage: {len(leakage_val_test) > 0})")
