python train.py file_name=../1_year_data/GData_train_0.csv model_record_path=../1_year_result/model_1/ file_record_path=../1_year_result/record_1/ method_name=mlp scaler_name=scaler_0.m pca=True pca_name=pca_0.m model_name=model_0/my_model
python test.py train_file_name=../1_year_data/GData_train_0.csv test_file_name=../1_year_data/GData_test_0.csv model_record_path=../1_year_result/model_1/ file_record_path=../1_year_result/record_1/ method_name=mlp scaler_name=scaler_0.m pca_name=pca_0.m model_name=model_0 record_name=result_0.csv
python analyse_result.py predict_label_file_name=mlp_result_0.csv true_label_file_name=GData_test_origin_0.csv model_record_path=../1_year_result/model_1/ file_record_path=../1_year_result/record_1/ method_name=mlp winner_number=3 record_name=result_0.txt
