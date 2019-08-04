import sys


method_name = 'pairwise_reference'
total_record_number = 4
test_number = 88
winner_number = 3

for record_number in range(1, total_record_number+1):
    record_name = 'execute_{0}.sh'.format(record_number)
    file = open(record_name,'w')
    for i in range(test_number):
    # i = 15
        """
        train parameter list:
        file_name
        model_record_path
        file_record_path
        method_name
        scaler_name
        pca_or_not
        pca_name
        model_name
        """
        file.write('python reference.py file_name=../1_year_data/GData_train_{0}.csv model_record_path=../1_year_result/model_{1}/ file_record_path=../1_year_result/record_{2}/ method_name={3} scaler_name=scaler_{4}.m pca=True pca_name=pca_{5}.m model_name=reference_model_{6}/my_model\n'.format(i, record_number, record_number, method_name, i, i, i))
        """
        train parameter list:
        file_name
        model_record_path
        file_record_path
        method_name
        scaler_name
        pca_or_not
        pca_name
        model_name
        """
        file.write('python train.py file_name=../1_year_data/GData_train_{0}.csv model_record_path=../1_year_result/model_{1}/ file_record_path=../1_year_result/record_{2}/ method_name={3} scaler_name=scaler_{4}.m pca=True pca_name=pca_{5}.m model_name=model_{6}/my_model reference_model_name=reference_model_{6}/my_model\n'.format(i, record_number, record_number, method_name, i, i, i))
        """
        test parameter list:
        file_name
        model_record_path
        file_record_path
        method_name
        scaler_name
        pca_name
        model_name
        record_name
        """

        file.write('python test.py train_file_name=../1_year_data/GData_train_{0}.csv test_file_name=../1_year_data/GData_test_{0}.csv model_record_path=../1_year_result/model_{1}/ file_record_path=../1_year_result/record_{2}/ method_name={3} scaler_name=scaler_{4}.m pca_name=pca_{5}.m model_name=model_{6} record_name=result_{7}.csv\n'.format(i, record_number, record_number, method_name, i, i, i, i))
        """
        analyse parameter list:
        predict_label_file_name
        true_label_file_name
        model_record_path
        file_record_path
        method_name
        winner_number
        """
        file.write('python analyse_result.py predict_label_file_name={0}_result_{1}.csv true_label_file_name=GData_test_origin_{2}.csv model_record_path=../1_year_result/model_{3}/ file_record_path=../1_year_result/record_{4}/ method_name={5} winner_number={6} record_name=result_{7}.txt\n'.format(method_name, i, i, record_number, record_number, method_name, winner_number, i))

    file.close()
