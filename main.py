import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
from self_balanced import train_model_chunk, self_learn
from load import load_all_data
from pre_process import pre_process_data, prepare_chunks, prepare_data,generate_classification_labels

# Check if GPU is available
if __name__ == '__main__':
    if tf.test.is_gpu_available():
        print("GPU is available")
        device = "/GPU:0"
    else:
        print("GPU not available, using CPU instead.")
        device = "/CPU:0"
    proportion_list = [0.2,0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    p_a = [0.05,0.05,0.05,0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    spectrum_type = 'StellarNetSpectrum'
    Intensity_all = load_all_data('data_config.json', spectrum_type)
    r_c, num_classes = 50, 9
    out_put = generate_classification_labels(r_c, num_classes)
    x_chunk_list_arranged, y_chunk_list_arranged = prepare_chunks(Intensity_all=Intensity_all, out_put=out_put)
    with tf.device(device):
        accuracy_list = []
        for i in range(1):
            if (i == 0):
                train_chunk_x = x_chunk_list_arranged[0]
                train_chunk_y = y_chunk_list_arranged[0]
            else:
                train_chunk_x = np.vstack((train_chunk_x, x_chunk_list_arranged[i]))
                train_chunk_y = np.vstack((train_chunk_y, y_chunk_list_arranged[i]))
            test_chunk_y = y_chunk_list_arranged[i+2]
            test_chunk_x = x_chunk_list_arranged[i+2]
            _, model = train_model_chunk(train_chunk_x, train_chunk_y, test_chunk_x, test_chunk_y)
            results_list = self_learn(model, train_chunk_x, train_chunk_y, test_chunk_x, test_chunk_y, proportion_list, p_a)
            accuracy_list.append(results_list)