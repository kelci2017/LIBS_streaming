from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical

def normalize_data(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    data_t = scaler.transform(data)
    return data_t.T

def my_filter(b, a, aa):
    return (b < aa) & (aa < a)

def filter_chunk(chunk_data, up_bound=0.85, low_bound=0.15):
    pca = PCA(n_components=2)
    components = pca.fit_transform(chunk_data)
    a = np.quantile(components[:,0], up_bound)
    b = np.quantile(components[:,0], low_bound)
    c = components[:,0]
    a_mask = my_filter(b, a, c)
    nnn = [ind[a_mask] for ind in np.indices(c.shape)]
    return chunk_data[nnn[0]], nnn[0]

def generate_classification_labels(r_c, num_classes):
    out_put = []
    for i in range(r_c * num_classes):
        class_label = i // r_c + 1
        out_put.append(class_label)
    out_put1 = np.array(out_put)
    return out_put1

def prepare_data(chunk_data_list, chunk_label_list, random_state=0):
    shuffled_data = []
    shuffled_labels = []
    for data, labels in zip(chunk_data_list, chunk_label_list):
        shuffled_data_i, shuffled_labels_i = shuffle(data, labels, random_state=random_state)
        shuffled_data.append(shuffled_data_i)
        shuffled_labels.append(shuffled_labels_i)

    Intensity_all = np.vstack(shuffled_data)
    Intensity_all = Intensity_all.reshape(Intensity_all.shape[0],33,66,1)
    out_put = np.concatenate(shuffled_labels)
    
    return Intensity_all, out_put

def prepare_chunks(Intensity_all, out_put, num_chunks=6, chunk_size=315, i_width=33, i_height=66, chunk_order=[4, 2, 0, 1, 3, 5]):
    y_one_hot = to_categorical(out_put)
    Intensity_all = Intensity_all.reshape(Intensity_all.shape[0],i_width,i_height,1)
    X = Intensity_all
    Y = y_one_hot
    x_chunk_list = []
    y_chunk_list = []
    for i in range(num_chunks):
        x_chunk_list.append(X[i*chunk_size:(i+1)*chunk_size])
        y_chunk_list.append(Y[i*chunk_size:(i+1)*chunk_size])

    x_chunk_list_arranged = [x_chunk_list[i] for i in chunk_order]
    y_chunk_list_arranged = [y_chunk_list[i] for i in chunk_order]
    
    return x_chunk_list_arranged, y_chunk_list_arranged