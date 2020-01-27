import os
import numpy as np

def get_fils(sub_path):
    images_files = []
    for subpath, subdirs, files in os.walk(sub_path):
        for name in files:
            images_files.append(os.path.join(subpath, name))
    return images_files

def yield_mb(files, batch_size=64, shuffle=False):
    if shuffle:
        _ = np.random.shuffle(files)
    for i in range(len(files)//batch_size):
        yield files[i*batch_size:(i+1)*batch_size]


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def categorical_label_from_full_file_name(files, class_indices):
    base_name = [i.split("/")[-2] for i in files]
    image_class = [class_indices[i] for i in base_name]
    return to_categorical(image_class, num_classes=len(class_indices))

def data_generator(batch_files, shape, class_indices, data_format = "channels_first", bounds=(0,1)):
    from PIL import Image
    
    batch_Y = categorical_label_from_full_file_name(batch_files, class_indices)
    batch_X = []

    for idx, input_path in enumerate(batch_files):
        #if input_path.split('.')[-1] == 'png':
        #    image = Image.open(input_path).convert('RGB')
        #else:
        #    image = Image.open(input_path)
        image = Image.open(input_path).convert('RGB')
        image = image.resize(shape)
        image = np.asarray(image, dtype=np.float32)
        image = image[:, :, :3]
        if data_format == "channels_first":
            image = np.transpose(image, (2, 0, 1))
        image = image / 255 * (bounds[1] - bounds[0]) + bounds[0]
        batch_X += [image]
        
    X = np.array(batch_X)
    Y = np.array(batch_Y)
        
    return (X, Y)

if __name__ == "__main__":

    path = '/media/dl/7a4fb85e-b20b-4ac9-bcd4-f7e0b49e0b00/rs-data/class/AID'
    
    train_path = os.path.join(path,'train')
    val_path = os.path.join(path,'val')
    classes = os.path.join(path,'classes.txt')
    
    train_files = get_fils(train_path)
    val_files = get_fils(val_path)
    
    class_indices = {}
    for line in open(classes).readlines():
        key,value = line.split(',')[0], int(line.split(',')[1])
        class_indices[key] = value
    
    batch_files = yield_mb(train_files, 16, shuffle=True).__next__()
    
    categorical_label_from_full_file_name(batch_files, class_indices)
    data, label = data_generator(batch_files, (224,224), class_indices)
