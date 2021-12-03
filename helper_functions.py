import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU, Precision, Recall, Accuracy
import numpy as np
import matplotlib.pyplot as plt



# display one "rgbs" image and one "masks" image. Prediction image is optional.
def display_images(rgb, mask, *preds):
    plt.figure(figsize=(15,20))
    plt.subplot(5,3,1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('RGB Image')
    plt.imshow(tf.keras.preprocessing.image.array_to_img(rgb))

    plt.subplot(5,3,2)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('Mask Image')
    plt.imshow(mask, cmap='gray')
    
    for pred in preds:
        plt.subplot(5,3,3)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.title('Predicted Image')
        plt.imshow(tf.keras.preprocessing.image.array_to_img(pred), cmap='gray')
    
    plt.show()


# convert original PanNuke labeled mask to a binary mask
def convert_to_binary(mask):
    
    new_mask = np.zeros((256, 256))
    
    for i, dims in enumerate(mask):
        for x, pixels in enumerate(dims):
            if len(np.where(pixels>=1)[0])==0 or np.where(pixels>=1)[0][0] in [2, 4, 5]:
                new_mask[i][x] = 0
            else:
                new_mask[i][x] = 1
    
    return new_mask


# convert all arrays in full dataset to binary masks and return as an array
def convert_all_arrays_to_binary(masks):
    
    new_masks = [convert_to_binary(mask) for mask in masks]
    new_masks = np.array(new_masks)
    
    return new_masks


# reshape input image and normalize pixels to [0, 1]
@tf.function
def normalize(image, mask):
    
    input_image = tf.image.resize(image, (128, 128))
    input_mask = tf.image.resize(mask, (128, 128))
    
    return input_image, input_mask


# data augmentation mean to be performed on random images in dataset
@tf.function
def augment(image, mask):
    
    input_image, input_mask = normalize(image, mask)
    
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
        
    if tf.random.uniform(()) > 0.7:
        input_image = tf.image.adjust_brightness(input_image, 0.3)
        # input_image = tf.image.adjust_saturation(input_image, -2)
    
    return input_image, input_mask


# convert model's inference to a mask that matches original labels and can be displayed
def create_mask(pred_mask, img_size):
    pred_mask[pred_mask>=0.5] = 1.0
    pred_mask[pred_mask<0.5] = 0.0
    pred_mask = tf.image.resize(pred_mask, img_size)
    pred_mask = tf.cast(pred_mask, np.float64)
    return pred_mask[0]


# calculate accuracy, precision, recall, and MeanIOU of true mask vs prediction mask
def calculate_metrics(model, pred_dataset, image_size):
    
    i = 0
    a = Accuracy()
    p = Precision()
    r = Recall()
    m = MeanIoU(2)
    
    for image, mask in pred_dataset:
        
        pred = model.predict(image)
        pred = create_mask(pred, image_size)
        pred_mask = pred.numpy()
        mask = tf.image.resize(mask, image_size)
        mask = np.squeeze(mask)
        
        a.update_state(mask, pred_mask)
        p.update_state(mask, pred_mask)
        r.update_state(mask, pred_mask)
        m.update_state(mask, pred_mask)
        
        i += 1
       
    acc = a.result().numpy()
    prec = p.result().numpy()
    recall = r.result().numpy()
    iou = m.result().numpy()

    
    return acc, prec, recall, iou