import pandas as pd
import os
import numpy as np
import json
import time
import pickle
from imageio import imread
from PIL import Image
import matplotlib.pyplot as plt

def process_json(json_path):
    image_age_list = []
    image_gender_list = []
    image_emotion_list = []
    image_id_list = []
    
    bad_json_list = []
    
    # for each json file in the json_path, load the json file and extract the attributes
    for file in sorted(os.listdir(json_path)):
        image_id = file.split('.')[0]
        
        if json.load(open(json_path+file)) == []:
            print(file+' is empty it will be ignored')
            bad_json_list.append(file)
            continue
            
        # print(file)
            
        attribute_data = json.load(open(json_path+file))[0]["faceAttributes"]
        
        image_id_list.append(image_id)
        image_gender_list.append(attribute_data['gender'])
        image_age_list.append(attribute_data['age'])
        
        # in the emotiion attribute data set only the highest value to 1 and all others to 0
        emotion_dict = attribute_data['emotion']
        # max_emotion = max(emotion_dict, key=emotion_dict.get)
        # for key in emotion_dict:
        #     if key == max_emotion:
        #         emotion_dict[key] = 1
        #     else:
        #         emotion_dict[key] = 0
        
        image_emotion_list.append(attribute_data['emotion'])
        
    pd.DataFrame(bad_json_list).to_csv('bad_json_list.csv')
    
    # create a dictionary with the extracted attributes
    json_dict = {
        'image id': image_id_list,
        'age': image_age_list,
        'gender': image_gender_list
        }
    
    # for each entry in emotion list, get the key and add it to the json_dict
    emotion_keys = set([key for emotion in image_emotion_list for key in emotion.keys()])
    
    # create a list of lists for each emotion key
    for key in sorted(emotion_keys):
        json_dict[key] = [emotion[key] for emotion in image_emotion_list]
        
    attribute_df = pd.DataFrame(json_dict)
    attribute_df.sort_values(by='image id', inplace=True)
    attribute_df.set_index('image id', inplace=True)
    attribute_df['gender'] = attribute_df['gender'].map({'male': 0.0, 'female': 1.0})
    
    # plot the sum of each of the keys in the emotion dictionary
    emotion_df = pd.DataFrame(image_emotion_list)
    print(emotion_df.sum())
    
    return attribute_df

def process_image(image_path):            
    processed_image = Image.open(image_path)
    # convert image to array
    image_vector = np.asarray(processed_image)            
    # flatten the image array to a single dimension
    image_vector = image_vector
    return image_vector


def pre_processing(dataset_path, json_path):
    # Load the images from the dataset_path
    # convert images to arrays
    # combine image ids with image vectors and add to a list
    # convert list to dataframe
    image_vector_list = []
    image_id_list = []
    dataset_dir_list = os.listdir(dataset_path)
    
    for dataset in sorted(dataset_dir_list):
        img_list = os.listdir(dataset_path+'/'+dataset)
        print(f'loading the images from {dataset}')
        images = sorted(os.listdir(dataset_path+'/'+dataset))
        for image in images:
            image_id_list.append(image.split('.')[0])
            image_vector_list.append(process_image(dataset_path+dataset+'/'+image))

            
    # convert list of image vectors to dataframe and add image id
    image_dict = { 'image id': image_id_list, 'image vector': image_vector_list }
    image_dataframe = pd.DataFrame(image_dict)
    image_dataframe.sort_values(by='image id', inplace=True)
    image_dataframe.set_index('image id', inplace=True)
    return image_dataframe
    
def compile_final_dataset(images_dataframe, json_path):
    # load the json files from the json_path and extract relevant attributes into a dataframe
    attribute_df = process_json(json_path)
    
    # add attribute dataframe to image dataframe
    final_dataframe = images_dataframe.merge(attribute_df, on='image id')

    # return the processed image dataset
    return final_dataframe

def main(type, resolution='128x128'):
    if type == 'sample':
        image_filepath = os.getcwd() + '/sample/sample_images/'
        json_filepath = os.getcwd() + '/sample/sample_json/'
    elif type == 'dataset':
        image_filepath = os.getcwd() + '/thumbnails128x128/'
        json_filepath = os.getcwd() + '/ffhq-features-dataset-master/json/'
    else:
        raise ValueError('Invalid type, please specify either test or train.')
            
    vectorized_images = pre_processing(image_filepath, json_filepath)
    processed_image_data_set = compile_final_dataset(vectorized_images, json_filepath)
        
    # save the processed image dataset to a pickle file
    pickle.dump(processed_image_data_set, open(f'processed_image_data_set_{resolution}.pkl', 'wb'))
    processed_image_data_set = []
    # load the processed image dataset from the pickle file (this is just proof that the pickle file works)
    processed_image_data_set = pickle.load(open(f'processed_image_data_set_{resolution}.pkl', 'rb'))
    print(f"image dimensions: {processed_image_data_set.loc['00000', 'image vector'].shape}")
    print(f"dataset shape: {processed_image_data_set.shape}")

    # confirming that bad data has been removed (should skip image 00036)
    # print(processed_image_data_set.iloc[[35]])
    # print(processed_image_data_set.iloc[[36]])
    print(processed_image_data_set.columns)

    processed_image_data_set.dropna(inplace=True)
    
if __name__ == '__main__':
    main('sample', 'sample')
    #main('dataset', '128x128')