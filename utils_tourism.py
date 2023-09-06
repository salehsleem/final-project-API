# utils_tourism.py
from PIL import Image
import numpy as np
from googletrans import Translator
import pandas as pd


def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Resize to the model's input size
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)


def get_tourism_prediction(processed_image, model, data_dict, class_names, lang):
    predicted_class = np.argmax(model.predict(processed_image))

    # Convert class index to class name
    predicted_class_name = class_names[predicted_class]
    final_predict = data_dict[predicted_class_name]
    # Create a Translator object
    translator = Translator()

    # Text to be translated
    text_to_translate = final_predict
    # Translate the text to another language
    if lang != 'en':
        translated_text = translator.translate(text_to_translate, src='en',
                                               dest=lang)
    else:
        translated_text = final_predict

    return translated_text, predicted_class_name


def get_tourism_data():
    # Load the Excel file
    xlsx_path = 'data_set.xlsx'
    data = pd.read_excel(xlsx_path)

    # Assuming the Excel file has columns named "key_column" and "value_column"
    key_column = "exhibits"
    value_column = "Text in English"

    # Convert the data to a dictionary
    data_dict = dict(zip(data[key_column], data[value_column]))
    return data_dict


def get_tourism_classes():
    class_names = ['10_The_HolyQuran', '11_King_Thutmose_III', '12_King_Fouad_I', '13_theVizier_Paser',
                   '14_Sphinxof_theking_Amenemhat_III', '15_Amun_Ra_Kingof_theGods', '16_Nazlet_Khater_Skeleton',
                   '17_Pen_Menkh_TheGovernerOf_Dendara', '18_TheCoffinOf_Lady_Isis', '19_CoffinOf_Nedjemankh',
                   '1_the_female_peasent', '20_TheCoffinOf_Sennedjem', '21_A_silo', '22_Captives_statuettes',
                   '23_Chair_from_the_tomb_of_Queen_Hetepheres', '24_Maat', '25_Mahalawi_water_ewers',
                   '26_Mamluk_Lamps',
                   '27_Khedive_Ismail', '28_Mohamed_Talaat_Pasha_Harb', '29_Model_of_building', '2_statue_ofthe_sphinx',
                   '30_Muhammad_Ali_Pasha', '31_Puplit _of_the_Mosque_of_Abu_Bakr_bin_Mazhar',
                   '32_The_Preist_Psamtik_seneb', '33_The_Madrasaa_and_Mosque_of_Sultan_Hassan', '34_Wekalet_al-Ghouri',
                   '35_The_birth_of_Isis', '36_King_Akhenaten', '37_The_Kiswa_Covering_of_holy_Kaaba',
                   '38_AQueen_in_the_form_of_the_Sphinx', '39_Purification_with_water', '3_Hassan_Fathi',
                   '40_Mashrabiya', '41_Astrolabe', '42_Baker', '43_The_Protective_Godesses', '44_Miller',
                   '45_Hapi_The_Scribe', '46_Thoth', '47_Ottoman_Period_Carpet', '48_Stela_of_King_Qaa',
                   '49_Zainab_Khatun_house', '4_Royal_Statues', '50_God_Nilus', '5_Greek_Statues', '6_Khonsu',
                   '7_Ra_Horakhty', '8_Senenmut', '9_Box_ofthe_Holy Quran', 'Akhenaten', 'Bent pyramid for senefru',
                   'Colossal Statue of Ramesses II', 'Colossoi of Memnon', 'Goddess Isis with her child', 'Hatshepsut',
                   'Hatshepsut face', 'Khafre Pyramid', 'Mask of Tutankhamun', 'Nefertiti', 'Pyramid_of_Djoser',
                   'Ramessum', 'Ramses II Red Granite Statue', 'Statue of King Zoser',
                   'Statue of Tutankhamun with Ankhesenamun', 'Temple_of_Isis_in_Philae', 'Temple_of_Kom_Ombo',
                   'The Great Temple of Ramesses II', 'amenhotep iii and tiye', 'bust of ramesses ii',
                   'menkaure pyramid', 'sphinx']

    return class_names
