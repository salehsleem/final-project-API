# utils_hieroglyphs.py
import numpy as np


def get_hieroglyphs_classes():
    class_names = ['100', 'Among', 'Angry', 'Ankh', 'Aroura', 'At', 'Bad_Thinking', 'Bandage', 'Bee', 'Belongs',
                   'Birth', 'Board_Game', 'Book', 'Boy', 'Branch', 'Bread', 'Brewer', 'Builder', 'Bury', 'Canal',
                   'Cloth_on_Pole', 'Cobra', 'Composite_Bow', 'Cooked', 'Corpse', 'Dessert', 'Divide', 'Duck',
                   'Elephant', 'Enclosed_Mound', 'Eye', 'Fabric', 'Face', 'Falcon', 'Fingre', 'Fish', 'Flail',
                   'Folded_Cloth', 'Foot', 'Galena', 'Giraffe', 'He', 'Her', 'Hit', 'Horn', 'King', 'Leg',
                   'Length_Of_a_Human_Arm', 'Life_Spirit', 'Limit', 'Lion', 'Lizard', 'Loaf', 'Loaf_Of_Bread', 'Man',
                   'Mascot', 'Meet', 'Mother', 'Mouth', 'Musical_Instrument', 'Nile_Fish', 'Not', 'Now', 'Nurse',
                   'Nursing', 'Occur', 'One', 'Owl', 'Pair', 'Papyrus_Scroll', 'Pool', 'QuailChick', 'Reed', 'Ring',
                   'Rope', 'Ruler', 'Sail', 'Sandal', 'Semen', 'Small_Ring', 'Snake', 'Soldier', 'Star', 'Stick',
                   'Swallow', 'This', 'To_Be_Dead', 'To_Protect', 'To_Say', 'Turtle', 'Viper', 'Wall', 'Water', 'Woman',
                   'You']

    return class_names


def predict_hieroglyphs(model, processed_image, class_names):
    predicted_class = np.argmax(model.predict(processed_image))
    predicted_class_name = class_names[predicted_class]
    return predicted_class_name


