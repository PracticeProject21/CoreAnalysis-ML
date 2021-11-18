import torch
from .tools import model_predict, text_generation


name_model_day = ''
name_model_ultra = 'DeepLabV3Plus_resnet18_ultraviolet.pth'

#DAY_model = torch.load(name_model_day)
ULTRA_model = torch.load(name_model_ultra)

def ML_part(PIL_image, label):
    if label == 'ultraviolet':
        predict_mask = model_predict(ULTRA_model, PIL_image)
        result = text_generation(predict_mask, label)
        return result
    else:
        pass
        # label == 'daylight'
        #predict_mask = model_predict(DAY_model, PIL_image)
        #result = text_generation(predict_mask, label)
        #return result

