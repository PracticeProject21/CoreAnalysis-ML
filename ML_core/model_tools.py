import torch
import numpy as np
import torch.nn.functional as f


##############################################################################
# Метрики для оценивания модели
def pixel_accuracy(predict, original):
    # accuracy = кол-во верных пикселей / кол-во пикселей
    with torch.no_grad():
        predict = torch.argmax(f.softmax(predict, dim=1), dim = 1)
        correct_pixels = torch.eq(predict, original).int()
        accuracy = float(correct_pixels.sum()) / float(correct_pixels.numel())
    return accuracy

def IoU(predict, original, number_of_class):
    #iou = объединение / пересечение
    eps = 0.00001
    with torch.no_grad():
        predict = torch.argmax(f.softmax(predict, dim=1), dim=1)

        # сделаем из масок одномерный тензор
        predict = predict.contiguous().view(-1)
        original = original.contiguous().view(-1)

        IoU = []
        for i in range(0, number_of_class):
            # посчитаем количество пикселей каждого класса
            # на predict и на original
            i_class_orig = (original == i).long()
            i_class_pred = (predict == i).long()

            if i_class_orig.sum().item() == 0:
                IoU.append(np.nan)
            else:
                intersection = torch.logical_and(i_class_orig,
                                                 i_class_pred).sum().float().item()
                union = torch.logical_or(i_class_orig,
                                                 i_class_pred).sum().float().item()
                IoU.append((intersection + eps)/(union + eps))

        return np.nanmean(IoU)
##############################################################################







def train_model(model, optimizer, scheduler,
                loss, metric, dataloaders, num_epochs):
    statistic_dict = {'pixel_accuracy':[], 'IoU':[],
                      'loss_train':[], 'loss_val':[]}


def prediction(model, image):
    pass