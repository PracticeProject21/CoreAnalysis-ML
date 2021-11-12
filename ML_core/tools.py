import torch
import numpy as np
import torch.nn.functional as f
from tqdm import tqdm

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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


#############################################################################
# обучение модели
def train_model(model, N_classes, model_name,
                optimizer, scheduler, loss,
                train_dl, val_dl, num_epochs,
                pixel_accuracy = pixel_accuracy,
                IoU = IoU,
                patch=False, device=DEVICE):
    statistic_dict = {'pixel_accuracy_val':[], 'IoU_val':[],
                      'pixel_accuracy_train': [], 'IoU_train': [],
                      'loss_train':[], 'loss_val':[]}
    min_loss = np.inf
    decrease = 0 # будем вести счетчик падения loss-функции
    device = model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch № {epoch+1}')
        for phase in ['training', 'validation']:
            running_loss = 0
            running_iou = 0
            running_accuracy = 0
            if phase == 'training':

                model.train()
                for img, mask in tqdm(train_dl):
                    if patch:
                        ## реализовать разбивку
                        pass
                    #img = img.to(device)
                    #mask = mask.to(device)

                    optimizer.zero_grad()
                    predict = model(img)
                    loss_value = loss(predict, mask)
                    loss_value.backward()
                    optimizer.step()
                    scheduler.step()

                    running_loss += loss_value.item()
                    running_accuracy += pixel_accuracy(predict, mask)
                    running_iou += IoU(predict, mask, number_of_class=N_classes)

                # подсчет статистик для тренировочной фазы
                loss_value_mean = running_loss / len(train_dl)
                accuracy_value_mean = running_accuracy / len(train_dl)
                iou_value_mean = running_iou / len(train_dl)
                statistic_dict['loss_train'].append(loss_value_mean)
                statistic_dict['pixel_accuracy_train'].append(accuracy_value_mean)
                statistic_dict['IoU_train'].append(iou_value_mean)

                print(f'train-loss: {loss_value_mean}',
                      f'train-accuracy: {accuracy_value_mean}',
                      f'train-iou: {iou_value_mean}', sep='\n')
            else:
                model.eval()
                with torch.no_grad():
                    for img, mask in tqdm(val_dl):
                        if patch:
                            ## реализовать разбивку
                            pass
                        img = img.to(device)
                        mask = mask.to(device)

                        predict = model(img)
                        loss_value = loss(predict, mask)

                        running_loss += loss_value.item()
                        running_accuracy += pixel_accuracy(predict, mask)
                        running_iou += IoU(predict, mask, number_of_class=N_classes)

                    # подсчет статистик для тренировочной фазы
                loss_value_mean = running_loss / len(train_dl)
                accuracy_value_mean = running_accuracy / len(train_dl)
                iou_value_mean = running_iou / len(train_dl)
                statistic_dict['loss_val'].append(loss_value_mean)
                statistic_dict['pixel_accuracy_val'].append(accuracy_value_mean)
                statistic_dict['IoU_val'].append(iou_value_mean)

                print(f'validation-loss: {loss_value_mean}',
                      f'validation-accuracy: {accuracy_value_mean}',
                      f'validation-iou: {iou_value_mean}', sep='\n')

                if loss_value_mean < min_loss:
                    decrease += 1
                    if decrease % 3 == 0:
                        print('saving model ...')
                        torch.save(model,
                                   f'{model_name}_{iou_value_mean}iou_{accuracy_value_mean}acc')

    return statistic_dict
def prediction(model, image):
    pass