import torch
import numpy as np
from PIL import Image
import cv2
import torch.nn.functional as f
from tqdm import tqdm
import torchvision.transforms as tt

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
                patch=False):
    statistic_dict = {'pixel_accuracy_val':[], 'IoU_val':[],
                      'pixel_accuracy_train': [], 'IoU_train': [],
                      'loss_train':[], 'loss_val':[]}
    max_iou = -1

    for epoch in range(num_epochs):
        print(f'Epoch № {epoch+1}', flush=True)
        for phase in ['training', 'validation']:
            running_loss = 0
            running_iou = 0
            running_accuracy = 0
            if phase == 'training':

                model.train()
                for img, mask in tqdm(train_dl):

                    if patch:
                        # делаем один большой батч из патчированных изображений
                        b_size, p_size, c, h, w = img.shape
                        img = img.view(-1, c, h, w)

                        b_size, p_size, h, w = mask.shape
                        mask = mask.view(-1, h, w)
                    # img = img.to(device)
                    # mask = mask.to(device)

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
                model.train(False)
                model.eval()
                with torch.no_grad():
                    for img, mask in tqdm(val_dl):
                        if patch:
                            b_size, p_size, c, h, w = img.shape
                            img = img.view(-1, c, h, w)

                            b_size, p_size, h, w = mask.shape
                            mask = mask.view(-1, h, w)
                        # img = img.to(device)
                        # mask = mask.to(device)

                        predict = model(img)
                        loss_value = loss(predict, mask)

                        running_loss += loss_value.item()
                        running_accuracy += pixel_accuracy(predict, mask)
                        running_iou += IoU(predict, mask, number_of_class=N_classes)

                    # подсчет статистик для тренировочной фазы
                loss_value_mean = running_loss / len(val_dl)
                accuracy_value_mean = running_accuracy / len(val_dl)
                iou_value_mean = running_iou / len(val_dl)
                statistic_dict['loss_val'].append(loss_value_mean)
                statistic_dict['pixel_accuracy_val'].append(accuracy_value_mean)
                statistic_dict['IoU_val'].append(iou_value_mean)

                print(f'validation-loss: {loss_value_mean}',
                      f'validation-accuracy: {accuracy_value_mean}',
                      f'validation-iou: {iou_value_mean}', sep='\n')

                if iou_value_mean > max_iou:
                    max_iou = iou_value_mean
                    print('saving model ...')
                    torch.save(model,
                               f'{model_name}.pth')

    return statistic_dict


#######################################################
# функция для предсказания
#######################################################
def model_predict(model, img, h_tf = 768, w_tf = 512):
    # изображение в формате np или PIL
    # img = Image.fromarray(img)
    # разобаться с ошибкой byte

    # сохраним исходный размер
    w, h = img.size

    # предобработка изображения
    need_tf = tt.Compose([
        tt.Resize([h_tf, w_tf]),
        tt.ToTensor(),
        tt.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])
    transformed_img = need_tf(img)

    with torch.no_grad():    # нахождение маски
        pred = model.predict(transformed_img.unsqueeze(0))
        pred = (pred.argmax(dim=1))
        pred = tt.Resize([h, w])(pred)
        pred = pred.squeeze()

    return pred.numpy()

def convert_into_rgb(labelformat, mask):
    # labelformat = 'ultra' or 'day'
    # переведем маски из hot в rgb
    colormap = np.array([(0, 0, 0),        # 1 Переслаивание / отсутствует
                          (128, 0, 128),    # 2 Алевролит глинистый / насыщенное
                          (250, 233, 0),    # 3 Песчаник / Карбонатное
                          (0, 128, 90),     # 4 Аргиллит
                          (0, 206, 247),    # 5 Проба
                          (111, 247, 0),    # 6 Разлом
                          (128, 128, 128),  # 7 Уголь
                          (0, 250, 221),    # 8 Аргиллит углистый
                          (64, 64, 192),    # 9 Алевролит
                          (255, 5, 255),    # 10 Карбонатная порода
                          (230, 5, 20),     # 11 Известняк
                          (124, 0, 30),     # 12 Глина аргиллитоподобная
                          (192, 128, 0),    # 13 Глинисто-кремнистая порода
                          (227, 178, 248)])  # 14 Песчаник глинистый
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)

    if labelformat == 'ultra':
        for l in range(3):
            idx = mask == l
            r[idx] = colormap[l, 0]
            g[idx] = colormap[l, 1]
            b[idx] = colormap[l, 2]
    else:
        for l in range(14):
            idx = mask == l
            r[idx] = colormap[l, 0]
            g[idx] = colormap[l, 1]
            b[idx] = colormap[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

def prediction(model, img, format, path, img_id, post_proccessing=True):
    # format: 'ultra' or 'day'
    predict = model_predict(model, img)
    rgb_mask = convert_into_rgb(format, predict)

    if post_proccessing:
        rgb_mask = cv2.GaussianBlur(rgb_mask, (17, 17), 0)

    rgb_mask = Image.fromarray(rgb_mask)
    rgb_mask.save(path + f'/{format}/{img_id}.jpg')


def text_generation(predict_mask, label, step_h=10, step_w=4):
    # mask in two-dimension numpy array ONLY WHICH HEIGHT IS MORE THAN 100 pixels
    # label is 'ultra' or 'day'
    result = []
    if label == 'ultraviolet':

        classes = ['Отсутствует', 'Насыщенное', 'Карбонатное']
        # в матрицах числами 0 обозначаены области, где свечение отсутствует
        #            числами 1 обозначаены области, где свечение насыщенное
        #            числами 2 обозначаены области, где свечение карбонатное
    else:
        classes = ['Переслаивание пород', 'Алевролит глинистый',
                   'Песчаник', 'Аргиллит', 'Разлом', 'Проба']
    h, w = predict_mask.shape

    one_percent = h // step_h
    one_percent_w = w // step_w

    last_class_name = ''  # необходим для того, чтобы разграничивать области между собой

    for i in range(step_h - 1):
        temporary = []
        i_part_of_mask = predict_mask[(i * one_percent): ((i + 1) * one_percent)][:]
        for j in range(step_w - 1):
            ij_part_of_mask = i_part_of_mask[:, (j * one_percent_w): ((j + 1) * one_percent_w)]
            ij_part_of_mask = ij_part_of_mask.flatten()

            index_of_main_class = np.argmax(np.bincount(ij_part_of_mask))

            class_name = classes[index_of_main_class]
            if class_name not in temporary:
                temporary.append(class_name)

        last_j_part_of_mask = i_part_of_mask[:, (step_w - 1) * one_percent_w:]
        last_j_part_of_mask = last_j_part_of_mask.flatten()
        index_of_main_class = np.argmax(np.bincount(last_j_part_of_mask))

        class_name = classes[index_of_main_class]
        if class_name not in temporary:
            temporary.append(class_name)

        if len(temporary) == 1:
            if temporary[0] != last_class_name:
                tuple_result = (round(i / (step_h), 2), temporary[0])
                result.append(tuple_result)
                last_class_name = temporary[0]
        else:
            if tuple(temporary) != result[-1][1:]:
                tuple_result = (round(i / (step_h), 2),) + tuple(set(temporary) - set(result[-1][1:]))
                result.append(tuple_result)
                last_class_name = ''

    last_part_of_mask = predict_mask[((step_h - 1) * one_percent):, :]
    temporary = []
    for j in range(step_w - 1):
        ij_part_of_mask = last_part_of_mask[:, j * one_percent_w: (j + 1) * one_percent_w]
        ij_part_of_mask = ij_part_of_mask.flatten()

        index_of_main_class = np.argmax(np.bincount(ij_part_of_mask))

        class_name = classes[index_of_main_class]
        if class_name not in temporary:
            temporary.append(class_name)

    last_j_part_of_mask = last_part_of_mask[:, (step_w - 1) * one_percent_w:]
    last_j_part_of_mask = last_j_part_of_mask.flatten()
    index_of_main_class = np.argmax(np.bincount(last_j_part_of_mask))

    class_name = classes[index_of_main_class]
    if class_name not in temporary:
        temporary.append(class_name)

    if len(temporary) == 1:
        if temporary[0] != last_class_name:
            tuple_result = (round((step_h-1) / (step_h), 2), temporary[0])
            result.append(tuple_result)
            last_class_name = temporary[0]
    else:
        if tuple(temporary) != result[-1][1:]:
            tuple_result = (round((step_h-1) / (step_h), 2),) + tuple((set(temporary) - set(result[-1][1:])))
            result.append(tuple_result)
            last_class_name = ''

    return result
