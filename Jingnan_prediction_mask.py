import copy
import glob
import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import sklearn
from torchsummary import summary
# import torchio as tio
import warnings
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
warnings.simplefilter('ignore')
import cv2
cv2.setNumThreads(0)
import pickle
import SimpleITK as sitk
import json
# from scipy.ndimage import zoom
# from tensorflow.keras.preprocessing.image import array_to_img
# from medcam3d import GradCAM
# from medcam3d.utils.model_targets import ClassifierOutputTarget
# from medcam3d.utils.image import show_cam_on_image
# from PIL import Image
device = torch.device('cuda')
print(device)
    
class scratch_nn(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=8, kernel_size=3, padding ='same')
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, padding ='same')
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, padding ='same')
        self.conv4 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding ='same')
        self.conv5 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
        self.conv6 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
        self.conv1_bn = nn.BatchNorm3d(8)
        self.conv2_bn = nn.BatchNorm3d(8)
        self.conv3_bn = nn.BatchNorm3d(16)
        self.conv4_bn = nn.BatchNorm3d(16)
        self.conv5_bn = nn.BatchNorm3d(16)
        self.conv6_bn = nn.BatchNorm3d(16)
        self.drop1 = nn.Dropout(0.6)
        self.drop2 = nn.Dropout(0.6)
        self.drop3 = nn.Dropout(0.6)
        self.drop4 = nn.Dropout(0.6)
        self.drop5 = nn.Dropout(0.6)
        self.drop6 = nn.Dropout(0.7)
        self.fc1 = nn.Linear(40960, num_classes)

    def forward(self, x):
        x = F.max_pool3d(x, kernel_size=(2, 2, 1), stride=(2, 2, 1)) #first int is used for the depth dimension, the second int for the height dimension and the third int for the width dimension
        x = F.relu(self.conv1_bn(self.conv1(x)))
        #x = self.drop1(x)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        #x = self.drop2(x)

        x = F.max_pool3d(x, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        #x = self.drop3(x)
        x = F.relu(self.conv4_bn(self.conv4(x)))
        #x = self.drop4(x)

        x = F.max_pool3d(x, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        #x = self.drop5(x)
        x = F.relu(self.conv6_bn(self.conv6(x)))
        #x = self.drop6(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)

        return x


class RAMRISDataset(Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data.astype('float32'), label


def load_itk(filename, require_ori_sp=False):
    """

    :param filename: absolute file path
    :return: ct, origin, spacing, all of them has coordinate (z,y,x) if filename exists. Otherwise, 3 empty list.
    """
    #     print('start load data')
    # Reads the image using SimpleITK
    if os.path.isfile(filename):
        itkimage = sitk.ReadImage(filename)

    else:
        raise FileNotFoundError(filename + " was not found")

    # Convert the image to a  numpy array first ands then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # ct_scan[ct_scan>4] = 0 #filter trachea (label 5)
    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))  # note: after reverseing, origin=(z,y,x)

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))  # note: after reverseing,  spacing =(z,y,x)
    orientation = itkimage.GetDirection()
    if orientation[-1] == -1:
        ct_scan = ct_scan[::-1]
    if require_ori_sp:
        return ct_scan, origin, spacing
    else:
        return ct_scan


def save_itk(filename, scan, origin, spacing, dtype='int16'):
    """
    Save a array to itk file.

    :param filename: saved file name, a string.
    :param scan: scan array, shape(z, y, x)
    :param origin: origin of itk file, shape (z, y, x)
    :param spacing: spacing of itk file, shape (z, y, x)
    :param dtype: 'int16' default
    :return: None
    """

    stk = sitk.GetImageFromArray(scan.astype(dtype))
    # origin and spacing 's coordinate are (z,y,x). but for image class,
    # the order shuld be (x,y,z), that's why we reverse the order here.
    stk.SetOrigin(origin[::-1])
    # numpy array is reversed after convertion from image, but origin and spacing keep unchanged
    stk.SetSpacing(spacing[::-1])

    sitk.WriteImage(stk, filename, useCompression=True)



def data_prepare(organ, encoding='rgb', tsy_threshold=3):
    
    # param_ls = [
    #     {'tissue': 'skin', 'label': [1]},
    #     {'tissue': 'vessel', 'label': [2]},
    #     {'tissue': 'othertissue', 'label': [3]},
    #     {'tissue': 'tendon', 'label': list(range(4, 18))},  # 4 to 17
    #     {'tissue': 'bone', 'label': list(range(18, 33))}, # 18 to 32
    #     {'tissue': 'synovitis', 'label': [1]},                 # label is 1 in new mask
    #     {'tissue': 'tenosynovitis', 'label': [2]},
    # ]
    
    organ_label_dt = {'background': 0,
                        'skin': 1,
                        'vessel': 2,
                        'tendon': 3,
                        'bone': 4,
                        'synovitis': 5,
                        'tenosynovitis': 6,
                        'othertissue': 8,
                        }
    
    CM_fpath = 'data/grayscale_CM.mha'  # common change map
    CM = load_itk(CM_fpath)
    # CM = np.load(CM_fpath)  # load change map
    
    mask_fpath = 'data/all_masks15.mha'
    mask = load_itk(mask_fpath)
        
    # mask = np.load(mask_fpath)  # load mask
    RAMRIS = CM
    if organ!='ori':
        # if organ == 'tenosynovitis':
        #     pass
        #     # old_tsy = load_itk('data/Jingnan_TSY_mask.mha')
        #     # old_tsy = (old_tsy > 0).astype(int)
        #     # new_syn = (mask==organ_label_dt['synovitis']).astype(int)
        #     # intersection = old_tsy * new_syn
        #     # mask2 = old_tsy - intersection
            
        #     # RAMRIS[mask2==1] = 128  # only include 0, 128, 256
        
        #     # mask_of_organ = np.zeros_like(mask2)
        #     # mask_of_organ[mask2==1] = 1
        #     # save_itk(f"data/mask_of_{organ}_test.mha", mask_of_organ, [1,1,1], [1,1,1])
        #     # save_itk(f"data/CM_exclude_{organ}_test.mha", RAMRIS, [1,1,1], [1,1,1])
            
        
        # else:

        if organ in ["tenosynovitis", "othertissue"] and tsy_threshold:
            mask_dist_map = sitk.ReadImage(f"data/mask_distance_map_tsy_with_spacing.mha")
            mask_dist_map = sitk.GetArrayFromImage(mask_dist_map)
            mask_dist_map[mask_dist_map<tsy_threshold] = 0
            mask_dist_map[mask_dist_map>tsy_threshold] = 1
            mask_of_organ = 1 - mask_dist_map
            
            # mask_of_tendon = np.zeros_like(mask)
            # mask_of_tendon[mask==organ_label_dt['tendon']] = 1
            
            for k, v in organ_label_dt.items():
                if k not in ["othertissue", "tenosynovitis"]:
                    mask_of_organ[mask==v] = 0
            if organ == "othertissue":
                mask_of_othertissue = np.zeros_like(mask)
                mask_of_othertissue[mask==organ_label_dt['othertissue']] = 1
                mask_of_othertissue[mask_of_organ==1] = 0
                mask_of_organ = mask_of_othertissue
                
            save_itk(f"data/mask_of_{organ}_threshold_{tsy_threshold}.mha", mask_of_organ, [1,1,1], [1,1,1])
            
            RAMRIS[mask_of_organ==1] = 128  # only include 0, 128, 256            
            save_itk(f"data/CM_exclude_{organ}_threshold_{tsy_threshold}.mha", RAMRIS, [1,1,1], [1,1,1])
            
            
        else:
            mask_of_organ = np.zeros_like(mask)
            mask_of_organ[mask==organ_label_dt[organ]] = 1
        
            save_itk(f"data/mask_of_{organ}.mha", mask_of_organ, [1,1,1], [1,1,1])
            
            RAMRIS[mask==organ_label_dt[organ]] = 128  # only include 0, 128, 256            
            save_itk(f"data/CM_exclude_{organ}.mha", RAMRIS, [1,1,1], [1,1,1])
            
        # mask_binary = mask!=organ_label_dt[organ]  # the voxels not equal to organ label should be 1
        # RAMRIS = CM * mask_binary

    if encoding=='gray':
        RAMRIS = RAMRIS.reshape(185, 10, 128, 128, 1)  # (185, 10, 128, 128, 1)
        rgb_array = np.concatenate((RAMRIS,RAMRIS,RAMRIS), axis=-1)  # (185, 10, 128, 128, 3)
    else:
        RAMRIS = RAMRIS.reshape(185, 10, 128, 128)  # (185, 10, 128, 128, 1)
        # 创建新的形状为 (185, 10, 128, 128, 3) 的数组
        rgb_array = np.zeros((185, 10, 128, 128, 3), dtype=np.uint8)

        # 将原数组中的 0 映射为 [255, 0, 0]
        rgb_array[RAMRIS == 0] = [0, 0, 255]

        # 将原数组中的 128 映射为 [0, 0, 0]
        rgb_array[RAMRIS == 128] = [0, 0, 0]

        # 将原数组中的 256 映射为 [0, 0, 255]
        rgb_array[RAMRIS == 255] = [255, 0, 0]

    imgs = rgb_array.transpose(0, 4, 2, 3, 1)  # (185, 3, 128, 128, 10)
    
    label_path = 'data/Label.npy'
    label = np.load(label_path)

    return imgs, label


# def train_step(model, optimizer, criterion, train_loader, test_dataloader, num_classes, random_label=False):
#     model.train()
#     avg_loss = []
#     running_loss = []
#     running_loss_val = []
#     for subjects_batch in train_loader:
#         x = subjects_batch['t1'][tio.DATA]
#         y = subjects_batch['label']
#         x = x.to(device)
#         y = y.type(torch.LongTensor).to(device)
#         y_pred = model(x=x)

#         loss = criterion(y_pred, y)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         avg_loss.append(loss.item())
#         ####################train loss and val loss

#         # running_loss += loss.item()
#         # running_loss = loss.item()
#         # print(f' loss: {running_loss :.3f}')
#         running_loss.append(loss.item())


#     for z, w in test_dataloader:
#         z = z.to(device)
#         w = w.type(torch.LongTensor).to(device)
#         model.eval()
#         y_pred_val = model(x=z)
#         loss_val = criterion(y_pred_val, w)
#         # running_loss_val += loss_val.item()
#         running_loss_val.append(loss_val.item())
#         # running_loss_val = loss_val.item()


#     #print(f' loss: {np.mean(running_loss) :.3f} ----------------> loss_val: {np.mean(running_loss_val) :.3f}')

#     return sum(avg_loss) / len(avg_loss)


def predict(model, test_loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    conf_before_max = []
    conf_after_max = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x=x)
            probs = torch.nn.functional.softmax(pred, dim=1)
            conf_before_max.append(probs)
            # print('probs=', probs )
            conf, classes = torch.max(probs, 1)
            y_pred = torch.argmax(pred, dim=1)
            conf_after_max.append(conf)
            # print('conf=', conf)
            # print('class =', classes)
            total_preds = torch.cat((total_preds, y_pred.cpu()), 0)
            total_labels = torch.cat((total_labels, y.cpu()), 0)
            # total_conf =
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), conf_before_max, conf_after_max

# def aug(dataset):
#     # data shape is (175,2). data[i][0] is X_train and data[i][1] is y_train
#     # https://torchio.readthedocs.io/transforms/transforms.html#torchio.transforms.Transform
#     subject_list = []
#     for i in range (len(dataset)):
#         tensor_4d = dataset[i][0]
#         subject = tio.Subject(
#             t1=tio.ScalarImage(tensor=tensor_4d),
#             label=(dataset[i][1]),
#         )
#         subject_list.append(subject)
#     #######rescale
#     rescale = tio.RescaleIntensity(out_min_max=(0, 1))
#     #######spatial
#     spatial = tio.OneOf({
#         tio.RandomAffine(): 0.8,
#         tio.RandomElasticDeformation(): 0.2,
#     },
#         p=0.75,
#     )
#     #######flip
#     flip = tio.RandomFlip(axes=(0,), flip_probability = 0.6)
#     #######intensity
#     intensity = tio.OneOf({
#       tio.RandomNoise(mean = 0, std = (0, 0.25)),
#       tio.RandomMotion(degrees = 10, translation= 10, num_transforms=2, image_interpolation='linear'),
#       tio.RandomBlur(std = (0, 2)),
#       tio.RandomGamma(log_gamma= (-0.2, 0.2))})
#     #############
#     transforms = [spatial]
#     transforms = tio.Compose(transforms)
#     transforms_1 = [rescale]
#     transforms_1 = tio.Compose(transforms_1)
#     transforms_2 = [flip]
#     transforms_2 = tio.Compose(transforms_2)
#     transforms_3 = [intensity]
#     transforms_3 = tio.Compose(transforms_3)
#     com_1 = [flip, intensity]
#     com_2 = [spatial, flip]
#     com_3 = [spatial,rescale,flip]
#     com_4 = [rescale,flip,intensity]
#     com_5 = [spatial, flip,intensity]
#     transforms_4 = tio.Compose(com_1)
#     transforms_5 = tio.Compose(com_2)
#     transforms_6 = tio.Compose(com_3)
#     transforms_7 = tio.Compose(com_4)
#     transforms_8 = tio.Compose(com_5)

#     return subject_list, transforms, transforms_1, transforms_2, transforms_3, transforms_4, transforms_5, transforms_6, transforms_7, transforms_8



# def train(model, dataset, val_set, lr=0.01, num_epoch=200):
#     lr = lr
#     weight_dec = 0.0001
#     max_accuracy = 0
#     max_auc = 0
#     subject_list, transform, transform_1, transform_2, transform_3, transform_4, transform_5, transform_6, transform_7, transform_8  = aug(dataset)
#     subjects_dataset = tio.SubjectsDataset(subject_list)
#     subjects_dataset_0 = tio.SubjectsDataset(subject_list, transform=transform)
#     subjects_dataset_1 = tio.SubjectsDataset(subject_list, transform=transform_1)
#     subjects_dataset_2 = tio.SubjectsDataset(subject_list, transform=transform_2)
#     subjects_dataset_3 = tio.SubjectsDataset(subject_list, transform=transform_3)
#     subjects_dataset_4 = tio.SubjectsDataset(subject_list, transform=transform_4)
#     subjects_dataset_5 = tio.SubjectsDataset(subject_list, transform=transform_5)
#     subjects_dataset_6 = tio.SubjectsDataset(subject_list, transform=transform_6)
#     subjects_dataset_7 = tio.SubjectsDataset(subject_list, transform=transform_7)
#     subjects_dataset_8 = tio.SubjectsDataset(subject_list, transform=transform_8)

#     subjects_dataset = np.concatenate((subjects_dataset, subjects_dataset_0, subjects_dataset_1, subjects_dataset_2, subjects_dataset_3, subjects_dataset_4, subjects_dataset_5, subjects_dataset_6, subjects_dataset_7, subjects_dataset_8), axis =0)

#     print('subjects_dataset=', np.shape(subjects_dataset))
#     train_dataloader = DataLoader(subjects_dataset, batch_size=16, shuffle=True, num_workers=2)
#     test_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)
#     #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_dec)
#     optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_dec)
#     criterion = nn.CrossEntropyLoss()
#     model = model.to(device)
#     for epoch in range(num_epoch):
#         tarin_loss = train_step(model, optimizer, criterion, train_dataloader, test_dataloader, num_classes=2)
#         G, P = predict(model, test_dataloader)
#         accuracy = accuracy_score(G, P)
#         auc1 = sklearn.metrics.roc_auc_score(G, P)
#         if accuracy > max_accuracy:
#             print('accuracy=', accuracy)
#             print('auc=', auc1)
#             max_accuracy = accuracy
#             max_auc = auc1
#     torch.save(model.state_dict(), '/exports/lkeb-hpc/thassanzadehkoohi/final_treatment_prediction_result/MTP/CM/model_CM_'+str(max_accuracy) +'___'+ str(max_auc)+'pth')
#     print('max accuracy in this fold:%s and max auc in this fold: %s' %(max_accuracy ,max_auc))
#     return model



# ######################################### Mask and the rest
# def Create_bone_mask(CM_first, l, patient_num, filter_size):

#     if l == 1 :
#         label = np.ones(16385)  # 16385  128*128+1
#     else:
#         label = np.zeros(16385)


#     C = CM_first[4]  # central slice
#     print('shape before pad = ', np.shape(C))
#     C =   np.pad(C, pad_width=[(int(filter_size/2), int(filter_size/2)),(int(filter_size/2), int(filter_size/2)),(0, 0)], mode='constant')
#     print('after padding=', np.shape(C))

#     CM = []
#     for i in range (10):
#         CM.append(C)
#     CM = np.array(CM)
#     print('CM_initial=', np.shape(CM))
#     CM_final = []
#     if filter_size == 8:
#         CM_trim = np.delete(CM, [0,1,2,3,128+int(filter_size/2),128+int(filter_size/2)+1,128+int(filter_size/2)+2,128+int(filter_size/2)+3], 1)
#         CM_trim = np.delete(CM_trim, [0, 1, 2, 3, 128+int(filter_size/2),128+int(filter_size/2)+1,128+int(filter_size/2)+2,128+int(filter_size/2)+3], 2)
#     elif filter_size == 4:
#         CM_trim = np.delete(CM, [0,1, 128+int(filter_size/2),128+int(filter_size/2)+1], 1)
#         CM_trim = np.delete(CM_trim, [0, 1,  128+int(filter_size/2),128+int(filter_size/2)+1], 2)
#     else:
#         CM_trim = np.delete(CM, [0, 128+int(filter_size/2)], 1)
#         CM_trim = np.delete(CM_trim, [0,  128+int(filter_size/2)], 2)

#     C_pic = array_to_img(CM_trim[4])
#     C_pic.save('patient'+str(patient_num)+'_each_pixel.png')
#     CM_final.append(CM_trim)  # fist image is original image
#     print('CM_initial_after_trim=', np.shape(CM_final))

#     for j in range(int(filter_size/2), 128+ int(filter_size/2)):
#         for k in range(int(filter_size/2), 128+ int(filter_size/2)):
#             CM_2 = copy.deepcopy((CM))
#             CM_2[:, j - filter_size :j + filter_size, k - filter_size :k + filter_size] = 0
#             CM_2 = np.array(CM_2)
#             if filter_size == 8:
#                 CM_2 = np.delete(CM_2, [0,1,2,3,128+int(filter_size/2),128+int(filter_size/2)+1,128+int(filter_size/2)+2,128+int(filter_size/2)+3], 1)
#                 # print('after first delete =', np.shape(CM_2))
#                 CM_2 = np.delete(CM_2, [0, 1, 2, 3,128+int(filter_size/2),128+int(filter_size/2)+1,128+int(filter_size/2)+2,128+int(filter_size/2)+3], 2)
#             elif filter_size == 4:
#                 CM_2 = np.delete(CM_2, [0,1,128+int(filter_size/2),128+int(filter_size/2)+1], 1)
#                 # print('after first delete =', np.shape(CM_2))
#                 CM_2 = np.delete(CM_2, [0, 1, 128+int(filter_size/2),128+int(filter_size/2)+1], 2)
#             else:
#                 CM_2 = np.delete(CM_2, [0,128+int(filter_size/2)], 1)
#                 # print('after first delete =', np.shape(CM_2))
#                 CM_2 = np.delete(CM_2, [0, 128+int(filter_size/2)], 2)

#             # print('after second delete =', np.shape(CM_2))
#             CM_final.append(CM_2)

#     CM_final = np.array(CM_final)
#     print('shape of array ====  ', np.shape(CM_final))
#     # for i in range (len(CM_final)):
#     #     pic= CM_final[i,0,:,:,:]
#     #     pic = array_to_img(pic)
#     #     pic.save('E:\\AIMIRA_final_treatment_effect_code\\create_mask_image_new\\pic2\\'+str(i)+'.png')

#     data = CM_final.transpose(0, 4, 2, 3, 1)
#     test_set = RAMRISDataset(data, label)
#     model.load_state_dict(torch.load(
#         'E:\\AIMIRA_final_treatment_effect_code\\create_mask_image_new\\models\\model_CM_joint_old_3_fold=1.pth',
#         map_location=torch.device('cuda')))
#     test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)
#     G, P, conf_before_max, conf_after_max = predict(model, test_dataloader)

#     # print('G =', G)
#     # print('P =', P)
#     # print('conf_before_max=', conf_before_max)
#     # print('conf_after_max=', conf_after_max)

#     np.save('conf_before_max_patient' + str(patient_num) + '_size' + str(filter_size) + '_each_pixel.npy', conf_before_max)
#     np.save('conf_after_max_patient' + str(patient_num) + '_size' + str(filter_size) + '_each_pixel.npy', conf_after_max)
#     np.save('patient_' + str(patient_num) + '_predict_label_size' + str(filter_size) + '_each_pixel.npy', P)
#     # np.save('patient_165_predict_confidence_4.npy', y_pred)

#     Final_label = []
#     for i in range(len(label)):
#         if G[i] == P[i]:
#             Final_label.append(1)
#         else:
#             Final_label.append(-1)

#     np.save('patient_' + str(patient_num) + '_masked_label_size' + str(filter_size) + '_each_pixel.npy', Final_label)

#     sub = G - P
#     print('sum =', np.sum(sub))
#     # print(len(P))
#     # print(len(G))
#     return


if __name__ == '__main__':

    path = 'data/'
    model = scratch_nn(num_classes=2)
    model = model.to(device)
    summary(model, (3, 128, 128, 10), device='cuda')

    wrist_img_masked_ls = [ 'background', 'skin', 'vessel', 'tendon', 'bone', 'synovitis', 'tenosynovitis', 'othertissue', 'ori' ]
    # wrist_img_masked_ls = [ 'othertissue', 'ori']

    
    with open('data_split.json', 'r', encoding='utf-8') as json_file:
        data_split = json.load(json_file)
        
    result_dt = {}
    acc_auc_dt = {}
    tsy_thresholds = np.arange(0.4, 1, 0.2)  # tune the threshold of TSY
    # tsy_thresholds = [0.2]
    for tsy_threshold in tsy_thresholds:
        for wrist_img_masked in wrist_img_masked_ls:   
            print(f"====={wrist_img_masked}:{tsy_threshold}======")        
            RAMRIS, label = data_prepare(organ=wrist_img_masked, tsy_threshold=tsy_threshold)
            
            for j in range(3,4):  # 5
                for i in range(1, 11):  # (1, 11)
                    x_train = data_split[f"round_{j}_fold{i}_train"]
                    x_test = data_split[f"round_{j}_fold{i}_test"]
                    
                    x_test = [x for x in x_test if x not in [91, 154, 101]]  # 91号原图质量太差，154号的源图没找见, 101的CM的背景是255（应该是128）                    

                    train_set = RAMRISDataset(RAMRIS[x_train], label[x_train])
                    test_set = RAMRISDataset(RAMRIS[x_test], label[x_test])
                    # CM = CM[x_test]
                    # Input1 = Input1[x_test]
                    # Input2 = Input2[x_test]
                    model_fpath = 'models/model_CM_joint_old_' + str(j) + '_fold=' + str(i) + '.pth'
                    model_fpath_ls = glob.glob(model_fpath)
                    model_fpath = model_fpath_ls[0]
                    model.load_state_dict(torch.load(model_fpath, map_location=torch.device('cuda')))
                    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)
                    G, P, conf_before_max, conf_after_max = predict(model, test_dataloader)
                    print('G =', G)
                    print('P =', P) 
                    # print('conf_before_max=', conf_before_max)
                    # print('conf_after_max=', conf_after_max)

                    
                    # np.save(f'results/conf_before_max_repeat{j}_fold{i}_exclude_{wrist_img_masked}.npy', conf_before_max)
                    # np.save(f'results/conf_after_max_repeat{j}_fold{i}_exclude_{wrist_img_masked}.npy', conf_after_max.cpu().numpy())
                    # np.save(f'results/repeat_{j}_predict_label_fold{i}_exclude_{wrist_img_masked}.npy', P)
                    # np.save(f'results/repeat_{j}_label_fold{i}_groundtruth.npy', G)

                    accuracy = accuracy_score(G, P)
                    auc1 = sklearn.metrics.roc_auc_score(G, P)
                    print('accuracy=', accuracy)
                    print('auc=', auc1)
                    if wrist_img_masked in ["tenosynovitis", "othertissue"] and tsy_threshold:
                        result_dt[f"repeat{j}_fold{i}_exclude_{wrist_img_masked}_threshold_{tsy_threshold}_pred"] = P.tolist()
                        result_dt[f"repeat{j}_fold{i}_groundtruth"] = G.tolist()
                        acc_auc_dt[f"repeat{j}_fold{i}_exclude_{wrist_img_masked}_threshold_{tsy_threshold}_accuracy"] = accuracy
                        acc_auc_dt[f"repeat{j}_fold{i}_exclude_{wrist_img_masked}_threshold_{tsy_threshold}_auc"] = auc1
                    else:
                        result_dt[f"repeat{j}_fold{i}_exclude_{wrist_img_masked}_pred"] = P.tolist()
                        result_dt[f"repeat{j}_fold{i}_groundtruth"] = G.tolist()
                        acc_auc_dt[f"repeat{j}_fold{i}_exclude_{wrist_img_masked}_accuracy"] = accuracy
                        acc_auc_dt[f"repeat{j}_fold{i}_exclude_{wrist_img_masked}_auc"] = auc1
                        
        with open(f'data/all_results3_tsy_threshold{tsy_threshold}.json', 'w', encoding='utf-8') as json_file:
            json.dump(result_dt, json_file, ensure_ascii=False, indent=4)
        with open(f'data/acc_auc_results3_tsy_threshold{tsy_threshold}.json', 'w', encoding='utf-8') as json_file:
            json.dump(acc_auc_dt, json_file, ensure_ascii=False, indent=4)
print(f"finish all")
