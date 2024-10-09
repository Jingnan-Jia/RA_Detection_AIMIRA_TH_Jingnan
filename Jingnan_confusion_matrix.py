import numpy as np
# import warnings
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
# warnings.simplefilter('ignore')
# import cv2
# cv2.setNumThreads(0)
import json

# organ_ls = [
#     # 'bones',
#     #        'skin',
#     #        'vessel',
#     #        'tendons',
#     #        'othertissue',
#     #         'TSY',
#     #         'SYN',
#             ]

# organ_ls = [ 'background', 'skin', 'vessel', 'tendon', 'bone', 'synovitis', 'tenosynovitis', 'othertissue']
organ_ls = [ 'othertissue']

tsy_thresholds = np.arange(0.4, 1, 0.2)
# tsy_thresholds = [0.2]
for tsy_threshold in tsy_thresholds:
  print(f"threshold: {tsy_threshold}")
  with open(f'data/all_results3_tsy_threshold{tsy_threshold}.json', 'r', encoding='utf-8') as json_file:
      prediction_split = json.load(json_file)
      
  for organ in organ_ls:
      TP_TP = 0
      TP_FN = 0
      FP_FP = 0
      FP_TN = 0
      TN_TN = 0
      TN_FP = 0
      FN_FN = 0
      FN_TP = 0
      for i in range (1,11):
          print(f"---------organ: {organ}, repeat: {i}--------")
          groundtruth =   prediction_split[f"repeat3_fold{i}_groundtruth"]
          original_pred = prediction_split[f"repeat3_fold{i}_exclude_ori_pred"]
          bone_blocked_pred = prediction_split[f"repeat3_fold{i}_exclude_{organ}_threshold_{tsy_threshold}_pred"]
          
          # bone_blocked_pred = np.load(f'results/repeat_3_predict_label_fold{i}_exclude_{organ}.npy')
          # groundtruth =   np.load('TH_repeat_3_label_fold'+str(i)+'_groundtruth.npy')
          # original_pred = np.load(path + 'repeat_3_predict_label_fold'+str(i)+'_orginal.npy')
          # bone_blocked_pred = np.load(f'TH_repeat_3_predict_label_fold{i}_bone.npy')
          # print(groundtruth)
          # print(original_pred)
          # print(bone_blocked_pred)
          for j in range (len(original_pred)):

              if groundtruth[j] == 1 and original_pred[j] == 1 and bone_blocked_pred[j] == 1:
                TP_TP = TP_TP + 1
              if groundtruth[j] == 1 and original_pred[j] == 1 and bone_blocked_pred[j] == 0:
                TP_FN = TP_FN + 1
              if groundtruth[j] == 0 and original_pred[j] == 1 and bone_blocked_pred[j] == 1:
                FP_FP = FP_FP + 1
                print('fold=', i)
                print('not helping patients FP_FP =', j)
              if groundtruth[j] == 0 and original_pred[j] == 1 and bone_blocked_pred[j] == 0:
                FP_TN = FP_TN + 1
                print('fold=', i)
                print('confusing patients FP_TN =', j)
              if groundtruth[j] == 0 and original_pred[j] == 0 and bone_blocked_pred[j] == 0:
                TN_TN = TN_TN + 1 
              if groundtruth[j] == 0 and original_pred[j] == 0 and bone_blocked_pred[j] == 1:
                TN_FP = TN_FP + 1
              if groundtruth[j] == 1 and original_pred[j] == 0 and bone_blocked_pred[j] == 0:
                FN_FN = FN_FN + 1
                print('fold=', i)
                print('not helping patients FN_FN =', j)
              if groundtruth[j] == 1 and original_pred[j] == 0 and bone_blocked_pred[j] == 1:
                FN_TP = FN_TP + 1
                print('fold=', i)
                print('confusing patients FN_TP =', j)

      print('TP_TP = ', TP_TP)
      print('TP_FN = ', TP_FN)
      print('FP_FP = ', FP_FP)
      print('FP_TN = ', FP_TN)
      print('TN_TN = ', TN_TN)
      print('TN_FP = ', TN_FP)
      print('FN_FN = ', FN_FN)
      print('FN_TP = ', FN_TP)

      irrelevant = TP_TP + TN_TN
      essential = TP_FN + TN_FP
      confusing = FP_TN + FN_TP
      not_helping = FP_FP + FN_FN

      print('irrelevant=', irrelevant)
      print('essential=', essential )
      print('confusing=', confusing)
      print('not_helping=', not_helping)

      NB = sum([irrelevant, essential, confusing, not_helping])
      print('%irrelevant=', irrelevant*100/NB)
      print('%essential=', essential*100/NB )
      print('%confusing=', confusing*100/NB)
      print('%not_helping=', not_helping*100/NB)


    # for i in range (64):
    #     tn, fp, fn, tp = confusion_matrix(all_original[:,i], all_predict[:,i]).ravel()
    #     print('confusion_'+str(i)+'_tn = %s_fp = %s_fn = %s_tp = %s_',tn, fp, fn, tp )
    # print(np.shape(all_original))
    # print(np.shape(all_predict))