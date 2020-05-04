import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os


def plot(evaluater, trackers, output_path):

    success_dict = {}
    precision_dict = {}
    success_auc_dict = {}
    precision_auc_dict = {}

    for name, param in tqdm(trackers.items()):
        if not param["repeat"]:
            success_dict[name], precision_dict[name] = evaluater.evaluate(param["path"])
            success_auc_dict[name] = success_dict[name].mean()
            precision_auc_dict[name] = precision_dict[name].mean()
        else:
            repeats = [ x for x in os.listdir(param["path"]) if os.path.isdir(os.path.join(param["path"], x)) ]
            result_s, result_p = [], []
            for repeat in repeats:
                s, p = evaluater.evaluate(os.path.join(param["path"], repeat))
                result_s.append(s)
                result_p.append(p)
            success_dict[name], precision_dict[name] = np.mean(result_s, axis=0), np.mean(result_p, axis=0)
            
            success_auc_dict[name] = success_dict[name].mean()
            precision_auc_dict[name] = precision_dict[name].mean()


    ######################################
    # Draw success plot and precision plot
    ######################################
    iou_thresholds = np.arange(0, 1.05, 0.05)
    dist_thresholds = np.arange(0, 51, 1)

    nx = 2
    ny = 1
    dxs = 8.0
    dys = 5.0
    colors = [  'blue', 'red', 'green', 'orange', 'gray', 'black', 'cyan', 
                'purple', 'navy', 'darkcyan', 'darkorchid', 'chocolate', 'burlywood']
    linestyles = ['-', '--', '-.']
    fig, ax = plt.subplots(ny, nx, sharey = False, figsize=(dxs*nx, dys*ny) )

    for name in success_dict:
        color = np.random.choice(colors)
        colors.remove(color)
        linestyle = np.random.choice(linestyles)
        ax[0].plot(iou_thresholds, success_dict[name], color=color, lw=2.5, label=name+'[%.3f]'%success_auc_dict[name], linestyle=linestyle)
        ax[1].plot(dist_thresholds, precision_dict[name], color=color, lw=2.5, label=name+'[%.3f]'%precision_auc_dict[name], linestyle=linestyle)
    ax[0].set_title('Success Plot of OPE',fontsize=15)
    ax[0].set_xlabel('Overlap threshold', fontsize=15)
    ax[0].set_ylabel('Success rate', fontsize=15)
    ax[0].legend(loc="lower left")
    ax[0].set_xlim(0.0, 1.0)
    ax[0].set_ylim(0.0, 1.0)

    ax[1].set_title('Precision Plot of OPE',fontsize=15)
    ax[1].set_xlabel('Location error threshold', fontsize=15)
    ax[1].set_ylabel('Precision', fontsize=15)
    ax[1].legend(loc="lower right")
    ax[1].set_xlim(0, 50)
    ax[1].set_ylim(0.0, 1.0)

    if output_path:
        filename = output_path
    else:
        filename = 'result.png'
    fig.savefig(filename)