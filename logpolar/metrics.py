import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pylab as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from PIL import Image
import scipy.io


def simulate_fixations(h, w, n):
    # hxw: image size
    # n: number of fixations
    # generates a hxw binary map with n pixels set to 1 (the rest are 0)
    map = np.zeros((h, w))
    ys, xs = [np.random.randint(low=0, high=z, size=n + 1) for z in [h, w]]
    map[ys, xs] = 1
    return map


def change_fixations(map, pct):
    # map: binary fixationmap
    # pct: percentge of points to change
    ys, xs = np.where(map == 1)  # find where fixations are
    n = xs.shape[0]  # number of points
    n_new = int(np.round(n * pct / 100))
    n_keep = n - n_new
    h, w = map.shape
    xs, ys = xs[:n_keep], ys[:n_keep]
    new_map = simulate_fixations(h, w, n_new)  # new points
    new_map[ys, xs] = 1  # existing points
    return new_map


def blur_map(map, sigma):
    # map: input 2D binary map
    # sigma: standard deviation for 2D Gaussian
    # produces a continuous fixation map from the input map
    blurred = gaussian_filter(map, sigma)
    bDisplay = False
    if bDisplay:
        h, _ = np.histogram(np.ravel(blurred), bins=25)
        plt.plot(h)
        plt.show(block=True)
    return blurred  # /np.sum(blurred) # normalize?


def draw_ROC(fpr, tpr):
    plt.plot(fpr, tpr, '.-')
    plt.title("ROC")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show(block=True)


def AUC(b_true_fixations, d_pred_fixations, bDisplay=False):
    # Note: b_ and d_ prefixes stand for "binary" and "distribution" (continuous) data
    # b_true_fixations: 2D map of (ground-truth) binary fixations
    # d_pred_fixations: 2D map of continuous predicted fixations

    y_true = np.ravel(b_true_fixations)
    y_score = np.ravel(d_pred_fixations)
    # bDisplay = True
    if bDisplay:
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        draw_ROC(fpr, tpr)
    return roc_auc_score(y_true, y_score)


def plot_metric_evol(pcts, metric_values, metric_name):
    plt.plot(pcts, metric_values, 's-')
    plt.xlabel("disimilarity of salience map to true fixations (% points changed)")
    plt.ylabel(metric_name)
    plt.show(block=True)


def show_map(map, title):
    plt.imshow(map, cmap='gray')
    plt.title(title)
    plt.show(block=True)


if __name__ == "__main__":
    f = open("results.txt", "a")
    f.write("Results:\n")
    images_to_evaluate = "1009#1026#1092#1093#1094#1097#1144#1159#1164#1278#1387#1408#1409#1414#1442#1471#1527#1572" \
                         "#1585#1617"

    folders = ['log-polar/high-quality', 'log-polar/low-quality', 'mouse_amt', 'mouse_lab']

    images_to_evaluate = images_to_evaluate.split("#")
    categories = {'big_objects': ['1585', '1009', '1026', '1144', '1097', '1094', '1617', '1408', '1442'],
                  'small_objects': ['1387', '1572', '1092', '1093', '1164', '1471', '1414'],
                  'one_object': ['1009', '1094', '1097', '1617', '1527', '1409', '1442'],
                  'various_objects': ['1572', '1585', '1387', '1026', '1144', '1159',
                                      '1092', '1093', '1164', '1471', '1408', '1414'],
                  'persons': ['1585', '1278', '1572', '1159', '1409', '1442']
                  }

    for folder in folders:
        aucs = []
        for image in images_to_evaluate:
            mat = scipy.io.loadmat('fixation_maps/{0}.mat'.format(image))
            b_gtrue = mat['fixationPts']
            # show_map(b_gtrue, 'fixations (binary)')

            imfile = 'mouse_maps/{1}/{0}.jpg'.format(image, folder)
            d_pred = Image.open(imfile).convert('L')
            # show_map(d_pred, 'saliency map')

            aucs.append(AUC(b_gtrue, d_pred, bDisplay=False))

        f.write("Total:\n")
        f.write(folder + "\n")
        f.write(str(sum(aucs) / len(aucs)) + "\n")

        f.write("" + "\n")
        f.write("Por categor??a:" + "\n")
        f.write(folder + "\n")
        for category in categories:
            aucs = []
            for im in categories[category]:
                mat = scipy.io.loadmat('fixation_maps/{0}.mat'.format(im))
                b_gtrue = mat['fixationPts']
                # show_map(b_gtrue, 'fixations (binary)')

                imfile = 'mouse_maps/{1}/{0}.jpg'.format(im, folder)
                d_pred = Image.open(imfile).convert('L')
                # show_map(d_pred, 'saliency map')

                aucs.append(AUC(b_gtrue, d_pred, bDisplay=False))

            f.write(category + "\n")
            f.write(str(sum(aucs) / len(aucs)) + "\n")

        f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write("\n")
    f.close()
    # h,w=100,150
    # n=150
    # pct=30
    # sigma=10
    # b_gtrue = simulate_fixations(h, w, n)
    #
    # b_pred = change_fixations(b_gtrue, pct)
    # d_pred = blur_map(b_pred, sigma)
    #
    # show_map(b_gtrue,"(mock) True fixations (binary)")
    # show_map(d_pred,"(mock) Estimated salience")
    #
    # print("AUC", AUC(b_gtrue, d_pred, bDisplay=True))
    #
    # aucs = []
    # pcts = [0,5,10,20,30,40,50,70,90,100]
    # bDisplayEachROC=True
    # for pct in pcts:
    #     b_pred = change_fixations(b_gtrue, pct)
    #     d_pred = blur_map(b_pred, 3)
    #     aucs.append(AUC(b_gtrue, d_pred, bDisplayEachROC))
    #
    # plot_metric_evol(pcts,aucs,"AUC")
