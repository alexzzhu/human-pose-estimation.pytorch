import numpy as np

# pred_jts and gt_jts are batch x n_jts x 2
def compute_pck(pred_jts, gt_jts):
    errs = np.linalg.norm(pred_jts - gt_jts, axis=-1)

def ap_per_class(tp): #tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum()
        tpc = (tp[i]).cumsum()

        # Recall
        recall = tpc / (n_gt + 1e-16)  # recall curve
        r.append(recall[-1])

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p.append(precision[-1])

        # AP from recall-precision curve
        ap.append(compute_ap(recall, precision))

        # Plot
        # fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        # ax.plot(np.concatenate(([0.], recall)), np.concatenate(([0.], precision)))
        # ax.set_xlabel('YOLOv3-SPP')
        # ax.set_xlabel('Recall')
        # ax.set_ylabel('Precision')
        # ax.set_xlim(0, 1)
        # fig.tight_layout()
        # fig.savefig('PR_curve.png', dpi=300)

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap
