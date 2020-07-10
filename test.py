import argparse
import json

from models.experimental import *
from utils.datasets import *
import pickle
from utils.utils import *


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_pgd(model, X, y, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):

    # pgd parameters
    upper_limit, lower_limit = 1,0
    epsilon = 1
    alpha = 0.05
    attack_iters = opt.pgd_iter
    restarts = 1
    norm = 'l_inf'

    # patch parameters
    patchSize = 155
    start_x = 5
    start_y = 5


    # initialize a mask
    mask = torch.zeros_like(X).cuda()
    mask[:, :, start_y:start_y + patchSize, start_x:start_x + patchSize] = 1 # only the area with Patch is changed to all ones.

    # init loss
    # loss = torch.zeros(1).cuda()
    # loss.requires_grad = True

    # create pgh attack
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        # elif norm == "l_2":
        #     delta.normal_()
        #     d_flat = delta.view(delta.size(0),-1)
        #     n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
        #     r = torch.zeros_like(n).uniform_(0, 1)
        #     delta *= r/n*epsilon
        else:
            raise ValueError


        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True

        # zero out non-patch area
        delta.data = delta.data * mask


        for _ in range(attack_iters):
            inf_out, train_out = model(X + delta)
            # output = model(X)
            # print('ssss pred shape',output[0].shape)
            # if early_stop:
            #     index = torch.where(output.max(1)[1] == y)[0]
            # else:
            index = slice(None,None,None)

            if not isinstance(index, slice) and len(index) == 0:
                break

            # loss = F.cross_entropy(output, y)

            # attack using full loss
            # loss, loss_items = compute_loss(output, y, model)

            # # attack using cls_loss
            tgt_cls_idx = opt.tgt_cls_idx
            # loss, loss_items = cls_loss(output, y, model, tgt_cls_idx)

            loss = cls_loss([x.float() for x in train_out], y, model, tgt_cls_idx)[0]

            print('ssss  loss', _, loss)



            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                # d = torch.clamp(d + alpha * g, min=-epsilon, max=epsilon)
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

            # zero out non-patch area
            mask = torch.zeros_like(delta).cuda()
            mask[:, :, start_y:start_y + patchSize, start_x:start_x + patchSize] = 1 # only the area with Patch is changed to all ones.
            delta.data = delta.data * mask

        # if mixup:
        #     criterion = nn.CrossEntropyLoss(reduction='none')
        #     all_loss = mixup_criterion(criterion, model((X+delta)), y_a, y_b, lam)
        # else:
        # all_loss = F.cross_entropy(model((X+delta)), y, reduction='none')

        # max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        # max_loss = torch.max(max_loss, all_loss)
        max_delta = delta.detach()

    return max_delta




def test(data,
         weights=None,
         batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         merge=False):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = torch_utils.select_device(opt.device, batch_size=batch_size)
        merge = opt.merge  # use Merge NMS

        # Remove previous
        for f in glob.glob('test_batch*.jpg'):
            os.remove(f)

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half = device.type != 'cpu' and torch.cuda.device_count() == 1  # half precision only supported on single-GPU
    if half:
        model.half()  # to FP16

    # Configure
    model.eval()
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
        # ssss
        # dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt,
        #                                hyp=None, augment=False, cache=False, pad=0.5, rect=True)[0]
        dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt,
                                       hyp=None, augment=False, cache=False, pad=0.5, rect=False)[0]

    seen = 0
    names = model.names if hasattr(model, 'names') else model.module.names
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)



        # ssss
        data_cloned = img.clone()


        # pgd attack
        if opt.attack=='adv':
            # # Hyperparameters
            # hyp = {'giou': 3.54,  # giou loss gain
            #        'cls': 37.4,  # cls loss gain
            #        'cls_pw': 1.0,  # cls BCELoss positive_weight
            #        'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
            #        'obj_pw': 1.0,  # obj BCELoss positive_weight
            #        'iou_t': 0.20,  # iou training threshold
            #        # ssss
            #         # 'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
            #        'lr0': 0.001,  # initial learning rate (SGD=5E-3, Adam=5E-4)
            #        'lrf': 0.0005,  # final learning rate (with cos scheduler)
            #        'momentum': 0.937,  # SGD momentum
            #        'weight_decay': 0.0005,  # optimizer weight decay
            #        'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
            #        'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
            #        'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
            #        'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
            #        'degrees': 1.98 * 0,  # image rotation (+/- deg)
            #        'translate': 0.05 * 0,  # image translation (+/- fraction)
            #        'scale': 0.05 * 0,  # image scale (+/- gain)
            #        'shear': 0.641 * 0}  # image shear (+/- deg)

            # data = opt.data
            # data_dict = parse_data_cfg(data)
            # nc = 1 if opt.single_cls else int(data_dict['classes'])  # number of classes
            # hyp['cls'] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset

            # model.nc = nc  # attach number of classes to model
            # model.hyp = hyp  # attach hyperparameters to model
            # model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
            # # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights


            pgd_delta = attack_pgd(model, img, targets)
            data_cloned = data_cloned + pgd_delta






        # Disable gradients
        with torch.no_grad():
            # Run model
            t = torch_utils.time_synchronized()
            inf_out, train_out = model(data_cloned, augment=augment)  # inference and training outputs

            # print('ssss img.shape',  img.shape)
            # print('ssss inf_out.shape', inf_out.shape, len(train_out), train_out[0].shape, train_out[1].shape,train_out[2].shape)
            # raise

            t0 += torch_utils.time_synchronized() - t

            # Compute loss
            if training:  # if model has loss hyperparameters
                loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # GIoU, obj, cls

            # Run NMS
            t = torch_utils.time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, merge=merge)
            t1 += torch_utils.time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])],
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero():
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if batch_i < 2:
            f = 'test_batch%g_gt.jpg' % batch_i  # filename
            plot_images(img, targets, paths, f, names)  # ground truth
            f = 'test_batch%g_pred.jpg' % batch_i
            plot_images(img, output_to_target(output, width, height), paths, f, names)  # predictions
            if opt.attack!='None':
                f = 'test_batch%g_pred_atk.jpg' % batch_i
                plot_images(data_cloned, output_to_target(output, width, height), paths, f, names)  # predictions

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    if save_json and map50 and len(jdict):
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataloader.dataset.img_files]
        f = 'detections_val2017_%s_results.json' % \
            (weights.split(os.sep)[-1].replace('.pt', '') if isinstance(weights, str) else '')  # filename
        print('\nCOCO mAP with pycocotools... saving %s...' % f)
        with open(f, 'w') as file:
            json.dump(jdict, file)

        # try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        cocoGt = COCO(glob.glob('../coco/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
        cocoDt = cocoGt.loadRes(f)  # initialize COCO pred api

        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds  # image IDs to evaluate

        # ssss
        if opt.tgt_cls!='None':
            cocoEval.params.catIds = [opt.tgt_cls_id]


        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        map, map50 = cocoEval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        # except:
        #     print('WARNING: pycocotools must be installed with numpy==1.17 to run correctly. '
        #           'See https://github.com/cocodataset/cocoapi/issues/356')

    # Return results
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')

    # ssss
    parser.add_argument('--attack', type=str, default='None', help='what attack to use')
    parser.add_argument('--tgt_cls', type=str, default='None', help='which class to attack')
    parser.add_argument('--log', type=str, default='None', help='where to log results')
    parser.add_argument('--pgd_iter', type=int, default=0, help='number of iterations for pgd attack')

    opt = parser.parse_args()
    opt.save_json = opt.save_json or opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file

    if opt.tgt_cls!='None':
        with open('cls_map.pickle', 'rb') as handle:
            cls_map = pickle.load(handle)
        opt.tgt_cls_id = cls_map[opt.tgt_cls]

        with open('clsIdx_map.pickle', 'rb') as handle:
            clsIdx_map = pickle.load(handle)
        opt.tgt_cls_idx = clsIdx_map[opt.tgt_cls]

        opt.data = check_file(f'data/by_class/{opt.tgt_cls}.yaml')  # check file

    else:
        opt.data = check_file(opt.data)  # check file


    print(opt)

    # task = 'val', 'test', 'study'
    if opt.task in ['val', 'test']:  # (default) run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        for weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
            f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem)  # filename to save to
            x = list(range(352, 832, 64))  # x axis
            y = []  # y axis
            for i in x:  # img-size
                print('\nRunning %s point %s...' % (f, i))
                r, _, t = test(opt.data, weights, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        # plot_study_txt(f, x)  # plot
