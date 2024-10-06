import os
import argparse
import numpy as np
from tqdm import tqdm
import torch.utils.data

from main_model import EnsembleModel
from util.dataloaders import get_eval_loaders
from util.common import check_eval_dirs, compute_p_r_f1_miou_oa, gpu_info, SaveResult, ScaleInOutput
from util.AverageMeter import AverageMeter, RunningMetrics
running_metrics =  RunningMetrics(2)
def eval(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gpu_info()  
    save_path, result_save_path = check_eval_dirs()

    save_results = SaveResult(result_save_path)
    save_results.prepare()

    model = EnsembleModel(opt.ckp_paths, device, input_size=opt.input_size)

    opt.dual_label = False
    eval_loader = get_eval_loaders(opt)

    p, r, f1, miou, oa, avg_loss = eval_for_metric(model, eval_loader, tta=opt.tta)

    save_results.show(p, r, f1, miou, oa)
    print("F1-mean: {}".format(f1.mean()))
    print("mIOU-mean: {}".format(miou.mean()))


def eval_for_metric(model, eval_loader, criterion=None, tta=False, input_size=512):   
    avg_loss = 0
    val_loss = torch.tensor([0])
    scale = ScaleInOutput(input_size)

    tn_fp_fn_tp = [np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])]

    model.eval()
    with torch.no_grad():
        eval_tbar = tqdm(eval_loader)
        for i, (batch_img1, batch_img2, batch_label1, batch_label2, _) in enumerate(eval_tbar):
            eval_tbar.set_description("evaluating...eval_loss: {}".format(avg_loss))
            batch_img1 = batch_img1.float().cuda()
            batch_img2 = batch_img2.float().cuda()
            batch_label1 = batch_label1.long().cuda()
            batch_label2 = batch_label2.long().cuda()

            if criterion is not None:
                batch_img1, batch_img2 = scale.scale_input((batch_img1, batch_img2))

            _, _, outs = model(batch_img1, batch_img2, tta)

            if not isinstance(outs, tuple):
                outs = (outs, outs)
            labels = (batch_label1, batch_label2)
            
            if criterion is not None:
                outs = scale.scale_output(outs)
                val_loss = criterion(outs, labels)  
                _, cd_pred1 = torch.max(outs[0], 1)  
                _, cd_pred2 = torch.max(outs[1], 1)
            else:
                cd_pred1 = outs[0]
                cd_pred2 = outs[1]
            cd_preds = (cd_pred1, cd_pred2)
            running_metrics.update(labels[0].data.cpu().numpy(),cd_preds[0].data.cpu().numpy())
            avg_loss = (avg_loss * i + val_loss.cpu().detach().numpy()) / (i + 1)
            for j, (cd_pred, label) in enumerate(zip(cd_preds, labels)):
                tn = ((cd_pred == 0) & (label == 0)).int().sum().cpu().numpy()
                fp = ((cd_pred == 1) & (label == 0)).int().sum().cpu().numpy()
                fn = ((cd_pred == 0) & (label == 1)).int().sum().cpu().numpy()
                tp = ((cd_pred == 1) & (label == 1)).int().sum().cpu().numpy()
                assert tn+fp+fn+tp == np.prod(batch_label1.shape)

                tn_fp_fn_tp[j] += [tn, fp, fn, tp]
            
    p, r, f1, miou, oa = compute_p_r_f1_miou_oa(tn_fp_fn_tp)
    score = running_metrics.get_scores()

    return p, r, f1, miou, oa, avg_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Eval')

    parser.add_argument("--ckp-paths", type=str,
                        default=[
                            "./runs/train/1/best_ckp/",
                        ])

    parser.add_argument("--cuda", type=str, default="0")  # GPU编号
    parser.add_argument("--dataset-dir", type=str, default="/mnt/Disk1/CDData/LEVIR-CD/")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--tta", type=bool, default=False)

    opt = parser.parse_args()
    print("\n" + "-" * 30 + "OPT" + "-" * 30)
    print(opt)

    eval(opt)
