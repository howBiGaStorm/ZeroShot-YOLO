"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate


def custom_collate_fn(batch):
    items = list(zip(*batch))
    items[0] = default_collate(items[0])
    items[1] = list(items[1])
    return items


def post_processing(logits, image_size, gt_classes, anchors, conf_threshold, nms_threshold):
    num_anchors = len(anchors)
    anchors = torch.Tensor(anchors)

    if isinstance(logits, Variable):
        logits = logits.data

    if logits.dim() == 3:
        logits.unsqueeze_(0)

    batch = logits.size(0)
    h = logits.size(2)
    w = logits.size(3)

    # Compute xc,yc, w,h, box_score on Tensor
    lin_x = torch.linspace(0, w - 1, w).repeat(h, 1).view(h * w)
    lin_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().contiguous().view(h * w)
    anchor_w = anchors[:, 0].contiguous().view(1, num_anchors, 1)
    anchor_h = anchors[:, 1].contiguous().view(1, num_anchors, 1)
    if torch.cuda.is_available():
        lin_x = lin_x.cuda()
        lin_y = lin_y.cuda()
        anchor_w = anchor_w.cuda()
        anchor_h = anchor_h.cuda()

    logits = logits.view(batch, num_anchors, -1, h * w) # [1,5,25,169]
    logits[:, :, 0, :].sigmoid_().add_(lin_x).div_(w) #中心点ｘ
    logits[:, :, 1, :].sigmoid_().add_(lin_y).div_(h) # 中心点y
    logits[:, :, 2, :].exp_().mul_(anchor_w).div_(w) # 宽度
    logits[:, :, 3, :].exp_().mul_(anchor_h).div_(h) # 高度
    logits[:, :, 4, :].sigmoid_() # 类别

    with torch.no_grad():
        cls_scores = torch.nn.functional.softmax(logits[:, :, 5:, :], 2)
    cls_max, cls_max_idx = torch.max(cls_scores, 2)
    cls_max_idx = cls_max_idx.float()
    cls_max.mul_(logits[:, :, 4, :])

    score_thresh = cls_max > conf_threshold
    score_thresh_flat = score_thresh.view(-1)

    if score_thresh.sum() == 0:
        predicted_boxes = []
        for i in range(batch):
            predicted_boxes.append(torch.Tensor([]))
    else:
        coords = logits.transpose(2, 3)[..., 0:4]
        coords = coords[score_thresh[..., None].expand_as(coords)].view(-1, 4)
        scores = cls_max[score_thresh]
        idx = cls_max_idx[score_thresh]
        detections = torch.cat([coords, scores[:, None], idx[:, None]], dim=1)

        max_det_per_batch = num_anchors * h * w
        slices = [slice(max_det_per_batch * i, max_det_per_batch * (i + 1)) for i in range(batch)]
        det_per_batch = torch.IntTensor([score_thresh_flat[s].int().sum() for s in slices])
        split_idx = torch.cumsum(det_per_batch, dim=0)

        # Group detections per image of batch
        predicted_boxes = []
        start = 0
        for end in split_idx:
            predicted_boxes.append(detections[start: end])
            start = end

    selected_boxes = []
    for boxes in predicted_boxes:
        if boxes.numel() == 0:
            return boxes

        a = boxes[:, :2]
        b = boxes[:, 2:4]
        bboxes = torch.cat([a - b / 2, a + b / 2], 1)
        scores = boxes[:, 4]

        # Sort coordinates by descending score
        scores, order = scores.sort(0, descending=True)
        x1, y1, x2, y2 = bboxes[order].split(1, 1)

        # Compute dx and dy between each pair of boxes (these mat contain every pair twice...)
        dx = (x2.min(x2.t()) - x1.max(x1.t())).clamp(min=0)
        dy = (y2.min(y2.t()) - y1.max(y1.t())).clamp(min=0)

        # Compute iou
        intersections = dx * dy
        areas = (x2 - x1) * (y2 - y1)
        unions = (areas + areas.t()) - intersections
        ious = intersections / unions

        # Filter based on iou (and class)
        conflicting = (ious > nms_threshold).triu(1)

        keep = conflicting.sum(0).byte()
        keep = keep.cpu()
        conflicting = conflicting.cpu()

        keep_len = len(keep) - 1
        for i in range(1, keep_len):
            if keep[i] > 0:
                keep -= conflicting[i]
        if torch.cuda.is_available():
            keep = keep.cuda()

        keep = (keep == 0)
        selected_boxes.append(boxes[order][keep[:, None].expand_as(boxes)].view(-1, 6).contiguous())

    final_boxes = []
    for boxes in selected_boxes:
        if boxes.dim() == 0:
            final_boxes.append([])
        else:
            boxes[:, 0:3:2] *= image_size
            boxes[:, 0] -= boxes[:, 2] / 2
            boxes[:, 1:4:2] *= image_size
            boxes[:, 1] -= boxes[:, 3] / 2

            final_boxes.append([[box[0].item(), box[1].item(), box[2].item(), box[3].item(), box[4].item(),
                                 gt_classes[int(box[5].item())]] for box in boxes])
    return final_boxes


def out_processing(tl,ts, tc, image_size, anchors, conf_threshold, nms_threshold, seen_vec): # [b,20,13,13]  [b,5,13,13]
    num_anchors = len(anchors)
    num_seen = seen_vec.shape[0]
    anchors = torch.Tensor(anchors)
    if isinstance(tl, Variable):
        tl = tl.data
    if isinstance(tc, Variable):
        tc = tc.data

    if tl.dim() == 3:
        tl.unsqueeze_(0)
    if tc.dim() == 3:
        tc.unsqueeze_(0)

    batch = tl.size(0)
    h = tl.size(2)
    w = tl.size(3)

    # Compute xc,yc, w,h, box_score on Tensor
    lin_x = torch.linspace(0, w - 1, w).repeat(h, 1).view(h * w)
    lin_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().contiguous().view(h * w)
    anchor_w = anchors[:, 0].contiguous().view(1, num_anchors, 1)
    anchor_h = anchors[:, 1].contiguous().view(1, num_anchors, 1)
    if torch.cuda.is_available():
        lin_x = lin_x.cuda()
        lin_y = lin_y.cuda()
        anchor_w = anchor_w.cuda()
        anchor_h = anchor_h.cuda()

    tl = tl.view(batch,num_anchors,-1,h*w)# [b,5,4,169]
    tl[:,:,0,:].sigmoid_().add_(lin_x).div_(w)
    tl[:, :, 1, :].sigmoid_().add_(lin_y).div_(h)
    tl[:, :, 2, :].exp_().mul_(anchor_w).div_(w)
    tl[:, :, 3, :].exp_().mul_(anchor_h).div_(h)
    tc = tc.view(batch,num_anchors,h*w).sigmoid_() # [1,5,169]

    ts = ts.view(batch,-1,64,1).repeat(1,1,1,num_seen)
    seen_attrs = torch.zeros(batch, num_anchors * h* w, 64, num_seen, requires_grad=False)
    for i, seen_attr in enumerate(torch.Tensor(seen_vec)):
        seen_attrs[:, :, :, i] = seen_attr.view(1, 1, 64).repeat(batch, num_anchors * h * w, 1)
    seen_attrs = seen_attrs.cuda()
    cos_sim = torch.nn.CosineSimilarity(dim=2, eps=1e-8)

    sim_score = cos_sim(ts,seen_attrs)
    score_max,score_max_idx = sim_score.max(2)
    score_max = score_max.view(batch,num_anchors,h*w)
    score_max_idx = score_max_idx.view(batch,num_anchors,h*w).float()
    score_max.mul_(tc)


    score_thresh = score_max > conf_threshold
    score_thresh_flat = score_thresh.view(-1) # 平铺得分阈值 [5*13*13]
    # print(score_thresh_flat.shape)

    if score_thresh.sum() == 0: # 即都是０，也就是都低于阈值，输出为空
        predicted_boxes = []
        for i in range(batch):
            predicted_boxes.append(torch.Tensor([]))
    else:
        # coords = logits.transpose(2, 3)[..., 0:4]
        coords = tl.transpose(2,3)
        coords = coords[score_thresh[..., None].expand_as(coords)].view(-1, 4) # 满足阈值的位置信息保留[n,4] n代表满足的框的个数

        # scores = cls_max[score_thresh] #　满足阈值的置信度信息保留 [n]
        # idx = cls_max_idx[score_thresh] # 满足阈值的类别信息保留 [n]
        scores = score_max[score_thresh]

        # detections = torch.cat([coords, scores[:, None], idx[:, None]], dim=1) # [n,6]
        detections = torch.cat([coords, scores[:, None]], dim=1)


        max_det_per_batch = num_anchors * h * w # 每个ｂａｔｃｈ　最多５×１３×１３个框
        slices = [slice(max_det_per_batch * i, max_det_per_batch * (i + 1)) for i in range(batch)]
        det_per_batch = torch.IntTensor([score_thresh_flat[s].int().sum() for s in slices])
        split_idx = torch.cumsum(det_per_batch, dim=0)

        # Group detections per image of batch
        predicted_boxes = [] # 存的每个图片的满足初步筛选（阈值筛选）的信息，例如[0.6307, 0.4467, 0.0336, 0.0426, 0.8983, 3.0000]
        start = 0
        for end in split_idx:
            predicted_boxes.append(detections[start: end])
            start = end
        # print(predicted_boxes)

    selected_boxes = []
    for boxes in predicted_boxes:
        if boxes.numel() == 0:
            return boxes

        a = boxes[:, :2]
        b = boxes[:, 2:4]
        bboxes = torch.cat([a - b / 2, a + b / 2], 1)
        scores = boxes[:, 4]

        # Sort coordinates by descending score　通过得分降序，对候选框们排序
        scores, order = scores.sort(0, descending=True)
        x1, y1, x2, y2 = bboxes[order].split(1, 1)

        # Compute dx and dy between each pair of boxes (these mat contain every pair twice...)
        dx = (x2.min(x2.t()) - x1.max(x1.t())).clamp(min=0)
        dy = (y2.min(y2.t()) - y1.max(y1.t())).clamp(min=0)

        # Compute iou
        intersections = dx * dy
        areas = (x2 - x1) * (y2 - y1)
        unions = (areas + areas.t()) - intersections
        ious = intersections / unions

        # Filter based on iou (and class)
        conflicting = (ious > nms_threshold).triu(1)

        keep = conflicting.sum(0).byte()
        keep = keep.cpu()
        conflicting = conflicting.cpu()

        keep_len = len(keep) - 1
        for i in range(1, keep_len):
            if keep[i] > 0:
                keep -= conflicting[i]
        if torch.cuda.is_available():
            keep = keep.cuda()

        keep = (keep == 0)
        # selected_boxes.append(boxes[order][keep[:, None].expand_as(boxes)].view(-1, 6).contiguous())
        selected_boxes.append(boxes[order][keep[:, None].expand_as(boxes)].view(-1, 5).contiguous())

    final_boxes = []
    for boxes in selected_boxes:
        if boxes.dim() == 0:
            final_boxes.append([])
        else:
            boxes[:, 0:3:2] *= image_size
            boxes[:, 0] -= boxes[:, 2] / 2
            boxes[:, 1:4:2] *= image_size
            boxes[:, 1] -= boxes[:, 3] / 2

            final_boxes.append([[box[0].item(), box[1].item(), box[2].item(), box[3].item(), box[4].item()] for box in boxes])
    return final_boxes