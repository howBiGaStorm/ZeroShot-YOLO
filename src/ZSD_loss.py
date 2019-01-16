"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import math
import torch
import torch.nn as nn


class ZSDLoss(nn.modules.loss._Loss): # 继承LOSS类
    # The loss I borrow from LightNet repo.
    def __init__(self, num_classes, anchors, seen_attrs, reduction=32, coord_scale=1.0, noobject_scale=1.0,
                 object_scale=5.0, class_scale=1.0, thresh=0.6,GPU = False):
        super(ZSDLoss, self).__init__()
        self.num_classes = num_classes  # 20
        self.num_anchors = len(anchors)  # 5
        self.anchor_step = len(anchors[0])  # 2
        self.anchors = torch.Tensor(anchors)
        self.reduction = reduction  # 32

        self.coord_scale = coord_scale  # 1
        self.noobject_scale = noobject_scale  # 1
        self.object_scale = object_scale  # 5
        self.class_scale = class_scale  # 1
        self.thresh = thresh
        self.seen_vec = seen_attrs
        self.GPU = GPU

    def forward(self, tl, ts, tc, target): # 计算输出与真实值之间的损失 ?输出的大小是[1,125,13,13]target格式是？

        batch_size = tl.data.size(0)
        height = tl.data.size(2)
        width = tl.data.size(3)

        tl = tl.view(batch_size,self.num_anchors,4,height * width)
        coord = torch.zeros_like(tl)
        coord[:, :, :2, :] = tl[:, :, :2, :].sigmoid()
        coord[:, :, 2:4, :] = tl[:, :, 2:4, :]

        # Create prediction boxes  预测边框
        pred_boxes = torch.FloatTensor(batch_size * self.num_anchors * height * width, 4)
        lin_x = torch.arange(0, width,1).repeat(height, 1).view(height * width).float()
        lin_y = torch.arange(0, height,1).repeat(width, 1).t().contiguous().view(height * width).float()
        anchor_w = self.anchors[:, 0].contiguous().view(self.num_anchors, 1)
        anchor_h = self.anchors[:, 1].contiguous().view(self.num_anchors, 1)


        pred_boxes = pred_boxes.cuda()
        lin_x = lin_x.cuda()
        lin_y = lin_y.cuda()
        anchor_w = anchor_w.cuda()
        anchor_h = anchor_h.cuda()

        pred_boxes[:, 0] = (coord[:, :, 0].detach() + lin_x).view(-1)
        pred_boxes[:, 1] = (coord[:, :, 1].detach() + lin_y).view(-1)
        pred_boxes[:, 2] = (coord[:, :, 2].detach().exp() * anchor_w).view(-1)
        pred_boxes[:, 3] = (coord[:, :, 3].detach().exp() * anchor_h).view(-1)
        pred_boxes = pred_boxes.cpu()

        # Get target values

        coord_mask, conf_mask, obj_mask, noobj_mask,semantic_mask, tcoord, tvec = self.build_targets(pred_boxes, target, height, width)
        coord_mask = coord_mask.expand_as(tcoord)


        coord_mask = coord_mask.cuda()
        conf_mask = conf_mask.cuda()
        obj_mask = obj_mask.cuda()
        noobj_mask = noobj_mask.cuda()
        tcoord = tcoord.cuda()
        tvec = tvec.cuda()  # [b,5,169,64]

        # ############################# loss coord #####################################
        # Compute losses
        mse = nn.MSELoss(size_average=False)
        self.loss_coord = self.coord_scale * mse(coord * coord_mask, tcoord * coord_mask) / batch_size  # 位置损失

        # ############################# loss conf #####################################
        tc = tc.view(batch_size,self.num_anchors,-1)
        self.loss_conf = 5*mse(tc*conf_mask,obj_mask)/batch_size

        #############################    loss  semantic  ####################################
        tsf = ts.view(batch_size,self.num_anchors,height * width,-1) # [b,5,169,64]

        seman_mask = obj_mask.view(batch_size, self.num_anchors, height * width,1).repeat(1,1,1,64)
        # loss前半部分，有物体时的余弦相似度
        cos_sim = torch.nn.CosineSimilarity(dim=3, eps=1e-8)
        obj_vec = tsf*seman_mask # [b,5,169,64]
        obj_sim = cos_sim(obj_vec,tvec)


        # loss后半部分，无物体时的和seen——vec的最大余弦相似度
        num_seen = self.seen_vec.shape[0]
        tsb = ts.view(batch_size, -1 ,64,1).repeat(1,1,1,num_seen) # [b,845,64,2]
        seen_attrs = torch.zeros( batch_size, self.num_anchors * height * width, 64, num_seen, requires_grad=False) # [b,645,64,2]

        for i,seen_attr in enumerate(torch.Tensor(self.seen_vec)):
            seen_attrs[:,:,:,i] = seen_attr.view(1,1,64).repeat(batch_size,self.num_anchors*height*width,1)

        seen_attrs = seen_attrs.cuda()
        cos_sim = torch.nn.CosineSimilarity(dim=2, eps=1e-8)
        noobj_sim_all = cos_sim(tsb, seen_attrs)
        noobj_sim, _ = noobj_sim_all.max(2)  # [5,845]

        noobj_sim = noobj_sim.view(batch_size,self.num_anchors,height*width)

        # ###################  相似度汇总 得到相似度矩阵   #########################
        sim = noobj_sim*noobj_mask+obj_sim
        self.loss_semantic = 5*mse(sim*conf_mask , obj_mask)/batch_size

        self.loss_tot = self.loss_coord +self.loss_conf+self.loss_semantic  # 总损失

        return  self.loss_tot,self.loss_coord ,self.loss_conf ,self.loss_semantic

    def build_targets(self, pred_boxes, ground_truth, height, width):
        batch_size = len(ground_truth)
        # 存储 有目标的为1,每目标的为（根号下1/5），是为了obj和noobj的5倍的均衡
        conf_mask = torch.ones(batch_size, self.num_anchors, height * width, requires_grad=False) * (math.sqrt(1.0/self.object_scale))
        # 有目标1,没目标为0
        coord_mask = torch.zeros(batch_size, self.num_anchors, 1, height * width, requires_grad=False)
        # 有目标1,没目标为0, 字节形，
        semantic_mask = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False).byte()
        obj_mask = torch.zeros(batch_size, self.num_anchors, height * width, requires_grad=False)
        # 存储真实的BBOX信息
        tcoord = torch.zeros(batch_size, self.num_anchors, 4, height * width, requires_grad=False)
        # 存储真实的类的语义向量信息
        tvec = torch.zeros(batch_size, self.num_anchors, height * width, 64,requires_grad=False)
        # 存一个反向的mask，即有目标为0,没目标为1
        noobj_mask = torch.ones(batch_size, self.num_anchors, height * width, requires_grad=False)



        for b in range(batch_size):
            if len(ground_truth[b]) == 0:
                continue

            # Build up tensors
            cur_pred_boxes = pred_boxes[
                             b * (self.num_anchors * height * width):(b + 1) * (self.num_anchors * height * width)]
            if self.anchor_step == 4:
                anchors = self.anchors.clone()
                anchors[:, :2] = 0
            else:
                anchors = torch.cat([torch.zeros_like(self.anchors), self.anchors], 1)
            gt = torch.zeros(len(ground_truth[b]), 4)
            for i, anno in enumerate(ground_truth[b]):
                gt[i, 0] = (anno[0] + anno[2] / 2) / self.reduction
                gt[i, 1] = (anno[1] + anno[3] / 2) / self.reduction
                gt[i, 2] = anno[2] / self.reduction
                gt[i, 3] = anno[3] / self.reduction

            # Set confidence mask of matching detections to 0
            iou_gt_pred = bbox_ious(gt, cur_pred_boxes)
            mask = (iou_gt_pred > self.thresh).sum(0) >= 1
            conf_mask[b][mask.view_as(conf_mask[b])] = 0

            # Find best anchor for each ground truth
            gt_wh = gt.clone()
            gt_wh[:, :2] = 0
            iou_gt_anchors = bbox_ious(gt_wh, anchors)
            _, best_anchors = iou_gt_anchors.max(1)

            # Set masks and target values for each ground truth
            for i, anno in enumerate(ground_truth[b]):
                gi = min(width - 1, max(0, int(gt[i, 0])))
                gj = min(height - 1, max(0, int(gt[i, 1])))
                best_n = best_anchors[i]
                iou = iou_gt_pred[i][best_n * height * width + gj * width + gi]
                coord_mask[b][best_n][0][gj * width + gi] = 1
                # cls_mask[b][best_n][gj * width + gi] = 1
                obj_mask[b][best_n][gj * width + gi] = 1
                noobj_mask[b][best_n][gj * width + gi] = 0
                semantic_mask[b][best_n][gj * width + gi] = 1
                conf_mask[b][best_n][gj * width + gi] = 1.0
                tcoord[b][best_n][0][gj * width + gi] = gt[i, 0] - gi
                tcoord[b][best_n][1][gj * width + gi] = gt[i, 1] - gj
                tcoord[b][best_n][2][gj * width + gi] = math.log(max(gt[i, 2], 1.0) / self.anchors[best_n, 0])
                tcoord[b][best_n][3][gj * width + gi] = math.log(max(gt[i, 3], 1.0) / self.anchors[best_n, 1])
                tvec[b][best_n][gj * width + gi] =  torch.FloatTensor(anno[5])

        return coord_mask, conf_mask, obj_mask,noobj_mask,semantic_mask, tcoord, tvec


def bbox_ious(boxes1, boxes2): # 计算iou
    b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
    b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
    b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
    b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + areas2.t()) - intersections

    return intersections / unions


# def cos_sim(vector1,vector2):
#     dot_product = 0.0
#     normA = 0.0
#     normB = 0.0
#     for a,b in zip(vector1,vector2):
#         dot_product += a*b
#         normA += a**2
#         normB += b**2
#     if normA == 0.0 or normB==0.0:
#         return 0
#     else:
#         return dot_product / ((normA*normB)**0.5)