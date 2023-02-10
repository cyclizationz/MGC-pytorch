import torch
from torchvision import transforms as transforms
import PIL.Image as Image

def unpack_image_sequence(image_seq, img_height, img_width, num_sources=None):
    if len(image_seq.shape)==2:
        image_seq = torch.expand_dims(image_seq, -1)
    channels = image_seq.shape[2]
    
    # 假设中间帧是目标渲染图像
    tgt_start_idx = int(img_width * (num_sources // 2))
    tgt_image = torch.slice(image_seq,
                            [0, tgt_start_idx, 0], 
                            [-1, img_width, -1])
    # 中间帧前的帧
    src_img_1 = torch.slice(image_seq,
                            [0,0,0],
                            [-1, tgt_start_idx, -1])
    # 中间帧后的帧
    src_img_2 = torch.slice(image_seq,
                            [0,int(img_width + tgt_start_idx),0],
                            [-1, tgt_start_idx, -1])
    src_img_seq = torch.cat([src_img_1, src_img_2],axis=1)
    # 沿颜色通道堆叠源图像(i.e. [H, W, N*3])
    src_img_stack = torch.cat([torch.slice(src_img_seq,
                                          [0, i * img_width, 0],
                                          [-1, img_width, -1])
                                 for i in range(num_sources)], axis=2)
    src_img_stack.resize([img_height, img_width, num_sources*channels])
    tgt_image.resize([img_height, img_width, num_sources])
    return tgt_image, src_img_stack

def data_augmentation_mul(img, intrinsic, out_h, out_w, matches=None):
    out_h = out_h.type(torch.IntTensor)
    out_w = out_w.type(torch.IntTensor)
    
    # 随机缩放
    def random_scaling(img, intrinsics, matches):
        _, in_h, in_w, _ = torch.unbind(torch.shape(img))
        in_h = in_h.type(torch.FloatTensor)
        in_w = in_w.type(torch.FloatTensor)
        scaling = torch.FloatTensor([2]).uniform_(1.0,1.2)
        x_scaling = scaling[0]
        y_scaling = scaling[0]
        
        out_h = y_scaling*out_h.type(torch.IntTensor)
        out_w = x_scaling*out_w.type(torch.IntTensor)
        
        img = torch.reshape(img,[out_h,out_w])
        
        list_intrinsics = []
        for i in range(intrinsics.shape[1]): # bs num_src+1, 3, 3
            fx = intrinsics[:, i, 0, 0] * x_scaling
            fy = intrinsics[:, i, 1, 1] * y_scaling
            cx = intrinsics[:, i, 0, 2] * x_scaling
            cy = intrinsics[:, i, 0, 2] * y_scaling
            intrinsics_new = make_intrinsics_matrix(fx,fy,cx,cy)
            list_intrinsics.append(intrinsics.new)
        intrinsics = torch.stack(list_intrinsics, axis=1)
        
        if matches is None:
            return img, intrinsics, None
        else:
            x = matches[:, :, :, 0] * x_scaling
            y = matches[:, :, :, 1] * y_scaling
            matches = torch.stack([x,y], axis=3) # bs, tar, num, axis
            return img, intrinsics, matches
        
# Random cropping
    def random_cropping(img, intrinsics, out_h, out_w, matches):
        batch_size, in_h, in_w, _ = torch.unbind(torch.shape[img])
        offset_y = torch.IntTensor([1]).uniform_(0, in_h-out_h+1)[0]
        offset_x = offset_y
        # im = tf.image.crop_to_bounding_box(
        #    im, offset_y, offset_x, out_h, out_w)
        # tf.image.crop_to_bounding_box :This op cuts a rectangular bounding box out of image. 
        # The top-left corner of the bounding box is at offset_height, 
        # offset_width in image, and the lower-right corner is at 
        # offset_height + target_height, offset_width + target_width.
        
        # PIL.Image.crop((left,top,right,bottom))
        img = Image.crop((offset_y, offset_x, offset_y+out_h, offset_x+out_w))
        
        list_intrinsics = []
        for i in range(intrinsics.shape[1]): # bs, num_src+1, 3, 3
            fx = intrinsics[:, i, 0, 0]
            fy = intrinsics[:, i, 1, 1]
            cx = intrinsics[:, i, 0, 2] - torch.FloatTensor(offset_x)
            cy = intrinsics[:, i, 1, 2] - torch.FloatTensor(offset_y)
            intrinsics_new = make_intrinsics_matrix(fx, fy, cx, cy)
            list_intrinsics.append(intrinsics_new)
        intrinsics = torch.stack(list_intrinsics, axis=1)
        
        if matches is None:
            return img, intrinsics, matches
        else:
            x = matches[:, :, :, 0] - torch.FloatTensor(offset_x)
            y = matches[:, :, :, 1] - torch.FloatTensor(offset_y)
            matches = torch.stack([x, y], axis=3)  # bs, tar, num, axis
            return img, intrinsics, matches
    
    batch_size, in_h, in_w, _ = torch.unbind(torch.shape(img))
    img, intrinsics, matches = random_scaling(img, intrinsics, matches)
    img, intrinsics, matches = random_cropping(img, intrinsics, out_h, out_w, matches)
    # im, intrinsics, matches = random_scaling(im, intrinsics, matches, in_h, in_w)
    img = torch.Tensor.to(img, dtype=torch.uint8)

    if matches is None:
        return img, intrinsics, None
    else:
        return img, intrinsics, matches
    
#
def unpack_image_batch_list(image_seq, img_height, img_width, num_source):
    tar_list = []
    src_list = []
    for i in range(image_seq.shape[0]):
        tgt_image, src_image_stack = unpack_image_sequence(image_seq[i], img_height, img_width, num_source)
        tar_list.append(tgt_image)
        src_list.append(src_image_stack)
    tgt_image_b = torch.stack(tar_list)
    src_image_stack_b = torch.stack(src_list)

    list_tar_image = [tgt_image_b]
    list_src_image = [src_image_stack_b[:, :, :, i * 3:(i + 1) * 3] for i in range(num_source)]
    list_image = list_tar_image + list_src_image

    return list_image    
    
# np
def unpack_image_np(image_seq, img_height, img_width, num_source):

    tgt_start_idx = int(img_width * (num_source // 2))

    tgt_image = image_seq[:, tgt_start_idx:tgt_start_idx+img_width, :]
    src_image_1 = image_seq[:, 0:int(img_width * (num_source // 2)), :]
    src_image_2 = image_seq[:, tgt_start_idx+img_width:tgt_start_idx+img_width+int(img_width * (num_source // 2)), :]

    return src_image_1, tgt_image, src_image_2, [tgt_image, src_image_1, src_image_2]