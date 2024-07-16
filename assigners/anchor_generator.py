"""
@author supermantx
@date 2024/7/4 15:51
根据锚点生成锚框
"""
import torch


def generate_anchor(img_size: (tuple, int), strides, grid_cell_size=5.0, grid_cell_offset=0.5, device='cpu'):
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    img_size = torch.tensor(img_size, device=device)
    anchors = []
    anchor_points = []
    stride_tensor = []
    num_anchors_list = []
    for i, stride in enumerate(strides):
        h, w = img_size / stride
        cell_half_size = grid_cell_size * stride * 0.5
        shift_x = (torch.arange(w, device=device) + grid_cell_offset) * stride
        shift_y = (torch.arange(h, device=device) + grid_cell_offset) * stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        anchor = torch.stack([shift_x - cell_half_size, shift_y - cell_half_size,
                              shift_x + cell_half_size, shift_y + cell_half_size], dim=-1).float()
        anchor_point = torch.stack([shift_x, shift_y], dim=-1).float()
        anchors.append(anchor.reshape([-1, 4]))
        anchor_points.append(anchor_point.reshape([-1, 2]))
        num_anchors_list.append(len(anchors[-1]))
        stride_tensor.append(torch.full([len(anchors[-1]), 1], stride, dtype=torch.float, device=device))
    anchors = torch.cat(anchors)
    anchor_points = torch.cat(anchor_points)
    stride_tensor = torch.cat(stride_tensor)
    num_anchors_list = torch.tensor(num_anchors_list)
    return anchors, anchor_points, num_anchors_list, stride_tensor


if __name__ == '__main__':
    print(generate_anchor(640, [8, 16, 32])[0])
