from torch.autograd import Function
import torch
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import torch.nn as nn

from collections import namedtuple
import cupy
from string import Template


Stream = namedtuple('Stream', ['ptr'])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


@cupy._util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


CUDA_NUM_THREADS = 1024

kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


_crop_rev_kernel = kernel_loop + '''
extern "C"
__global__ void crop_rev_forward_kernel(
const ${Dtype}* bottom_data, const ${Dtype}* index_data, ${Dtype}* top_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int b = index / ${num_hw} / ${top_height} / ${top_width};
    const int n = (index / ${top_height} / ${top_width}) % ${num_hw};
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    ${Dtype} value = 0;
    const int pt_center_y = index_data[b * ${num_hw} * 2 + n * 2 + 1];
    const int pt_center_x = index_data[b * ${num_hw} * 2 + n * 2];
    const int dst_id = b * ${num_hw} * ${top_height} * ${top_width} + n * ${top_height} * ${top_width} + h * ${top_width} + w;
    const int y_bottom = h - pt_center_y + ${hk_h};
    const int x_bottom = w - pt_center_x + ${hk_w};
    if ((y_bottom >= 0) && (y_bottom < ${bottom_height}) && (x_bottom >= 0) && (x_bottom < ${bottom_width})) {
      const int src_id = b * ${num_hw} * ${bottom_height} * ${bottom_width} + n * ${bottom_height} * ${bottom_width} + y_bottom * ${bottom_width} + x_bottom;
      value = bottom_data[src_id];
    }
    else {
      value = 0;
    }
    top_data[dst_id] = value;
  }
}
'''


_crop_rev_kernel_backward_grad_input = kernel_loop + '''
extern "C"
__global__ void crop_rev_backward_grad_input_kernel(
    const ${Dtype}* bottom_diff, const ${Dtype}* index_data, ${Dtype}* top_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int b = index / ${num_hw} / ${top_height} / ${top_width};
    const int n = (index / ${top_height} / ${top_width}) % ${num_hw};
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    ${Dtype} value = 0;
    const int pt_center_y = index_data[b * ${num_hw} * 2 + n * 2 + 1];
    const int pt_center_x = index_data[b * ${num_hw} * 2 + n * 2];
    const int dst_id = b * ${num_hw} * ${top_height} * ${top_width} + n * ${top_height} * ${top_width} + h * ${top_width} + w;
    const int y_bottom = pt_center_y - ${hk_h} + h;
    const int x_bottom = pt_center_x - ${hk_w} + w;
    if ((y_bottom >= 0) && (y_bottom < ${bottom_height}) && (x_bottom >= 0) && (x_bottom < ${bottom_width})) {
      const int src_id = b * ${num_hw} * ${bottom_height} * ${bottom_width} + n * ${bottom_height} * ${bottom_width} + y_bottom * ${bottom_width} + x_bottom;
      value = bottom_diff[src_id];
    }
    else {
      value = 0;
    }
    top_diff[dst_id] = value;
  }
}
'''


class _crop_rev(Function):
    @staticmethod
    def forward(ctx, input, index, hks, hws):
        assert input.dim() == 4 and input.is_cuda
        assert index.dim() == 3 and index.is_cuda
        batch_size, num_hw, bottom_height, bottom_width = input.size()
        hk_h, hk_w = hks
        h, w = hws
        top_height = h
        top_width = w

        output = input.new(batch_size, num_hw, top_height, top_width)
        n = output.numel()

        with torch.cuda.device_of(input):
            f = load_kernel('crop_rev_forward_kernel', _crop_rev_kernel, Dtype=Dtype(input), nthreads=n,
                            num_hw=num_hw, bottom_height=bottom_height, bottom_width=bottom_width,
                            top_height=top_height, top_width=top_width,
                            hk_h=hk_h, hk_w=hk_w)
            f(block=(CUDA_NUM_THREADS,1,1),
              grid=(GET_BLOCKS(n),1,1),
              args=[input.data_ptr(), index.data_ptr(), output.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        ctx.save_for_backward(input, index)
        ctx.hk_h, ctx.hk_w = hk_h, hk_w
        ctx.h, ctx.w = h, w

        return output


    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda and grad_output.is_contiguous()
        input, index = ctx.saved_tensors
        h, w = ctx.h, ctx.w
        hk_h, hk_w = ctx.hk_h, ctx.hk_w

        batch_size, num_hw, bottom_height, bottom_width = grad_output.size()
        assert bottom_height * bottom_width == h * w
        top_height, top_width = input.size()[2:]

        opt = dict(Dtype=Dtype(grad_output),
                  num_hw=num_hw, bottom_height=bottom_height, bottom_width=bottom_width,
                  top_height=top_height, top_width=top_width,
                  hk_h=hk_h, hk_w=hk_w)

        with torch.cuda.device_of(input):
            if ctx.needs_input_grad:
                grad_input = input.new(input.size())

                n = grad_input.numel()
                opt['nthreads'] = n

                f = load_kernel('crop_rev_backward_grad_input_kernel',
                                _crop_rev_kernel_backward_grad_input, **opt)
                f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(GET_BLOCKS(n),1,1),
                  args=[grad_output.data_ptr(), index.data_ptr(), grad_input.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        return grad_input, None, None, None
 

def Crop_Rev_cu(x, idx, hks, hws):
    """ crop_rev kernel
    x.shape = b, hw, k, k
    """
    if x.is_cuda:
        out = _crop_rev.apply(x, idx, hks, hws)
    else:
        raise NotImplementedError
    return out

