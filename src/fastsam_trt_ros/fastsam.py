#!/usr/bin/env python

import cv2
import torch
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

import tensorrt as trt
from ultralytics.engine.results import Results
from ultralytics.utils import ops
import threading

TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine, max_boxes, total_classes):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    out_shapes = []
    input_shapes = []
    out_names = []
    max_batch_size = engine.get_profile_shape(0, 0)[2][0]
    print('Profile shape: ', engine.get_profile_shape(0, 0))
    # max_batch_size = 1
    for binding in engine:
        binding_shape = engine.get_binding_shape(binding)
        #Fix -1 dimension for proper memory allocation for batch_size > 1
        if binding == 'input':
            max_width = engine.get_profile_shape(0, 0)[2][3]
            max_height = engine.get_profile_shape(0, 0)[2][2]
            size = max_batch_size * max_width * max_height * 3
        else:
            size = max_batch_size * max_boxes * (total_classes + 5)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
            input_shapes.append(engine.get_binding_shape(binding))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            #Collect original output shapes and names from engine
            out_shapes.append(engine.get_binding_shape(binding))
            out_names.append(binding)
    return inputs, outputs, bindings, stream, input_shapes, out_shapes, out_names, max_batch_size

def allocate_buffers_nms(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    out_shapes = []
    input_shapes = []
    out_names = []
    max_batch_size = engine.get_profile_shape(0, 0)[2][0]
    print('Profile shape: ', engine.get_profile_shape(0, 0))
    # max_batch_size = 1
    for binding in engine:
        binding_shape = engine.get_binding_shape(binding)
        print('binding:', binding, '- binding_shape:', binding_shape)
        #Fix -1 dimension for proper memory allocation for batch_size > 1
        if binding == 'input':
            max_width = engine.get_profile_shape(0, 0)[2][3]
            max_height = engine.get_profile_shape(0, 0)[2][2]
            size = max_batch_size * max_width * max_height * 3
        else:
            binding_shape = (max_batch_size,) + binding_shape[1:]
            size = trt.volume(binding_shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
            input_shapes.append(engine.get_binding_shape(binding))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            #Collect original output shapes and names from engine
            out_shapes.append(binding_shape[1:])
            out_names.append(binding)
    return inputs, outputs, bindings, stream, input_shapes, out_shapes, out_names, max_batch_size

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

class TrtModelNMS(object):
    def __init__(self, model, max_size):
        self.engine_file = model
        self.engine = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None
        self.context = None
        self.cfx = cuda.Device(0).make_context()
        self.input_shapes = None
        self.out_shapes = None
        self.max_batch_size = 1
        self.max_size = max_size

        if self.engine is None:
            self.build()

    def build(self):
        with open(self.engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        # Allocate
        self.inputs, self.outputs, self.bindings, self.stream, self.input_shapes, self.out_shapes, self.out_names, self.max_batch_size = \
                allocate_buffers_nms(self.engine)
        # print(self.inputs, self.outputs, self.bindings, self.stream, self.input_shapes, self.out_shapes, self.out_names, self.max_batch_size)
        self.context = self.engine.create_execution_context()
        self.context.active_optimization_profile = 0
        # non lazy load implementation

    def run(self, input, deflatten: bool = True, as_dict = False):
        threading.Thread.__init__(self)
        self.cfx.push()

        input = np.asarray(input)
        batch_size, _, im_height, im_width = input.shape
        assert batch_size <= self.max_batch_size
        assert max(im_width, im_height) <= self.max_size, "Invalid shape: {}x{}, max shape: {}".format(im_width, im_height, self.max_size)
        allocate_place = np.prod(input.shape)

        self.inputs[0].host[:allocate_place] = input.flatten(order='C').astype(np.float32)
        self.context.set_binding_shape(0, input.shape)

        trt_outputs = do_inference(
            self.context, bindings=self.bindings,
            inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        # print(self.inputs, self.outputs, self.bindings, self.stream, self.input_shapes, self.out_shapes, self.out_names, self.max_batch_size)
        # Reshape TRT outputs to original shape instead of flattened array

        if deflatten:
            out_shapes = [(batch_size, ) + self.out_shapes[ix] for ix in range(len(self.out_shapes))]
            trt_outputs = [output[:np.prod(shape)].reshape(shape) for output, shape in zip(trt_outputs, out_shapes)]
        if as_dict:
            return {self.out_names[ix]: trt_output[:batch_size] for ix, trt_output in enumerate(trt_outputs)}
        
        self.cfx.pop()
        return [trt_output[:batch_size] for trt_output in trt_outputs]

def postprocess(preds, img, orig_imgs, retina_masks, conf, iou, agnostic_nms=False):
    """TODO: filter by classes."""
    
    p = ops.non_max_suppression(preds[0],
                                conf,
                                iou,
                                agnostic_nms,
                                max_det=100,
                                nc=1)

    results = []
    proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
    for i, pred in enumerate(p):
        orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
        img_path = "ok"
        if not len(pred):  # save empty boxes
            results.append(Results(orig_img=orig_img, path=img_path, names="segment", boxes=pred[:, :6]))
            continue
        if retina_masks:
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
        else:
            masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        results.append(
            Results(orig_img=orig_img, path=img_path, names="1213", boxes=pred[:, :6], masks=masks))
    return results

def pre_processing(img_origin, imgsz=1024):
    h, w = img_origin.shape[:2]
    if h>w:
        scale   = min(imgsz / h, imgsz / w)
        inp     = np.zeros((imgsz, imgsz, 3), dtype = np.uint8)
        nw      = int(w * scale)
        nh      = int(h * scale)
        a = int((nh-nw)/2) 
        inp[: nh, a:a+nw, :] = cv2.resize(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB), (nw, nh))
    else:
        scale   = min(imgsz / h, imgsz / w)
        inp     = np.zeros((imgsz, imgsz, 3), dtype = np.uint8)
        nw      = int(w * scale)
        nh      = int(h * scale)
        a = int((nw-nh)/2) 

        inp[a: a+nh, :nw, :] = cv2.resize(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB), (nw, nh))
    rgb = np.array([inp], dtype = np.float32) / 255.0
    return np.transpose(rgb, (0, 3, 1, 2))

class FastSam(object):
    def __init__(self, 
            model_weights , 
            max_size = 480):
        self.imgsz = (max_size, max_size)
        # Load model
        self.model = TrtModelNMS(model_weights, max_size)


    def segment(self, bgr_img, conf, iou, retina_mask, agnostic_nms):
        ## Padded resize
        inp = pre_processing(bgr_img, self.imgsz[0])
        
        ## Inference
        preds = self.model.run(inp)
        
        data_0 = torch.from_numpy(preds[5])
        data_1 = [[torch.from_numpy(preds[2]), torch.from_numpy(preds[3]), torch.from_numpy(preds[4])], torch.from_numpy(preds[1]), torch.from_numpy(preds[0])]
        preds = [data_0, data_1]

        results = postprocess(preds, inp, bgr_img, retina_mask, conf, iou, agnostic_nms)
        
        return results
    
    def draw_masks(self, img, results, img_size):
        return draw_masks(img, results, img_size)