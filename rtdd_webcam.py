from imutils.video import VideoStream
from imutils import paths
from skimage.io import imread,imsave
from io import BytesIO
from PIL import Image
from multiprocessing import Manager
import argparse
import scipy.ndimage as spi
import numpy as np
import PIL
import os
import sys
import tensorflow as tf
import itertools
import argparse
import imutils
import time
import random
import cv2
import multiprocessing 

#Global Values
last_layer = None
last_grad = None
last_channel = None

model_path = 'tensorflow_inception_graph.pb'
model_fn = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_path)

def newlayer(ns):
    #selects random layer and channel
    x = random.randint(0,4)
    if x==0:
        ns.layer = "mixed4c_pool_reduce"
        ns.channel = random.randint(0,63)
    elif x==1:
        ns.layer = "mixed4c"
        ns.channel = random.randint(0,100)
    elif x==2:
        ns.layer = "mixed4c"
        ns.channel = random.randint(100,200)
    elif x==3:
        ns.layer = "mixed4c"
        ns.channel = random.randint(400,500)
    elif x==4:
        ns.layer = "mixed4b"
        ns.channel = random.randint(0,100)
    else:
        ns.layer = "mixed3b"
        ns.channel = random.randint(0,100)

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res
    
def dream(graph, sess, t_input, imagenet_mean, t_preprocessed,input_img,layer_name,channel_value,iter_value, step_size, octave_value):
    tile_size = 256
    
    octave_value = 1
    octave_scale_value = 1
    model_path = 'tensorflow_inception_graph.pb'
    print_model = False
    verbose = False
    #input_img = spi.imread(input_img, mode="RGB")
    
    
    #model_fn = os.path.join(os.path.dirname(os.path.realpath(__file__)), model_path)
    # creating TensorFlow session and loading the model
    #graph = tf.Graph()
    #sess = tf.InteractiveSession(graph=graph)
    #with tf.gfile.FastGFile(model_fn, 'rb') as f:
    #    graph_def = tf.GraphDef()
    #    graph_def.ParseFromString(f.read())
    #t_input = tf.placeholder(np.float32, name='input') # define the input tensor
    #imagenet_mean = 117.0
    #t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
    #tf.import_graph_def(graph_def, {'input':t_preprocessed})

    # Optionally print the inputs and layers of the specified graph.
    #if not print_model:
      #print(graph.get_operations())

    def T(layer):
        '''Helper for getting layer output tensor'''
        return graph.get_tensor_by_name("import/%s:0"%layer)

    def tffunc(*argtypes):
        '''Helper that transforms TF-graph generating function into a regular one.
        See "resize" function below.
        '''
        placeholders = list(map(tf.placeholder, argtypes))
        def wrap(f):
            out = f(*placeholders)
            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
            return wrapper
        return wrap

    # Helper function that uses TF to resize an image
    def resize(img, size):
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0,:,:,:]
    resize = tffunc(np.float32, np.int32)(resize)

    def calc_grad_tiled(img, t_grad, tile_size=512):
        '''Compute the value of tensor t_grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile boundaries over
        multiple iterations.'''
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h-sz//2, sz),sz):
            for x in range(0, max(w-sz//2, sz),sz):
                sub = img_shift[y:y+sz,x:x+sz]
                g = sess.run(t_grad, {t_input:sub})
                grad[y:y+sz,x:x+sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)

    def render_deepdream(t_grad, img0, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
        # split the image into a number of octaves
        img = img0
        octaves = []
        for i in range(octave_n-1):
            hw = img.shape[:2]
            lo = resize(img, np.int32(np.float32(hw)/octave_scale))
            hi = img-resize(lo, hw)
            img = lo
            octaves.append(hi)

        # generate details octave by octave
        for octave in range(octave_n):
            if octave>0:
                hi = octaves[-octave]
                img = resize(img, hi.shape[:2])+hi
            for i in range(iter_n):
                #g = calc_grad_tiled(img, t_grad)
                g = calc_grad_tiled(img, t_grad, tile_size)
                img += g*(step / (np.abs(g).mean()+1e-7))
            #if not verbose:
                #print ("Iteration Number: %d" % i)
        #if not verbose:
                #print ("Octave Number: %d" % octave)


        return Image.fromarray(np.uint8(np.clip(img/255.0, 0, 1)*255)) 
    
    def render(img, layer='mixed4d_3x3_bottleneck_pre_relu', channel=139, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
        global last_layer, last_grad, last_channel
        if last_layer == layer and last_channel == channel:
            t_grad = last_grad
        else:
            if channel == 4242:
                t_obj = tf.square(T(layer))
            else:
                t_obj = T(layer)[:,:,:,channel]
            t_score = tf.reduce_mean(t_obj) # defining the optimization objective
            t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
            last_layer = layer
            last_grad = t_grad
            last_channel = channel
        img0 = np.float32(img)
        return render_deepdream(t_grad, img0, iter_n, step, octave_n, octave_scale)
        
        
    output_img = render(input_img, layer=layer_name, channel=channel_value, iter_n=iter_value, step=step_size, octave_n=octave_value, octave_scale=octave_scale_value)
    return output_img
    
    
    
#--------------------------------
#-----  PROCESS  FUNCTIONS ------
#--------------------------------
    

def deepdream_process(ns, globalframelock):

    # creating TensorFlow session and loading the model
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    with tf.gfile.FastGFile(model_fn, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf.placeholder(np.float32, name='input') # define the input tensor
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
    tf.import_graph_def(graph_def, {'input':t_preprocessed})

    globalframelock.acquire()
    dreamframe = ns.globalnewframe.copy()
    globalframelock.release()
    
    while(1):
        #dreamframe = ns.globalnewframe.copy()
        dreamframe = dream(graph, sess, t_input, imagenet_mean, t_preprocessed, dreamframe, ns.layer, ns.channel, ns.iter_value, ns.step_size, ns.octave_value)
        dreamframe = np.array(dreamframe)
        globalframelock.acquire()
            
        
        warpeddream = warp_flow(dreamframe.copy(), ns.totalflow)
        dreamframe = warp_flow(ns.globaldreamframe.copy(), ns.totalflow)
            
        #blendeddream = ((ns.globaldreamframe.copy()*(ns.frameblendfactor/255)) + (warpeddream*((1-ns.frameblendfactor)/255)))
        #blendeddream = np.average(ns.globaldreamframe.copy()*(ns.frameblendfactor/255), ns.globaldreamframe.copy()*(ns.frameblendfactor/255))
        blendeddream = cv2.addWeighted(dreamframe,.6,warpeddream.copy(),.4,0)
        blendeddream = cv2.addWeighted(blendeddream,.9,ns.globalnewframe.copy(),.1,0)
        
        ns.globaldreamframe = blendeddream
        ns.totalflow = np.zeros_like(ns.totalflow)
        
        #selects random layer and channel
        x = random.randint(0,40)
        if x==0:
            newlayer(ns)
            print("new layer: " + ns.layer +str(ns.channel))
        
        globalframelock.release()
    
        
def opticalflow_process(ns, vs, globalframelock):

    while(1):
    
        globalframelock.acquire()
    
        # grab the frame from the threaded video stream
        newframe = vs.read()
        
        # resize the frame to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        ns.globalnewframe = imutils.resize(newframe, width=360)
        ns.next = cv2.cvtColor(ns.globalnewframe,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(ns.prev, ns.next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        ns.totalflow = ns.totalflow + flow
        
        mag, ang = cv2.cartToPolar(ns.totalflow[...,0], ns.totalflow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        
        output = warp_flow(ns.globaldreamframe.copy(), ns.totalflow)
        
        #matframe = ns.globalnewframe*mat
        
        #find mat 
        #outputmat = cv2.cvtColor(output.copy(),cv2.COLOR_BGR2GRAY)
        #mat = cv2.threshold(outputmat,127,255,cv2.THRESH_BINARY_INV)
        #output2 = mat*ns.globalnewframe.copy()
        
        #ns.globaldreamframe = np.array(output)
        output = cv2.resize(output, dsize=(1000, 720), interpolation=cv2.INTER_CUBIC)
        
        # show frames
        cv2.imshow("Input", ns.globalnewframe)
        cv2.imshow("Flow", rgb)
        cv2.imshow("Output", output)
        
        ns.prev = ns.next.copy()
        
        globalframelock.release()
        
        key = cv2.waitKey(1) & 0xFF

        # if the `n` key is pressed change layer
        if key == ord("n"):
            newlayer(ns)
            print("new layer: " + ns.layer +str(ns.channel))
        if key == ord("i"):
            ns.iter_value += 1
            print("iter_value: " + str(ns.iter_value))
        if key == ord("k"):
            if(ns.iter_value > 1):
                ns.iter_value -= 1
            print("iter_value: " + str(ns.iter_value))
        if key == ord("u"):
            if(ns.frameblendfactor < 1):
                ns.frameblendfactor += .05
            print("frameblendfactor: " + str(ns.frameblendfactor))
        if key == ord("j"):
            if(ns.frameblendfactor > 0):
                ns.frameblendfactor -= .05
            print("frameblendfactor: " + str(ns.frameblendfactor))
        if key == ord("o"):
            ns.step_size += 1
            print("step_size: " + str(ns.step_size))
        if key == ord("l"):
            if(ns.iter_value > 0):
                ns.step_size -= 1
            print("step_size: " + str(ns.step_size))
        # otheriwse, if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            break
        
        
                
        
    
    
if __name__ == '__main__':

    #globals
    manager = Manager()
    ns = manager.Namespace()
    ns.frameblendfactor = .01
    ns.iter_value = 1
    ns.step_size = 2
    ns.octave_value = 1
    ns.layer = ""
    ns.channel = 0
    
    
    newlayer(ns)
    
    print("starting video stream")
    time.sleep(2)
    vs = VideoStream(src=0).start()

    globalframelock = multiprocessing.Lock()

    #Start video stream
    frame1 = vs.read()
    ns.frameblendfactor = .7

    frame = imutils.resize(frame1, width=360)
    ns.globaldreamframe = frame.copy()
    ns.globalnewframe = frame.copy()
    ns.prev = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ns.next = ns.prev.copy()


    hsv = np.zeros_like(frame)
    hsv[...,1] = 255
    ns.totalflow = 0
    
    dd = multiprocessing.Process(target=deepdream_process, args=(ns, globalframelock))
    
    #Start multiprocessing
    print("Start multiprocessing")
    dd.start()
    opticalflow_process(ns, vs, globalframelock)
    
    
    #Cleanup
    cv2.destroyAllWindows()
    vs.stop()
    

