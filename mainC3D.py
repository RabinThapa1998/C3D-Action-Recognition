import torch
from collections import deque
from time import time
import os
import shutil
import time
from pathlib import Path

import cv2
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages

from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
import win32gui, win32ui, win32con
import torch
from torch.autograd import Variable

from os.path import join
from glob import glob
import numpy as np
import skimage.io as io
from skimage.transform import resize

from C3D_model import C3D

from PIL import Image

from capturemod import WindowCapture
from hidePrints import HiddenPrints

import threading
from multiprocessing import  Process
import sqlite3
import calendar

stop_thread = False




def detect(save_img, weights, source, output, img_size, conf_thres, iou_thres, device, view_img, save_txt, classes,
           agnostic_nms, augment, update, model,half=True):

    out, source, weights, view_img, save_txt, imgsz = output, source, weights, view_img, save_txt, img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    label = None #valuemadebymyself


    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)

    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    for path, img, im0s, vid_cap in dataset:
        # print(f'img={np.shape(img)},im0s={np.shape(im0s)},path={path}')
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = time_synchronized()

        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Add bbox to image
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            print('%sDone. (%.3fs)' % (s, t2 - t1))
            try:
                cv2.imshow('p', im0)
            except:
                print("opencv can't imshow")


            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                raise StopIteration


    if label != None:
        return (label)

        # print('Done. (%.3fs)' % (time.time() - t0))









# # Initialize
# set_logging()
# device = select_device('')
# if os.path.exists('inference/output'):
#     shutil.rmtree('inference/output')  # delete output folder
# os.makedirs('inference/output')  # make new output folder
# half = device.type != 'cpu'  # half precision only supported on CUDA
# print(half)
#
# # Load model
# model = attempt_load('best_new_single_gun.pt',map_location=torch.device("cuda"))  # load FP32 model
# # imgsz = check_img_size(416, s=model.stride.max())  # check img_size
# if half:
#     model.half()  # to FP16

set_logging()
device = select_device('')
model = attempt_load('best_new_single_gun.pt',
                         map_location=torch.device("cuda"))
half = device.type != 'cpu'
model.half()



net = C3D()
net.load_state_dict(torch.load('C3D_Accu_90.pt'))
net.cuda()
net.eval()



def read_labels_from_file(filepath):
    with open(filepath, 'r', encoding="utf8") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def fight_recog(img,j):

    clip = np.array([resize(frame, output_shape=(112, 200), preserve_range=True) for frame in img])
    clip = clip[:, :, 44:44 + 112, :]  # crop centrally
    clip = clip.transpose(3, 0, 1, 2)  # ch, fr, h, w
    clip = np.expand_dims(clip, axis=0)  # batch axis
    clip = np.float32(clip)
    clip = torch.from_numpy(clip)
    # fight_pred = model_1(torch.from_numpy(clip), net)

    clip = clip.cuda()
    prediction = net(clip)
    prediction = prediction.data.cpu().numpy()

    # read labels
    labels = read_labels_from_file('labels1.txt')

    # print top predictions
    top_inds = prediction[0].argsort()[::-1][:5]  # reverse sort and take five largest items
    #     print('\nTop 5:')
    #     for i in top_inds:
    #         print('{:.5f} {}'.format(prediction[0][i], labels[i]))
    print('{:.1f} {}'.format(prediction[0][top_inds[0]], labels[top_inds[0]]))
    res_from_model = [prediction[0][top_inds[0]], labels[top_inds[0]]]

    res_from_model_format = f"{res_from_model[0]}/{res_from_model[1]}"

    lbx2.insert(j, res_from_model_format)
    root.update_idletasks()

    return res_from_model_format




import datetime
import time

def start1():

    inferinput = scvalue.get()
    print(inferinput)
    wincap = WindowCapture(inferinput)
    i = 0
    data_entry_list = []
    gun_pred = []
    with torch.no_grad():
        while (True):
                screenshotss = wincap.get_screenshot()
                if np.array_equal(screenshotss,np.zeros((2,2))) == False:
                    i += 1
                    cv2.imwrite('screenshot_1.jpg', screenshotss)

                    with HiddenPrints():
                        gun_pred_tmp = detect(False, 'Weaponwithalbumentationbest.pt', 'screenshot_1.jpg',
                                                   'inference/output', 416,
                                                  0.5, 0.5, torch.device('cuda'), True, False, None, False, False, False, model)

                    gun_pred.append(gun_pred_tmp)
                    ###DATE AND TIME WORK ####
                    t = time.localtime()
                    current_time = time.strftime("%H:%M:%S", t)
                    d = datetime.date.today()
                    date_form = f'{d.year}/{d.month}/{d.day}'
                    day_value = calendar.day_name[d.weekday()]
                    itm1 = (date_form,day_value, current_time, gun_pred_tmp)
                    data_entry_list.append(itm1)
                    lbx.insert(i, gun_pred_tmp)
                    root.update_idletasks()

                if stop_thread == True:
                    cv2.destroyAllWindows()
                    print(cv2.getWindowProperty('p', 1))

                    conn = sqlite3.connect('action.db')
                    c = conn.cursor()
                    c.execute("""CREATE TABLE IF NOT EXISTS guntable  
                                (date text,day text, time text, gun_action text)""")
                    c.executemany("INSERT INTO guntable VALUES (?,?,?,?)",data_entry_list)
                    conn.commit()
                    conn.close()

                    for i in range(5):  # maybe 5 or more
                        cv2.waitKey(1)
                    break




def start2():
    inferinput = scvalue.get()
    wincap_2 = WindowCapture(inferinput)
    i = 0
    img = []
    data_entry_list = []
    with torch.no_grad():
        while (True):
            screenshot = wincap_2.get_screenshot()
            if np.array_equal(screenshot,np.zeros((2,2))) == False:
                i += 1
                img.append(screenshot)
                if len(img) == 16:
                    res_from_model_format = fight_recog(img, i)
                    img = []

                    t = time.localtime()
                    current_time = time.strftime("%H:%M:%S", t)
                    d = datetime.date.today()
                    day_value = calendar.day_name[d.weekday()]
                    date_form = f'{d.year}/{d.month}/{d.day}'
                    itm1 = (date_form,day_value, current_time, res_from_model_format)
                    data_entry_list.append(itm1)


            if stop_thread == True:
                conn = sqlite3.connect('action.db')
                c = conn.cursor()
                c.execute("""CREATE TABLE IF NOT EXISTS fighttable  
                            (date text, day text, time text, fight_action text)""")
                conn.commit()
                c.executemany("INSERT INTO fighttable VALUES (?,?,?,?)", data_entry_list)
                conn.commit()
                conn.close()

                torch.cuda.empty_cache()
                break





def stop_it():
    print("stop pressed")
    global stop_thread
    stop_thread = True

def starter():
    lbx.delete('0', 'end')
    lbx2.delete('0', 'end')
    global stop_thread
    stop_thread = False
    th_1 = threading.Thread(target=start1)
    th_2 = threading.Thread(target=start2)
    th_1.start()
    th_2.start()



from tkinter import *
from tkinter import ttk
from ttkthemes import themed_tk
import sqlite3

conn = sqlite3.connect('action.db')
c1 = conn.cursor()
c2 = conn.cursor()
c1.execute("SELECT * FROM guntable")
c2.execute("SELECT * FROM fighttable")
rows1 = c1.fetchall()
rows2 = c2.fetchall()
conn.close()


def comboclick_2(event):
    if tab2_combo.get() == 'Recent':
        conn = sqlite3.connect('action.db')
        c1 = conn.cursor()
        c2 = conn.cursor()
        c1.execute("SELECT * FROM guntable ORDER BY date DESC, time DESC")
        c2.execute("SELECT * FROM fighttable ORDER BY date DESC, time DESC")
        rows1 = c1.fetchall()
        rows2 = c2.fetchall()
        conn.close()
        tv.delete(*tv.get_children())
        tv2.delete(*tv2.get_children())
        for i in rows1:
            tv.insert('', 'end', values=i)
        for i in rows2:
            tv2.insert('', 'end', values=i)

    elif tab2_combo.get() == 'Ascending':
        conn = sqlite3.connect('action.db')
        c1 = conn.cursor()
        c2 = conn.cursor()
        c1.execute("SELECT * FROM guntable ORDER BY date ASC,time ASC")
        c2.execute("SELECT * FROM fighttable ORDER BY date ASC,time ASC")
        rows1 = c1.fetchall()
        rows2 = c2.fetchall()
        conn.close()
        tv.delete(*tv.get_children())
        tv2.delete(*tv2.get_children())
        for i in rows1:
            tv.insert('', 'end', values=i)
        for i in rows2:
            tv2.insert('', 'end', values=i)


options = ["Recent","Ascending"]
#THEMES_SELECTED:
#black,radiance,elegance,clearlooks

root = themed_tk.ThemedTk(theme="radiance")

root.title("S.H.A.D")
root.geometry('500x500')

my_notebook = ttk.Notebook(root)
my_notebook.pack(fill=BOTH,expand=YES)

my_frame1 = Frame(my_notebook,width=500, height = 500)
my_frame2 = Frame(my_notebook,width=500, height = 500)

my_frame1.pack(fill=BOTH,expand=YES)
my_frame2.pack(fill=BOTH,expand=YES)

my_notebook.add(my_frame1,text="tab_0")
my_notebook.add(my_frame2,text="tab_1")

frame_inside_fr_1 = Frame(my_frame1)
frame_inside_fr_1.pack(anchor = 'nw', pady = 60,padx=20)


tab2_combo = ttk.Combobox(my_frame2, value=options)
tab2_combo.pack(anchor='nw',padx=30,pady=20)
tab2_combo.current(1)
tab2_combo.bind("<<ComboboxSelected>>",comboclick_2)




##frame_inside_fr_1
scvalue = StringVar()
screen = Entry(frame_inside_fr_1, textvariable=scvalue, font=('verdana',14))
# screen.grid(row =0, column = 1)
screen.pack(padx=40,pady=2,side=LEFT)
# b = Button(frame_inside_fr_1, text='start', font=('verdana',12),pady=0)
b = Button(frame_inside_fr_1, text ="Start", command = starter)
b.pack(padx=1,pady=2,side=LEFT)

b2 = Button(frame_inside_fr_1, text ="Stop", command = stop_it)
b2.pack(padx=25,pady=2,side=RIGHT)


frame_inside_fr_2 = Frame(my_frame1)
frame_inside_fr_2.pack(side=LEFT,pady = 5,padx=20 )

fr1= LabelFrame(frame_inside_fr_2,text="frame1",font=('Fixedsys',10))
# fr1.grid(row=0,column=4)
fr1.pack(side=LEFT ,padx=20,pady=10)
sbr = Scrollbar(fr1,)
sbr.pack(side=RIGHT, fill=Y)
lbx = Listbox(fr1, font=('verdana',10),width=30,height=20)
lbx.pack(side=LEFT, fill= Y, expand=True)
sbr.config(command=lbx.yview)
lbx.config(yscrollcommand=sbr.set)

fr2= LabelFrame(frame_inside_fr_2,text="frame2",font=('Fixedsys',10))
# fr1.grid(row=0,column=4)
fr2.pack(side=LEFT ,padx=20,pady=10)
sbr2 = Scrollbar(fr2,)
sbr2.pack(side=RIGHT, fill=Y)
lbx2 = Listbox(fr2, font=('verdana',10),width=30,height=20)
lbx2.pack(side=LEFT, fill= Y, expand=True)
sbr2.config(command=lbx2.yview)
lbx2.config(yscrollcommand=sbr2.set)



par_frm = Frame(my_frame2,padx=20)
par_frm.pack(anchor="sw",padx=5,expand=YES)


frm1 = Frame(par_frm)
frm1.pack(side=LEFT)
tv = ttk.Treeview(frm1,columns=(1,2,3,4) , show="headings", height="25")
tv.pack(pady=10 ,side= LEFT, expand=YES,fill=X)
tv.column("#0", minwidth=0, width=100, stretch=NO)
tv.column(1, minwidth=100, width=100, stretch=NO)
tv.column(2, minwidth=100, width=100, stretch=NO)
tv.column(3, minwidth=100, width=100, stretch=NO)
tv.column(4, minwidth=100, width=100, stretch=NO)
tv.heading(1, text="Date")
tv.heading(2, text="Day")
tv.heading(3, text="Time")
tv.heading(4, text="Action")

sbr = Scrollbar(frm1,)
sbr.pack(side=LEFT, fill=Y,pady=10)
sbr.config(command=tv.yview)
tv.config(yscrollcommand=sbr.set)

###NEXT###FRAME###
frm2 = Frame(par_frm)
frm2.pack(side=RIGHT)
# frm2.grid(row=2, column=1)
tv2 = ttk.Treeview(frm2,columns=(1,2,3,4) , show="headings", height="25",selectmode='browse')
tv2.pack(pady=10,side= LEFT, expand=YES,fill=X)
tv2.column("#0", minwidth=0, width=100, stretch=NO)
tv2.column(1, minwidth=100, width=100, stretch=NO)
tv2.column(2, minwidth=100, width=80, stretch=NO)
tv2.column(3, minwidth=100, width=200, stretch=NO)
tv2.column(4, minwidth=100, width=200, stretch=NO)

tv2.heading(1, text="Date")
tv2.heading(2, text="Day")
tv2.heading(3, text="Time")
tv2.heading(4, text="Action")

sbr = Scrollbar(frm2,)
sbr.pack(side=RIGHT, fill=Y,pady=10)
sbr.config(command=tv2.yview)
tv2.config(yscrollcommand=sbr.set)




for i in rows1:
    tv.insert('','end',values =i)

for i in rows2:
    tv2.insert('','end',values =i)





root.mainloop()





















#
# from tkinter  import *
#
# root = Tk()
# root.geometry("655x333")
# root.title("pytorch")
#
#
#
# # f = Frame(root, bg='grey')
# scvalue = StringVar()
# screen = Entry(root, textvariable=scvalue, font=('verdana',8))
# # screen.pack(side=LEFT,ipadx=5, pady=5)
# screen.grid(row=0,column=1)
# b = Button(root, text='start', font='lucida 10 bold')
# b.grid(row=0,column=2)
# b.bind('<Button-1>', starter)
#
# # f.pack()
# b2 = Button(root, text='stop', font='lucida 10 bold')
# b2.grid(row=0,column=3)
# b2.bind('<Button-1>', stop_it)
#
#
# fr = Frame(root)
# # fr.pack(side=RIGHT)
# fr.grid(row=2,column=1)
# sbr = Scrollbar(fr,)
# sbr.pack(side=RIGHT, fill=Y)
#
# lbx = Listbox(fr, font=('verdana',16))
# lbx.pack(side=LEFT, fill="both", expand=True)
#
# sbr.config(command=lbx.yview)
# lbx.config(yscrollcommand=sbr.set)
#
# fr2 = Frame(root)
# # fr2.pack(side=LEFT)
# fr2.grid(row=2,column=2)
# lbx2=Listbox(fr2, font=('verdana',8))
# lbx2.pack(side=LEFT, fill="both", expand=True)
# sbr2 = Scrollbar(fr2,)
# sbr2.pack(side=RIGHT, fill=Y)
# sbr2.config(command=lbx2.yview)
# lbx2.config(yscrollcommand=sbr2.set)
#
#
# root.mainloop()









