import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from glob import glob
from matplotlib.widgets import Button, Slider
from natsort import natsorted
from scipy.ndimage import measurements
import pickle
from aic_models import data_preprocess as dp

class Viewer2D(object):
    def __init__(self,
                data_path: str,
                folder_mask: str,
                frame=0, 
                threshold_min = 130,
                threshold_max = 450,
                mask_agatston=None, 
                agatston=False, 
                area=None):

        self.frame_init = frame
        self.data_path = data_path
        self.folder_mask = folder_mask
        self.mask_agatston = mask_agatston
        self.agatston = agatston
        self.area = area
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        
        # Callback
        self.draw()

    def draw(self):
        
        if self.agatston :
            fig, ax = plt.subplots(1, 2, figsize=(15, 10))
            # set window title
            if fig.canvas.manager is not None:
                fig.canvas.manager.set_window_title("Agatston Score")
            ax0, ax1 = ax[0], ax[1]
            ax0.get_xaxis().set_ticks([])
            ax0.get_yaxis().set_ticks([])
            ax1.get_xaxis().set_ticks([])
            ax1.get_yaxis().set_ticks([])
            axslice = plt.axes([0.20, 0.9, 0.65, 0.03])
            callback = Image_2D(self.data_path, 
                                self.folder_mask,
                                self.frame_init, 
                                ax0, 
                                ax1, 
                                axslice, 
                                fig, 
                                threshold_min=self.threshold_min,
                                threshold_max=self.threshold_max,
                                mask_agatston=self.mask_agatston, 
                                agatston=self.agatston, 
                                area=self.area)
            
            plt.show()
        else :
            fig, ax = plt.subplots(1, 2, figsize=(15, 10))
            ax0, ax1 = ax[0], ax[1]
            ax0.axis('off')
            ax1.axis('off')
            axslice = plt.axes([0.20, 0.9, 0.65, 0.03])
            callback = Image_2D(self.data_path, 
                                self.folder_mask,
                                self.frame_init, 
                                ax0, 
                                ax1, 
                                axslice, 
                                fig)
            axprev = plt.axes([0.125, 0.05, 0.15, 0.075])
            axnext = plt.axes([0.7, 0.05, 0.15, 0.075])
            axsave = plt.axes([0.5, 0.05, 0.15, 0.075])
            bnext = Button(axnext, 'Next Patient')
            bnext.on_clicked(callback.next)
            bprev = Button(axprev, 'Previous Patient')
            bprev.on_clicked(callback.prev)
            bsave = Button(axsave, 'Save Image')
            bsave.on_clicked(callback.save_image)
            plt.show()
class Image_2D(object):
    def __init__(self, 
                data_path, 
                label_folder, 
                frame, 
                axis1, 
                axis2, 
                axislicer, 
                fig,
                threshold_min = 130,
                threshold_max = 450,
                mask_agatston=None, 
                agatston=False, 
                area=None):

        self.data_path = data_path
        self.label_folder = label_folder
        self.frame = frame
        self.axislicer = axislicer
        self.fig_canvas = fig
        self.prediction = None
        self.mask_agatston = mask_agatston
        self.agatston_bool = agatston
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.area = area

        # Dicom image
        self.slices = None
        self.slices = dp.load_scan(self.data_path[self.frame])
        self.image = dp.get_pixels_hu(self.slices)
        
        if self.label_folder != "":
            folder = os.path.join(self.label_folder,self.data_path[self.frame].split('/')[-2],data_path[frame].split('/')[-1])
            self.label = dp.load_mask(folder)
            self.fig_canvas.suptitle(self.data_path[self.frame].split('/')[-2], fontsize=12)
            if len(self.label) == 0:
                self.label = None
            self.init_label = True
        else :
            self.label = None
        
        # Slider Gui
        self.index = len(self.image)//2
        self.slicer = Slider(self.axislicer, 'Image', 0, len(self.image)-1, valinit=len(self.image)//2, valstep=1)
        self.axis1 = axis1

        if axis2 is not None:
            self.axis2 = axis2
            self.brush_activate = None
            self.purple_yellow_activate = None
            if not self.agatston_bool:
                # Brush activate GUI
                self.axbrush = plt.axes([0.3, 0.05, 0.15, 0.075])
                self.bbrush = Button(self.axbrush, 'Brush OFF')
                self.bbrush.on_clicked(self.brush_state)

                # State Purple-Yellow GUI
                self.axlabel_state = plt.axes([0.92, 0.7, 0.05, 0.075])
                self.bpurple_yellow = Button(self.axlabel_state, 'Purple')
                self.bpurple_yellow.on_clicked(self.brush_pixel)
                self.purple_yellow_activate = False
                self.brush_activate = False
        else :
            self.brush_activate = None
            self.purple_yellow_activate = None

        # Need to activate Press and release
        self.pressed = False
        self.fig_canvas.canvas.mpl_connect('button_press_event', self.mouse_pressed)
        self.fig_canvas.canvas.mpl_connect('button_release_event', self.mouse_released)
        self.fig_canvas.canvas.mpl_connect('motion_notify_event', self.mouse_position)

        self._image = None
        self._image2 = None
        self._image = self.axis1.imshow(self.image[self.index], cmap='gray')
        self._image2 = self.axis2.imshow(self.image[self.index], cmap='gray')
                
        if self.label is not None:
            self.label_image = np.load(self.label[self.index])
            self._label = self.axis2.imshow(self.label_image)
            self.init_label = False
        
        if self.agatston_bool : 
            self.score = 0.0
            self._prediction_view = None
            self.agatston_score_slice()
            self._prediction_view = self.axis2.imshow(self.prediction, cmap='jet')
            self.score = self.agatston_score()
            self.axis2.set_xlabel("Agatston score : {:.3f}".format(self.score))
            self.save_prediction()

        self.slicer.on_changed(self.update)

    def save_prediction(self):
        # Save prediction in dictionnary
        save_predict = {}
        save_predict["data_path"] = "/".join([self.data_path[self.frame].split("/")[-2],self.data_path[self.frame].split("/")[-1]])
        save_predict["score"] = self.score
        save_predict["image"] = self.image
        save_predict["mask_agatston"] = self.mask_agatston
        folder =  self.data_path[self.frame].replace("datasets_dcm","predictions")
        os.makedirs(folder,exist_ok=True)
        with open(folder + "/prediction.pkl",'wb') as f:
            pickle.dump(save_predict,f)
    
    def agatston_score_slice(self):
        self.prediction = self.image[self.index].copy()
        self.prediction[self.mask_agatston[self.index] == 0] = 0
        if self.threshold_min is not None :
            self.prediction[self.prediction < self.threshold_min] = 0
        if self.threshold_max is not None :
            self.prediction[self.prediction > self.threshold_max] = 0
        self.prediction[self.prediction > 0] = 1
        area_,lw = self.area_measurements(self.prediction)
        for j,number_of_pix in enumerate(area_):
            if j != 0:
                if number_of_pix*self.area <= 1: #(density higher than 1mm2)
                    self.prediction[lw==j] = 0
        self.prediction = np.ma.masked_where(self.prediction == 0, self.prediction)

    def area_measurements(self,slice):
        slice[slice != 0] = 1
        lw, num = measurements.label(slice)
        area_ = measurements.sum(slice, lw, index=np.arange(lw.max() + 1))
        return area_,lw
    
    def agatston_score(self):
        score = 0.0
        for i in range(len(self.image)):
            prediction = self.image[i].copy()
            prediction[self.mask_agatston[i] == 0] = 0
            if self.threshold_min is not None :
                prediction[prediction < self.threshold_min] = 0
            if self.threshold_max is not None :
                prediction[prediction > self.threshold_max] = 0
            area_,lw = self.area_measurements(prediction)
            for j,number_of_pix in enumerate(area_):
                    if j != 0 :
                        if number_of_pix*self.area <= 1: #(density higher than 1mm2)
                            prediction[lw==j] = 0

            prediction[np.logical_and(prediction >= 130, prediction < 200)] = 1
            prediction[np.logical_and(prediction >= 200, prediction < 300)] = 2
            prediction[np.logical_and(prediction >= 300, prediction < 400)] = 3
            prediction[prediction>400] = 4

            score += self.area*np.sum(prediction)         
        
        return score 

    def next(self, event):
        if self.frame == len(self.data_path)-1:
            self.frame = len(self.data_path)-1
        else:
            self.frame += 1
        self.slices = self.load_scan(self.data_path, self.frame)
        self.image = self.get_pixels_hu(self.slices)
        
        if self.label is not None:
            self.label = self.get_mask(self.data_path, self.label_folder, self.frame, self.fig_canvas)

        self.slicer.valmax = len(self.image)-1
        self.slicer.valinit = len(self.image)//2
        self.slicer.reset()

    def prev(self, event):
        if self.frame == 0:
            self.frame = 0
        else:
            self.frame -= 1

        self.slices = self.load_scan(self.data_path, self.frame)
        self.image = self.get_pixels_hu(self.slices)

        if self.label is not None:
            self.label = self.get_mask(self.data_path, self.label_folder, self.frame, self.fig_canvas)
        
        self.slicer.valmax = len(self.image)-1
        self.slicer.valinit = len(self.image)//2
        self.slicer.reset()

    def update(self, val):
        self.index = int(self.slicer.val)
        if self._image is None:
            self._image = self.axis1.imshow(self.image[self.index], cmap='gray')
            self._image2 = self.axis2.imshow(self.image[self.index], cmap='gray')
        else:
            self._image.set_data(self.image[self.index])
            self._image2.set_data(self.image[self.index])

        if self.agatston_bool:
            self.agatston_score_slice()
            if self._prediction_view is None :
                self._prediction_view = self.axis2.imshow(self.prediction,cmap='jet')
            else : 
                self._prediction_view.set_data(self.prediction)   

        if self.label is not None and not self.init_label:
            self.label_image = np.load(self.label[self.index])
            self._label.set_data(self.label_image)

        elif self.label is not None and self.init_label:
            self.label_image = np.load(self.label[self.index])
            self.axis2.axis('off')
            self._label = self.axis2.imshow(self.label_image, vmin=np.min(self.label_image), vmax=np.max(self.label_image))

        else:
            self.init_label = True

    def save_image(self, event):
        if self.label is not None :
            np.save(self.label[self.index],self.label_image)
        print('Saved')

    def brush_state(self, event):
        if self.brush_activate is not None :
            if not self.brush_activate:
                self.bbrush.label.set_text('Brush ON')
                self.brush_activate = not self.brush_activate
            else:
                self.bbrush.label.set_text('Brush OFF')
                self.brush_activate = not self.brush_activate

    def brush_pixel(self, event):
        if self.purple_yellow_activate is not None :
            if not self.purple_yellow_activate:
                self.bpurple_yellow.label.set_text('Yellow')
                self.purple_yellow_activate = not self.purple_yellow_activate
            else:
                self.bpurple_yellow.label.set_text('Purple')
                self.purple_yellow_activate = not self.purple_yellow_activate

    def mouse_pressed(self, event):
        self.pressed = True

    def mouse_released(self, event):
        self.pressed = False

    def mouse_position(self, event):
        if self.brush_activate is not None :
            if self.pressed and self.brush_activate:
                if event.xdata is not None and event.ydata is not None :
                    x_coord = int(np.floor(event.xdata))
                    y_coord = int(np.floor(event.ydata))
                    if not self.purple_yellow_activate:
                        self.label_image[y_coord, x_coord] = 1
                        self._label.set_data(self.label_image)
                        self.fig_canvas.canvas.draw()
                    else:
                        self.label_image[y_coord, x_coord] = 0
                        self._label.set_data(self.label_image)
                        self.fig_canvas.canvas.draw()
