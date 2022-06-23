#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Library for the 2D interface
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import aic.misc.utils as ut
import aic.processing.operations as op
import aic.processing.scoring as sc


class Viewer2D(object):
    """Class supporting the 2D Viewer interface."""

    def __init__(self,
                 data_path: str,
                 folder_mask: str,
                 frame=0,
                 threshold_min=None,
                 threshold_max=None,
                 mask_agatston=None,
                 agatston=False,
                 area=None,
                 valve=None,
                 candidate=None):
        """Init function."""
        self.frame_init = frame
        self.data_path = data_path
        self.folder_mask = folder_mask
        self.mask_agatston = mask_agatston
        self.agatston = agatston
        self.area = area
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max

        # Extra parameters passed
        self.valve = valve
        self.candidate = candidate

        # Callback
        self.draw()

    def draw(self):
        """Draw interface."""
        if self.agatston:
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
                                area=self.area,
                                valve=self.valve,
                                candidate=self.candidate)

            plt.show()
        else:
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
    """Class supporting the Image interface."""

    def __init__(self,
                 data_path,
                 label_folder,
                 frame,
                 axis1,
                 axis2,
                 axislicer,
                 fig,
                 threshold_min=None,
                 threshold_max=None,
                 mask_agatston=None,
                 agatston=False,
                 area=None,
                 valve=None,
                 candidate=None):
        """Init function."""
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
        self.valve = valve
        self.candidate = candidate

        # Dicom image
        self.slices = None
        self.slices = ut.load_scan(self.data_path[self.frame])
        self.image = op.get_pixels_hu(self.slices)

        if self.label_folder != "":
            folder = \
                os.path.join(
                    self.label_folder,
                    self.data_path[self.frame].split('/')[-2],
                    self.data_path[self.frame].split('/')[-1])

            if os.path.exists(folder):
                self.label = ut.load_mask(folder)
            else:
                self.label = []
            self.fig_canvas.suptitle(self.data_path[self.frame].split('/')[-2],
                                     fontsize=12)
            if len(self.label) == 0:
                self.label = None
            self.init_label = True
        else:
            self.label = None

        # Slider Gui
        self.index = len(self.image)//2
        self.slicer = \
            Slider(self.axislicer,
                   'Image',
                   0,
                   len(self.image)-1,
                   valinit=len(self.image)//2,
                   valstep=1)
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
        else:
            self.brush_activate = None
            self.purple_yellow_activate = None

        # Need to activate Press and release
        self.pressed = False
        self.fig_canvas.canvas.mpl_connect('button_press_event',
                                           self.mouse_pressed)
        self.fig_canvas.canvas.mpl_connect('button_release_event',
                                           self.mouse_released)
        self.fig_canvas.canvas.mpl_connect('motion_notify_event',
                                           self.mouse_position)

        self._image = None
        self._image2 = None
        self._image = \
            self.axis1.imshow(self.image[self.index],
                              cmap='gray')
        self._image2 = \
            self.axis2.imshow(self.image[self.index],
                              cmap='gray')

        if self.label is not None:
            self.label_image = np.load(self.label[self.index])
            self._label = self.axis2.imshow(self.label_image)
            self.init_label = False

        if self.agatston_bool:
            self.score = 0.0
            self._prediction_view = None
            self.prediction = \
                sc.agatston_score_slice(
                    self.image,
                    self.mask_agatston,
                    self.index,
                    self.area,
                    self.threshold_min,
                    self.threshold_max)
            self._prediction_view = \
                self.axis2.imshow(self.prediction,
                                  cmap='jet')
            self.score = \
                sc.agatston_score(
                    self.image,
                    self.mask_agatston,
                    self.area,
                    self.threshold_min,
                    self.threshold_max)
            self.axis2.set_xlabel("Agatston score : {:.3f}".format(self.score))
            _ = sc.save_prediction(self.image,
                                   self.mask_agatston,
                                   self.data_path[self.frame],
                                   self.score,
                                   self.area,
                                   self.threshold_min,
                                   self.threshold_max,
                                   self.valve,
                                   self.candidate)

        self.slicer.on_changed(self.update)

    def next(self, event):
        """Update slice by next event."""
        if self.frame == len(self.data_path)-1:
            self.frame = len(self.data_path)-1
        else:
            self.frame += 1
        self.slices = ut.load_scan(self.data_path[self.frame])
        self.image = op.get_pixels_hu(self.slices)

        folder = \
            os.path.join(
                self.label_folder,
                self.data_path[self.frame].split('/')[-2],
                self.data_path[self.frame].split('/')[-1])

        if os.path.exists(folder):
            self.label = ut.load_mask(folder)
        else:
            self.label = []
        if len(self.label) == 0:
            self.label = None

        self.fig_canvas.suptitle(
            self.data_path[self.frame].split('/')[-2],
            fontsize=12)

        self.slicer.ax.set_xlim(0, len(self.image)-1)
        self.slicer.valmax = len(self.image)-1
        self.slicer.valinit = len(self.image)//2
        self.slicer.reset()

    def prev(self, event):
        """Update slice by previous event."""
        if self.frame == 0:
            self.frame = 0
        else:
            self.frame -= 1

        self.slices = ut.load_scan(self.data_path[self.frame])
        self.image = op.get_pixels_hu(self.slices)

        folder = \
            os.path.join(
                self.label_folder,
                self.data_path[self.frame].split('/')[-2],
                self.data_path[self.frame].split('/')[-1])

        if os.path.exists(folder):
            self.label = ut.load_mask(folder)
        else:
            self.label = []
        if len(self.label) == 0:
            self.label = None

        self.fig_canvas.suptitle(
            self.data_path[self.frame].split('/')[-2],
            fontsize=12)

        self.slicer.ax.set_xlim(0, len(self.image)-1)
        self.slicer.valmax = len(self.image)-1
        self.slicer.valinit = len(self.image)//2
        self.slicer.reset()

    def update(self, val):
        """Update slice render."""
        self.index = int(self.slicer.val)
        if self._image is None:
            self._image = \
                self.axis1.imshow(self.image[self.index],
                                  cmap='gray')
            self._image2 = \
                self.axis2.imshow(self.image[self.index],
                                  cmap='gray')
        else:
            self._image.set_data(self.image[self.index])
            self._image2.set_data(self.image[self.index])

        if self.agatston_bool:
            self.prediction = \
                sc.agatston_score_slice(
                    self.image,
                    self.mask_agatston,
                    self.index,
                    self.area,
                    self.threshold_min,
                    self.threshold_max)
            if self._prediction_view is None:
                self._prediction_view = \
                    self.axis2.imshow(self.prediction,
                                      cmap='jet')
            else:
                self._prediction_view.set_data(self.prediction)

        if self.label is not None and not self.init_label:
            self.label_image = np.load(self.label[self.index])
            self._label.set_data(self.label_image)

        elif self.label is not None and self.init_label:
            self.label_image = np.load(self.label[self.index])
            self.axis2.axis('off')
            self._label = \
                self.axis2.imshow(self.label_image,
                                  vmin=np.min(self.label_image),
                                  vmax=np.max(self.label_image))
        else:
            self.init_label = True

    def save_image(self, event):
        """Save image."""
        if self.label is not None:
            np.save(self.label[self.index],
                    self.label_image)
        print('Saved')

    def brush_state(self, event):
        """Update brush state."""
        if self.brush_activate is not None:
            if not self.brush_activate:
                self.bbrush.label.set_text('Brush ON')
                self.brush_activate = not self.brush_activate
            else:
                self.bbrush.label.set_text('Brush OFF')
                self.brush_activate = not self.brush_activate

    def brush_pixel(self, event):
        """Update pixel attribute."""
        if self.purple_yellow_activate is not None:
            if not self.purple_yellow_activate:
                self.bpurple_yellow.label.set_text('Yellow')
                self.purple_yellow_activate = \
                    not self.purple_yellow_activate
            else:
                self.bpurple_yellow.label.set_text('Purple')
                self.purple_yellow_activate = \
                    not self.purple_yellow_activate

    def mouse_pressed(self, event):
        """Mouse pressed status update."""
        self.pressed = True

    def mouse_released(self, event):
        """Mouse released status update."""
        self.pressed = False

    def mouse_position(self, event):
        """Mouse position status update."""
        if self.brush_activate is not None:
            if self.pressed and self.brush_activate:
                if event.xdata is not None and event.ydata is not None:
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
