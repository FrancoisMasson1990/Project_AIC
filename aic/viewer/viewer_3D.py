#!/usr/bin/env python3.9
# *-* coding: utf-8*-*
"""
Copyright (C) 2022 Project AIC - All Rights Reserved.

Unauthorized copy of this file, via any medium is strictly
prohibited. Proprietary and confidential.

Note
----
Library for the 3D interface
"""

import vtk
from vedo import *
import numpy as np
import os
from vedo import settings
from vtk.util.numpy_support import vtk_to_numpy
import glob
from aic.viewer.widget import *
import aic.processing.data_preprocess as dp
from tqdm import tqdm
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull as cvxh
from tqdm import tqdm
from cylinder_fitting import fit
import aic.viewer.viewer_2D as v2d
vtk.vtkObject.GlobalWarningDisplayOff()


class Viewer3D(object):
    """Class supporting the 3D Viewer interface."""

    def __init__(self,
                 data_path: str,
                 frame=0,
                 mode=1,
                 label='/label_mask/',
                 npy=None,
                 multi_label=False,
                 model=None,
                 template: bool = False,
                 model_version=1,
                 template_directory="./templates/",
                 **kwargs):
        """Init function."""
        self.frame = frame
        self.init = True
        self.icol = 0
        self.data_path = data_path
        self.colors = vtk.vtkNamedColors()
        self.window_size = (1200, 800)
        self.model = model
        if mode == 1:
            self.mode = ['ray_cast']
        elif mode == 2:
            self.mode = ['ray_cast',
                         'iso']
        elif mode == 4:
            self.mode = ['ray_cast',
                         'iso',
                         'slicer_2d',
                         'inference']

        self.actor_list = []
        self.buttons = []
        self.render_list = []
        self.axes_list = []
        self.grid_list = []
        self.actor_gt_list = []
        self.actor_infer_list = []
        self.actor_fitting_list = []
        self.cutter_tool = False
        self.mover_tool = False
        self.valve_value = 20
        self.window_2D = False
        self.window_3Dslice = False
        self.cutter_obj = []
        self.mover_obj = []
        self.widget_cut = None
        self.mask = None
        self.render_score = None
        self.spacing = None
        self.title = None
        self.predictions_final = None
        self.predictions_fitting = None
        self.area = None
        self.label_folder = label
        self.npy_folder = npy
        self.multi_label = multi_label
        self.template = template
        self.max = None
        self.model_version = model_version
        self.template_directory = template_directory

        # Extra parameters passed
        self.crop_dim = kwargs.get("crop_dim", -1)
        self.z_slice_min = kwargs.get("z_slice_min", None)
        self.z_slice_max = kwargs.get("z_slice_max", None)
        self.threshold = kwargs.get("threshold", None)
        self.ratio_spacing = kwargs.get("spacing", None)

        '''One render window, multiple viewports'''
        self.rw = vtk.vtkRenderWindow()
        self.rw.SetSize(int(self.window_size[0]), int(self.window_size[1]))
        self.iren = vtk.vtkRenderWindowInteractor()

        # Freeze windows if not clicked on actor !!!!
        vsty = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(vsty)
        self.view_mode = True
        self.gt_view_mode = False
        self.infer_view_mode = False
        self.fitting_view_mode = False
        self.slicer_2d_view_mode = False

        # Callback to update content
        self.iren.RemoveObservers('KeyPressEvent')
        self.iren.AddObserver('LeftButtonPressEvent', self._mouseleft)
        self.iren.AddObserver('KeyPressEvent', self.update)

        self.iren.SetRenderWindow(self.rw)
        self.viewport_frame()
        self.viewport()

    def viewport_frame(self):
        """Set viewport properties."""
        self.xmins = []
        self.xmaxs = []
        self.ymins = []
        self.ymaxs = []

        # Define viewport ranges
        for size in range(len(self.mode)):
            self.xmins.append(size/len(self.mode))
            self.xmaxs.append(size/len(self.mode) + 1/len(self.mode))
            self.ymins.append(0)
            self.ymaxs.append(1)

        if len(self.mode) == 4:
            self.xmins = [0, 0.5, 0, 0.5]
            self.xmaxs = [0.5, 1, 0.5, 1]
            self.ymins = [0.5, 0.5, 0, 0]
            self.ymaxs = [1, 1, 0.5, 0.5]

        self.xmins = np.asarray(self.xmins)
        self.xmaxs = np.asarray(self.xmaxs)
        self.ymins = np.asarray(self.ymins)
        self.ymaxs = np.asarray(self.ymaxs)

        # Have some fun with colors
        # self.ren_bkg = ['AliceBlue', 'GhostWhite', 'WhiteSmoke', 'Seashell']
        # self.ren_bkg = ['GhostWhite']
        self.ren_bkg = ['Black']

    def viewport(self):
        """Generate viewport."""
        for i in range(len(self.mode)):
            self.ren = vtk.vtkRenderer()
            self.render_list.append(self.ren)
            self.rw.AddRenderer(self.ren)
            self.ren.SetViewport(self.xmins[i],
                                 self.ymins[i],
                                 self.xmaxs[i],
                                 self.ymaxs[i])
            self.actor = self.add_actors(self.mode[i], self.frame)
            if self.actor:
                self.ren.AddActor(self.actor)
            if self.mode[i] == 'ray_cast':
                # button used for the colors changing
                states, c, bc, pos, size, font, bold, italic, alpha, angle = \
                    self.button_cast(pos=[0.7, 0.035],
                                     states=["State 1", "State 2"])
                self.but = \
                    Button(self.buttonfuncMode,
                           states,
                           c,
                           bc,
                           pos,
                           size,
                           font,
                           bold,
                           italic,
                           alpha,
                           angle).status(int(0))
                self.ren.AddActor2D(self.but.actor)
                self.buttons.append(self.but)

                if len(self.axes_list) == 0:
                    self.axes = Axes(self.iren)
                    self.axes_list.append(self.axes.ax)
                if len(self.grid_list) == 0:
                    self.grid = Grid(self.actor)
                    self.grid_list.append(self.grid.grid)
                    self.ren.AddActor(self.grid.grid)

            elif self.mode[i] == 'iso':
                # button used for the slicer 2d
                states, c, bc, pos, size, font, bold, italic, alpha, angle = \
                    self.button_cast(pos=[0.7, 0.035],
                                     states=["Cutter On", "Cutter Off"])
                self.cutter = \
                    Button(self.buttonfuncMode_cutter,
                           states,
                           c,
                           bc,
                           pos,
                           size,
                           font,
                           bold,
                           italic,
                           alpha,
                           angle).status(int(0))
                self.ren.AddActor2D(self.cutter.actor)
                self.buttons.append(self.cutter)

                # button used for saving
                states, c, bc, pos, size, font, bold, italic, alpha, angle = \
                    self.button_cast(pos=[0.5, 0.035], states=["Save"])
                self.saver = \
                    Button(self.buttonfuncMode_saving,
                           states,
                           c,
                           bc,
                           pos,
                           size,
                           font,
                           bold,
                           italic,
                           alpha,
                           angle).status(int(0))
                self.ren.AddActor2D(self.saver.actor)
                self.buttons.append(self.saver)

            elif self.mode[i] == 'slicer_2d':
                # button used for the 2D Slicer and change view
                # button used for the colors changing
                states, c, bc, pos, size, font, bold, italic, alpha, angle = \
                    self.button_cast(pos=[0.7, 0.035],
                                     states=["Start Slicer Mode",
                                             "Stop Slicer Mode"])
                self.but_ = \
                    Button(self.buttonviewMode,
                           states,
                           c,
                           bc,
                           pos,
                           size,
                           font,
                           bold,
                           italic,
                           alpha,
                           angle).status(int(0))
                self.ren.AddActor2D(self.but_.actor)
                self.buttons.append(self.but_)

            elif self.mode[i] == 'inference':
                self.render_score = self.ren
                # button used for inference and GT
                states, c, bc, pos, size, font, bold, italic, alpha, angle = \
                    self.button_cast(pos=[0.5, 0.035],
                                     states=["Inference (On)",
                                             "Inference (Off)"])
                self.infer = \
                    Button(self.buttonfuncInference,
                           states,
                           c,
                           bc,
                           pos,
                           size,
                           font,
                           bold,
                           italic,
                           alpha,
                           angle).status(int(0))
                self.ren.AddActor2D(self.infer.actor)
                self.buttons.append(self.infer)

                states, c, bc, pos, size, font, bold, italic, alpha, angle = \
                    self.button_cast(pos=[0.15, 0.035],
                                     states=["Ground Truth (On)",
                                             "Ground Truth (Off)"])
                self.ground_truth = \
                    Button(self.buttonfuncGroundTruth,
                           states,
                           c,
                           bc,
                           pos,
                           size,
                           font,
                           bold,
                           italic,
                           alpha,
                           angle).status(int(0))
                self.ren.AddActor2D(self.ground_truth.actor)
                self.buttons.append(self.ground_truth)

                states, c, bc, pos, size, font, bold, italic, alpha, angle = \
                    self.button_cast(pos=[0.8, 0.035],
                                     states=["Fitting (On)",
                                             "Fitting (Off)"])
                self.fitting = \
                    Button(self.buttonfuncFitting,
                           states,
                           c,
                           bc,
                           pos,
                           size,
                           font,
                           bold,
                           italic,
                           alpha,
                           angle).status(int(0))
                self.ren.AddActor2D(self.fitting.actor)
                self.buttons.append(self.fitting)

                states, c, bc, pos, size, font, bold, italic, alpha, angle = \
                    self.button_cast(pos=[0.12, 0.94],
                                     states=["Agatston Score"])
                self.score_ratio = \
                    Button(self.buttonfuncAgatston,
                           states,
                           c,
                           bc,
                           pos,
                           size,
                           font,
                           bold,
                           italic,
                           alpha,
                           angle).status(int(0))
                self.ren.AddActor2D(self.score_ratio.actor)
                self.buttons.append(self.score_ratio)

                # button used for the cylinder properties
                states, c, bc, pos, size, font, bold, italic, alpha, angle = \
                    self.button_cast(pos=[0.8, 0.135],
                                     states=["Move On",
                                             "Move Off"])
                self.mover = \
                    Button(self.buttonfuncMode_mover,
                           states,
                           c,
                           bc,
                           pos,
                           size,
                           font,
                           bold,
                           italic,
                           alpha,
                           angle).status(int(0))
                self.ren.AddActor2D(self.mover.actor)
                self.buttons.append(self.mover)

            self.ren.SetBackground(self.colors.GetColor3d(self.ren_bkg[0]))
            self.ren.ResetCamera()
            self.camera_position()

    def add_actors(self, mode, frame):
        """Add actors."""
        txt = glob.glob(self.data_path[frame] + '/*.txt')
        self.title = self.data_path[frame].split('/')
        self.title = self.title[-2] + '/' + self.title[-1]
        if len(txt) > 0:
            txt = txt[0]
            f = open(txt, "r")
            lines = f.readlines()
            valve = lines[2]
            # Remove wrong asci character
            valve = valve[:-1]
            # Add progress bar
            valve += " file : {} on {}".format(frame+1, len(self.data_path))
            self.rw.SetWindowName(valve)
        else:
            self.rw.SetWindowName('Valve Unknown')

        # Add actors according to the mode and create tmp file
        if mode == 'ray_cast':
            actor = self.ray_cast(self.data_path[frame])
            self.actor_list.append(actor)
        if mode == 'iso':
            actor = self.iso_surface(self.data_path[frame])
            self.cutter_actor = actor
            self.actor_list.append(actor)
        if mode == 'slicer_2d':
            actor = self.slicer_2d(self.data_path[frame])
            self.actor_list.append(actor)

        self.init = False

        if mode == 'inference':
            return None
        else:
            return actor

    def buttonfuncMode(self):
        """Switch button mode."""
        s = self.volume.mode()
        snew = (s + 1) % 2
        self.volume.mode(snew)
        self.but.switch()

    def buttonviewMode(self):
        """Set view mode."""
        self.but_.switch()
        if self.view_mode:
            style = vtk.vtkInteractorStyleImage()
            style.SetInteractionModeToImage3D()
            self.view_mode = not self.view_mode
            printc("Slicer Mode:", invert=1, c="m")
            printc(
                """Press\n
                SHIFT+Left mouse to rotate the camera for oblique slicing\n
                SHIFT+Middle mouse to slice perpendicularly through the image\n
                Left mouse and Drag to modify luminosity and contrast\n
                X                   to Reset to sagittal view\n
                Y                   to Reset to coronal view\n
                Z                   to Reset to axial view\n
                R                   to Reset the Window/Levels\n
                Q                   to Quit.""",
                c="m",
            )
        else:
            style = vtk.vtkInteractorStyleTrackballCamera()
            self.view_mode = not self.view_mode
        self.iren.SetInteractorStyle(style)

    def buttonfuncMode_cutter(self):
        """Switch Cutter button mode."""
        if not self.cutter_tool:
            self.cutter_obj.append(
                Cutter(self.render_list[-1],
                       self.iren,
                       self.cutter_actor))
            for cut in self.cutter_obj:
                self.widget_cut = cut.boxWidget
                self.widget_cut.On()
            self.cutter_tool = not self.cutter_tool
        else:
            for cut in self.cutter_obj:
                self.widget_cut = cut.boxWidget
                self.widget_cut.Off()
                cut.actor.mapper().RemoveAllClippingPlanes()
            self.cutter_tool = not self.cutter_tool
            self.cutter_obj = []
        self.cutter.switch()

    def buttonfuncMode_mover(self):
        """Switch Mover button mode."""
        if self.fitting_view_mode:
            if not self.mover_tool:
                self.mover_obj.append(
                    Mover(self.render_list[-1],
                          self.iren,
                          self.cylinder))
                for mov in self.mover_obj:
                    self.widget_mov = mov.boxWidget
                    self.widget_mov.On()
                self.mover_tool = not self.mover_tool
            else:
                for mov in self.mover_obj:
                    self.widget_mov = mov.boxWidget
                    self.widget_mov.Off()
                self.mover_tool = not self.mover_tool
                self.mover_obj = []
            self.mover.switch()

    def buttonfuncMode_saving(self):
        """Switch Save button mode."""
        if self.cutter_tool:
            for cut in self.cutter_obj:
                # Save the remaining points
                planes = vtk.vtkPlaneCollection()
                for faces in range(cut.planes.GetNumberOfPlanes()):
                    plane = vtk.vtkPlane()
                    plane.SetOrigin(cut.origins[faces])
                    plane.SetNormal(cut.normals[faces])
                    planes.AddItem(plane)

                clipper = vtk.vtkClipClosedSurface()
                clipper.SetInputData(cut.actor.GetMapper().GetInput())
                clipper.SetClippingPlanes(planes)
                clipper.Update()
                self.numpy_saving(clipper)

    def buttonfuncGroundTruth(self):
        """Switch Ground Truth button mode."""
        self.ground_truth.switch()
        if not self.gt_view_mode:
            numpy_3d = \
                glob.glob(os.path.join(self.npy_folder,
                                       self.title) + '/*.npy')
            if len(numpy_3d) > 0:
                actor_ = self.label_3d(numpy_3d[0],
                                       c=[1, 0, 0])
                self.actor_gt_list.append(actor_)
                for render in self.render_list:
                    if render == self.render_score:
                        render.AddActor(actor_)

            self.rw.Render()
            self.ren.ResetCamera()
            self.camera_position()
            self.gt_view_mode = not self.gt_view_mode

        else:
            for render in self.render_list:
                if render == self.render_score:
                    for actor_ in self.actor_gt_list:
                        render.RemoveActor(actor_)

            self.actor_gt_list = []
            self.rw.Render()
            self.gt_view_mode = not self.gt_view_mode

    def buttonfuncInference(self):
        """Switch Inference button mode."""
        self.infer.switch()
        if self.model is not None:
            if not self.infer_view_mode:
                # Prediction
                idx = os.path.join(self.data_path[self.frame])
                img = dp.load_scan(idx)
                img = dp.get_pixels_hu(img)
                crop_values = None
                # old version, input images were normalized for each slice
                if self.model_version == 0:
                    img = dp.preprocess_inputs(img)
                # new version, input images were normalized according to z
                elif self.model_version == 1:
                    # padding
                    # Need for the mesh reconstruct
                    padding = np.zeros(img.shape) - 2
                    if self.crop_dim != -1:
                        img = dp.crop_dim(img,
                                          crop_dim=self.crop_dim)
                    if (self.z_slice_min is not None) \
                            and (self.z_slice_max is not None):
                        min_ = int(self.z_slice_min*img.shape[0])
                        max_ = int(self.z_slice_max*img.shape[0])
                        index_z_crop = np.arange(min_, max_)
                        img = img[index_z_crop]
                    img = dp.preprocess_img(img)
                    img = np.expand_dims(img, -1)
                else:
                    print("Unknown/Unsupported version")
                    exit()

                pred_list = []
                # https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
                for i in tqdm(range(img.shape[0])):
                    pred = np.expand_dims(img[i, :, :, :], 0)
                    prediction = self.model.predict(pred)
                    if self.model_version == 0:
                        prediction = np.argmax(prediction.squeeze(), axis=-1)
                        prediction = np.rot90(prediction, axes=(1, 0))
                        prediction = np.expand_dims(prediction, 0)
                        prediction[prediction == 0] = -1
                    elif self.model_version == 1:
                        prediction = prediction[0, :, :, 0]
                        prediction = np.rot90(prediction, axes=(1, 0))
                        prediction = np.expand_dims(prediction, 0)
                        prediction[prediction != 1.0] = -2
                    else:
                        print("Unknown/Unsupported version")
                        exit()
                    pred_list.append(prediction)
                predictions = np.vstack(pred_list)
                # Padding reconstruction
                if self.model_version == 1:
                    if self.crop_dim != -1:
                        xc = (self.dimensions[0] - self.crop_dim) // 2
                        yc = (self.dimensions[1] - self.crop_dim) // 2
                    else:
                        xc = 0
                        yc = 0
                    if (self.z_slice_min is not None) \
                            and (self.z_slice_max is not None):
                        padding[index_z_crop,
                                xc:xc+img.shape[1],
                                yc:yc+img.shape[2]] = predictions
                    else:
                        padding[:,
                                xc:xc+img.shape[1],
                                yc:yc+img.shape[2]] = predictions
                    predictions = padding
                    crop_values = [xc*self.spacing[0],
                                   xc+img.shape[1]*self.spacing[0],
                                   yc*self.spacing[1],
                                   yc+img.shape[2]*self.spacing[1]]

                predictions, _ = dp.resample(predictions,
                                             [self.spacing[0],
                                              self.spacing[1]],
                                             self.spacing[2],
                                             [1, 1, 1])
                vertices, _ = dp.make_mesh(predictions, -1)

                # Clustering
                self.vertices_predictions = \
                    dp.clustering(vertices,
                                  self.model_version,
                                  self.center,
                                  self.all_numpy_nodes,
                                  ratio=0.4,
                                  threshold=4000,
                                  max_=self.max,
                                  dimensions=self.dimensions,
                                  spacings=self.spacing,
                                  crop_values=crop_values)

                # Volume Cropping
                # First prediction : UX visual
                self.predictions_final = self.img.clone()
                self.predictions_final = dp.boxe_3d(self.predictions_final,
                                                    self.vertices_predictions,
                                                    max_=self.max)
                # Second prediction : Volume
                self.predictions_agatston = self.volume.clone()
                self.predictions_agatston = \
                    dp.boxe_3d(self.predictions_agatston,
                               self.vertices_predictions,
                               max_=self.max)
                # Get the all points in isosurface Mesh/Volume
                self.predictions_agatston_points = \
                    dp.to_points(self.predictions_agatston,
                                 template=self.template)
                self.predictions_final_points = \
                    dp.to_points(self.predictions_final)
                self.predictions_final_points_threshold = \
                    self.predictions_agatston_points[
                        self.predictions_agatston_points[:, 3] >
                        self.threshold]

                # Convex-Hull estimation
                hull = \
                    cvxh(
                        self.predictions_final_points_threshold[:, :3])
                mask = \
                    dp.isInHull(self.predictions_agatston_points[:, :3],
                                hull)
                self.predictions_agatston_points = \
                    self.predictions_agatston_points[mask]

                actor_ = self.label_3d(self.predictions_final_points,
                                       c=[0, 1, 0])

                self.actor_infer_list.append(actor_)
                for render in self.render_list:
                    if render == self.render_score:
                        render.AddActor(actor_)
                self.rw.Render()
                self.ren.ResetCamera()
                self.camera_position()
                self.infer_view_mode = not self.infer_view_mode

            else:
                for render in self.render_list:
                    if render == self.render_score:
                        for actor_ in self.actor_infer_list:
                            render.RemoveActor(actor_)
                self.actor_infer_list = []
                self.rw.Render()
                self.infer_view_mode = not self.infer_view_mode

    def buttonfuncFitting(self):
        """Generate cylinder fitting.

        Parameters
        ----------
            w_fit = Direction of the cylinder axis
            C_fit = A point on the cylinder axis
            r_fit = Radius of the cylinder
            fit_err = Fitting error (G function)
        """
        self.fitting.switch()
        if (self.fitting.status() == "Fitting (Off)") \
                and (self.predictions_final_points_threshold is not None):
            # Cylinder Fit
            print("performing fitting...")
            self.w_fit, self.C_fit, self.r_fit, self.fit_err = \
                fit(self.predictions_final_points_threshold[:, :3],
                    guess_angles=None)
            print("fitting done !")
            self.cylinder = \
                Cylinder(pos=tuple(self.C_fit),
                         r=self.r_fit,
                         height=20,
                         axis=tuple(self.w_fit),
                         alpha=0.5,
                         c="white")
            self.actor_fitting_list.append(self.cylinder)
            for render in self.render_list:
                if render == self.render_score:
                    render.AddActor(self.cylinder)

            self.rw.Render()
            self.ren.ResetCamera()
            self.camera_position()
            self.fitting_view_mode = True
        else:
            for render in self.render_list:
                if render == self.render_score:
                    for actor_ in self.actor_fitting_list:
                        render.RemoveActor(actor_)
            self.actor_fitting_list = []
            self.rw.Render()
            self.fitting_view_mode = False

    def buttonfuncAgatston(self):
        """Estimation of Agatston score by projecting prediction along z axis.

        For each layer, least_square cylinder fitting and rrremove point
        outside the fitted cylinder
        """
        if self.fitting_view_mode:
            # Update center and axis if object was moved
            self.C_fit = np.asarray(self.cylinder.GetCenter())
            self.w_fit = np.asarray(self.cylinder.normalAt(48))

            # Projection along z axis and centered in (0,0,0)
            points = dp.z_projection(self.predictions_agatston_points,
                                     self.w_fit)
            x_mean = \
                (np.min(points[:, 0]) + np.max(points[:, 0]))/2
            y_mean = \
                (np.min(points[:, 1]) + np.max(points[:, 1]))/2
            z_mean = \
                (np.min(points[:, 2]) + np.max(points[:, 2]))/2

            if points.shape[1] == 4:
                # 4 dimension due to intensity value
                points -= np.array([x_mean,
                                    y_mean,
                                    z_mean,
                                    0])
            else:
                points -= np.array([x_mean,
                                    y_mean,
                                    z_mean])
            # Rounding of layer due to floating error
            points[:, 2] = np.round(points[:, 2])

            predictions_agatston_points = \
                points[points[:, 3] < self.threshold]
            predictions_final_points_threshold = \
                points[points[:, 3] >= self.threshold]

            # Loop to remove outside points
            inner_points = []
            # Hack to avoid first and last layer where few points lead to wrong
            # circle estimation
            first_slices = \
                int(np.percentile(
                    np.arange(
                        len(np.unique(
                            predictions_agatston_points[:, 2])
                            )
                        ), 20))
            last_slices = \
                int(np.percentile(
                    np.arange(
                        len(np.unique(
                            predictions_agatston_points[:, 2])
                            )
                        ), 80))
            for i, z in enumerate(
                    np.unique(predictions_agatston_points[:, 2])):
                predictions_final_tmp = \
                    predictions_final_points_threshold.copy()
                predictions_final_tmp = \
                    predictions_final_tmp[predictions_final_tmp[:, 2] == z]
                predictions_agatston = \
                    predictions_agatston_points.copy()
                predictions_agatston = \
                    predictions_agatston[predictions_agatston[:, 2] == z]
                if predictions_final_tmp.shape[0] > 2:
                    xc, yc, _, _ = \
                        dp.leastsq_circle(predictions_final_tmp[:, 0],
                                          predictions_final_tmp[:, 1])
                    circle_center = np.array([xc, yc, z])
                    # Estimate the min value by slices
                    r_fit = []
                    for point in predictions_final_tmp[:, :3]:
                        # Exclude intensity points
                        r_fit.append(dp.euclidean(point, circle_center))
                    if len(r_fit) > 0:
                        r_fit = np.array(r_fit)
                        # Based on experimental analysis on template valve,
                        # residual space along stent
                        if self.ratio_spacing is not None:
                            r_fit = np.min(r_fit) - \
                                self.ratio_spacing*self.spacing[0]
                        else:
                            r_fit = np.min(r_fit)
                        # Estimate the distance of each point for the agatston
                        d = []
                        for point in predictions_agatston[:, :3]:
                            # Exclude intensity points
                            d.append(dp.euclidean(point, circle_center))
                        d = np.array(d)
                        if i < first_slices or i > last_slices:
                            if predictions_final_tmp.shape[0] > 30:
                                p = \
                                    predictions_agatston[np.where(d < r_fit)]
                            else:
                                p = np.empty((0, 4))
                        else:
                            p = \
                                predictions_agatston[np.where(d < r_fit)]
                    else:
                        p = np.empty((0, 4))
                else:
                    p = np.empty((0, 4))
                inner_points.append(p)
            inner_points = np.concatenate(inner_points)
            mask = \
                np.where(np.all(np.isin(points, inner_points),
                                axis=1))
            mask_agatston = \
                self.get_mask_2D(
                    self.predictions_agatston_points[mask])
            # Show the score in 2D mode
            v2d.Viewer2D(data_path=self.data_path,
                         folder_mask="",
                         frame=self.frame,
                         mask_agatston=mask_agatston,
                         agatston=True,
                         area=self.area,
                         threshold_max=None)
        else:
            v2d.Viewer2D(data_path=self.data_path,
                         folder_mask="",
                         frame=self.frame,
                         mask_agatston=self.mask.copy(),
                         agatston=False,
                         area=self.area)

    def button_cast(self,
                    pos: list = None,
                    states: list = None):
        """Cast button."""
        c = ["bb", "gray"]
        # colors of states
        bc = ["gray", "bb"]
        font = "arial"
        size = 16
        bold = 0
        italic = False
        alpha = 1
        angle = 0
        return states, c, bc, pos, size, font, bold, italic, alpha, angle

    def camera_position(self, zoom=0.7):
        """Set camera position."""
        self.camera = self.ren.GetActiveCamera()
        self.camera.Zoom(zoom)

    def ray_cast(self, data):
        """Set ray cast."""
        self.img = load(data).imagedata()
        if self.init:
            self.volume = Volume(self.img,
                                 c='jet',
                                 mode=int(0))
            self.volume.jittering(True)
        else:
            self.volume._update(self.img)

        if self.threshold is None:
            scrange = self.img.GetScalarRange()
            self.threshold = (2 * scrange[0] + scrange[1]) / 3.0

        if self.template:
            points = vtk_to_numpy(self.volume.topoints()
                                  .GetMapper().GetInput()
                                  .GetPoints().GetData())
            intensity = vtk_to_numpy(self.volume.imagedata()
                                     .GetPointData()
                                     .GetScalars())
            # Pixel value intensity
            index = np.argmax(intensity)
            self.max = points[index]

        return self.volume

    def iso_surface(self, data):
        """Generate iso surface."""
        # Check that mask and image have the same size
        self.slices = dp.load_scan(self.data_path[self.frame])
        self.shape = dp.get_pixels_hu(self.slices)

        self.img = load(data)
        # Following lines used to get the mask
        self.mask = tuple(reversed(load(data).imagedata().GetDimensions()))

        if self.mask != self.shape.shape:
            self.img.resize(self.shape.shape[1],
                            self.shape.shape[2],
                            self.shape.shape[0])
            self.mask = self.shape.shape

        self.mask = np.zeros(self.mask, dtype=int)
        self.spacing = self.img.imagedata().GetSpacing()
        self.area = self.spacing[0]*self.spacing[1]
        self.dimensions = self.img.imagedata().GetDimensions()

        # High value for the cylinder fitting
        self.img_z_threshold = self.img.isosurface(self.threshold)
        self.img = self.img.isosurface()

        # Get the all points in isosurface
        self.points = self.img.GetMapper().GetInput()
        self.all_array = self.points.GetPoints()
        self.all_numpy_nodes = vtk_to_numpy(self.all_array.GetData())
        # x y z center from isovolume
        self.center = \
            np.array(
                [(np.min(self.all_numpy_nodes[:, 0]) +
                    np.max(self.all_numpy_nodes[:, 0]))/2,
                 (np.min(self.all_numpy_nodes[:, 1]) +
                     np.max(self.all_numpy_nodes[:, 1]))/2,
                 (np.min(self.all_numpy_nodes[:, 2]) +
                     np.max(self.all_numpy_nodes[:, 2]))/2])

        return self.img

    def slicer_2d(self, data):
        """Generate 2D Slicer."""
        self.image = load(data).imagedata()
        self.im = vtk.vtkImageResliceMapper()
        self.im.SetInputData(self.image)
        self.im.SliceFacesCameraOn()
        self.im.SliceAtFocalPointOn()
        self.im.BorderOn()
        self.ip = vtk.vtkImageProperty()
        self.ip.SetInterpolationTypeToLinear()
        self.ia = vtk.vtkImageSlice()
        self.ia.SetMapper(self.im)
        self.ia.SetProperty(self.ip)
        self.init = False

        return self.ia

    def get_mask_2D(self, data):
        """Generate mask."""
        print("Generating mask...")
        mask_agatston = np.zeros([self.dimensions[2],
                                  self.dimensions[0],
                                  self.dimensions[1]],
                                 dtype=np.uint8)

        # all voxels have value zero except ones predicted:
        for d in tqdm(data):
            x = int(d[0]/self.spacing[0])
            y = int(d[1]/self.spacing[1])
            z = int(d[2]/self.spacing[2])
            mask_agatston[z, x, y] = 1

        for k in range(self.dimensions[2]):
            mask_agatston[k, ::] = np.rot90(mask_agatston[k, ::])

        return mask_agatston

    def label_3d(self, data, c=[1, 0, 0]):
        """Generate 3D Labels."""
        if isinstance(data, str):
            with open(data, 'rb') as f:
                img = np.load(f)
        if isinstance(data, np.ndarray):
            img = data
        self.points = vtk.vtkPoints()
        # Create the topology of the point (a vertex)
        self.vertices = vtk.vtkCellArray()
        # Add points
        for i in range(0, len(img)):
            p = img[i].tolist()
            if len(p) > 3:
                p = p[:3]
            point_id = self.points.InsertNextPoint(p)
            self.vertices.InsertNextCell(1)
            self.vertices.InsertCellPoint(point_id)

        # Create a poly data object
        polydata = vtk.vtkPolyData()
        # Set the points and vertices we created as the geometry
        # and topology of the polydata
        polydata.SetPoints(self.points)
        polydata.SetVerts(self.vertices)
        polydata.Modified()

        # Mapper for points
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        self.actor_point = vtk.vtkActor()
        self.actor_point.SetMapper(mapper)
        self.actor_point.GetProperty().SetColor(c)

        return self.actor_point

    def numpy_saving(self, points):
        """Save labels into numpy format."""
        # Will try to fit a rectangle and to retrieve
        # the x,y,z coordinate for segmentation...
        array = points.GetOutput().GetPoints()
        numpy_nodes = vtk_to_numpy(array.GetData())

        if os.path.exists(self.npy_folder):
            directory = self.npy_folder
        else:
            directory = os.getcwd() + self.npy_folder
        # Make folder recursively
        os.makedirs(os.path.join(
            directory,
            self.title),
                    exist_ok=True)

        with open(os.path.join(directory,
                               self.title) + '/' + 'volume_label.npy',
                  'wb') as f:
            np.save(f, numpy_nodes)

        numpy_nodes[:, 0] = np.around(numpy_nodes[:, 0]/self.spacing[0])
        numpy_nodes[:, 1] = np.around(numpy_nodes[:, 1]/self.spacing[1])
        numpy_nodes[:, 2] = np.around(numpy_nodes[:, 2]/self.spacing[2])
        # Along the z axis : Later Along x axis and y axis ...
        # (3D Object Detection from CT Scans using a Slice-and-fuse Approach)
        slicer = np.unique(numpy_nodes[:, 2]).astype(int)

        if self.multi_label:
            self.all_numpy_nodes[:, 0] = \
                np.around(self.all_numpy_nodes[:, 0]/self.spacing[0])
            self.all_numpy_nodes[:, 1] = \
                np.around(self.all_numpy_nodes[:, 1]/self.spacing[1])
            self.all_numpy_nodes[:, 2] = \
                np.around(self.all_numpy_nodes[:, 2]/self.spacing[2])
            slicer_tot = np.unique(self.all_numpy_nodes[:, 2]).astype(int)

        if os.path.exists(self.label_folder):
            directory = self.label_folder
        else:
            directory = os.getcwd() + self.label_folder
        os.makedirs(os.path.join(directory,
                                 self.title),
                    exist_ok=True)

        mask_1 = self.mask.copy()
        mask_2 = self.mask.copy()
        for z_axis in range(self.mask.shape[0]):
            if self.multi_label:
                if z_axis in slicer_tot:
                    # save and diplay the mask (Not magna valve label)
                    nodes_copy = self.all_numpy_nodes
                    z = nodes_copy[self.all_numpy_nodes[:, 2] == z_axis]
                    x_y = np.ndarray(shape=(z.shape[0], 2), dtype=int)
                    x_y[:] = z[:, 0:2]
                    x_y = np.unique(x_y, axis=0)
                    for value in x_y:
                        mask_1[z_axis, value[0], value[1]] = 1
                    self.mask[z_axis, ::] = \
                        ndi.binary_fill_holes(
                            mask_1[z_axis, ::]).astype(int)

            if z_axis in slicer:
                # save and diplay the mask (Magna valve label)
                nodes_copy = numpy_nodes
                z = nodes_copy[numpy_nodes[:, 2] == z_axis]
                x_y = np.ndarray(shape=(z.shape[0], 2), dtype=int)
                x_y[:] = z[:, 0:2]
                x_y = np.unique(x_y, axis=0)
                for value in x_y:
                    mask_2[z_axis,
                           value[0],
                           value[1]] = 1
                if self.multi_label:
                    mask_2[z_axis, ::] = \
                        ndi.binary_fill_holes(
                            mask_2[z_axis, ::]).astype(int)
                self.mask[z_axis, ::] += mask_2[z_axis, ::]

            self.mask[z_axis, ::] = \
                np.rot90(self.mask[z_axis, ::])
            # To be aligned with 2D show.
            # Need to fill contour image
            # Not perfect so will use manual tool correction
            # https://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html
            # https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
            # Can not be binary_fill_holes in multi label detection otherwise
            # will fill everything with 1 value
            if not self.multi_label:
                self.mask[z_axis, ::] = \
                    ndi.binary_fill_holes(self.mask[z_axis, ::]).astype(int)
            np.save(
                os.path.join(directory,
                             self.title) + '/' + str(z_axis),
                self.mask[z_axis, ::])
            # plt.imsave(
            #     os.path.join(directory,
            #                  self.title) + '/' + str(z_axis) + ".png",
            #     self.mask[z_axis, ::])

        print('Labeling done !')

    def create_window(self,
                      render,
                      renderwindow,
                      interactor):
        """Create windows."""
        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetSize(int(self.window_size[0]),
                             int(self.window_size[1]))
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        return renderer, renderWindow, renderWindowInteractor

    def close_window(self, iren):
        """Close render."""
        render_window = iren.GetRenderWindow()
        render_window.Finalize()

    def show(self):
        """Show render."""
        self.rw.Render()
        self.iren.Start()

    def update(self, obj, ev):
        """Update render."""
        key = self.iren.GetKeySym()
        if key == 'Left':
            if self.frame > 0:
                for render, actor in zip(self.render_list,
                                         self.actor_list):
                    render.RemoveActor(actor)
                for render in self.render_list:
                    if render == self.render_score:
                        for actor_ in self.actor_gt_list:
                            render.RemoveActor(actor_)
                        for actor_ in self.actor_infer_list:
                            render.RemoveActor(actor_)
                        for actor_ in self.actor_fitting_list:
                            render.RemoveActor(actor_)
                self.actor_list = []
                self.actor_gt_list = []
                self.actor_infer_list = []
                self.actor_fitting_list = []
                self.frame += -1
                for mode in self.mode:
                    _ = self.add_actors(mode, self.frame)
                for render, actor in zip(self.render_list,
                                         self.actor_list):
                    render.AddActor(actor)
                self.rw.Render()

        elif key == 'Right':
            if self.frame < (len(self.data_path)-1):
                for render, actor in zip(self.render_list,
                                         self.actor_list):
                    render.RemoveActor(actor)
                for render in self.render_list:
                    if render == self.render_score:
                        for actor_ in self.actor_gt_list:
                            render.RemoveActor(actor_)
                        for actor_ in self.actor_infer_list:
                            render.RemoveActor(actor_)
                        for actor_ in self.actor_fitting_list:
                            render.RemoveActor(actor_)
                self.actor_list = []
                self.actor_gt_list = []
                self.actor_infer_list = []
                self.actor_fitting_list = []
                self.frame += 1
                for mode in self.mode:
                    _ = self.add_actors(mode, self.frame)
                for render, actor in zip(self.render_list,
                                         self.actor_list):
                    render.AddActor(actor)
                self.rw.Render()

        elif key == 'p':
            for render in self.render_list:
                render.ResetCamera()
        elif key == 'plus':
            self.valve_value += 1
            for render in self.render_list:
                if render == self.render_score:
                    render.RemoveActor(self.valve_value_actor)
                    self.valve_value_actor = \
                        Text_2D(str(self.valve_value),
                                pos=[0.49, 0.87],
                                s=0.8,
                                c=None,
                                alpha=1,
                                bg=None,
                                font="Montserrat",
                                justify="bottom-left",
                                bold=False,
                                italic=False)
                    render.AddActor2D(self.valve_value_actor)
            self.rw.Render()
        elif key == 'minus':
            self.valve_value -= 1
            for render in self.render_list:
                if render == self.render_score:
                    render.RemoveActor(self.valve_value_actor)
                    self.valve_value_actor = \
                        Text_2D(str(self.valve_value),
                                pos=[0.49, 0.87],
                                s=0.8,
                                c=None,
                                alpha=1,
                                bg=None,
                                font="Montserrat",
                                justify="bottom-left",
                                bold=False,
                                italic=False)
                    render.AddActor2D(self.valve_value_actor)
            self.rw.Render()
        else:
            pass
        return

    def _mouseleft(self, obj, event):
        iren = self.iren
        x, y = iren.GetEventPosition()

        renderer = iren.FindPokedRenderer(x, y)
        self.renderer = renderer

        picker = vtk.vtkPropPicker()
        picker.PickProp(x, y, renderer)
        clickedActor = picker.GetActor()

        # check if any button objects are clicked
        clickedActor2D = picker.GetActor2D()
        if clickedActor2D:
            for bt in self.buttons:
                if clickedActor2D == bt.actor:
                    bt.function()
                    break
