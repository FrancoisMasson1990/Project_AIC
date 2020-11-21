## Main class to visualize data dicom
import vtk
# Disable vtk Warning
vtk.vtkObject.GlobalWarningDisplayOff()
from vtkplotter import *
import numpy as np
from vtkplotter import settings
import vtkplotter.settings as settings
import vtkplotter.addons as addons
from vtk.util.numpy_support import vtk_to_numpy,numpy_to_vtk
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian
import pydicom._storage_sopclass_uids
import glob
from widget import *
from tqdm import tqdm 
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from aic_models import data_preprocess as dp
from skimage.transform import resize
from tqdm import tqdm
from cylinder_fitting import fit
from viewer_2D import Viewer2D

class Viewer3D(object):

    def __init__(self,data_path:str,frame=0,mode=1,label='/label_mask/',npy=None,multi_label=False,model=None):
        
        self.frame = 0
        self.init = True
        self.icol = 0
        self.data_path = data_path
        self.colors = vtk.vtkNamedColors()
        self.window_size = (1200, 800)
        self.model = model
        if mode ==  1 :
            self.mode = ['ray_cast']
        elif mode == 2 :
            self.mode = ['ray_cast','iso']
        elif mode == 4 :
            self.mode = ['ray_cast','iso','slicer_2d','inference']
        
        self.actor_list = []
        self.buttons = []
        self.render_list = []
        self.axes_list = []
        self.grid_list = []
        self.actor_gt_list = []
        self.actor_infer_list = []
        self.actor_fitting_list = []
        self.cutter_tool = False
        self.valve_value = 20
        self.window_2D = False
        self.window_3Dslice = False
        self.fitting_bool = False
        self.cutter_obj = []
        self.widget_cut = None
        self.mask = None
        self.render_score = None
        self.spacing = None
        self.title = None
        self.predictions_final = None
        self.area = None
        self.label_folder = label
        self.npy_folder = npy
        self.multi_label = multi_label

        '''One render window, multiple viewports'''
        self.rw = vtk.vtkRenderWindow()
        self.rw.SetSize(int(self.window_size[0]), int(self.window_size[1]))
        self.iren = vtk.vtkRenderWindowInteractor()

        ## Freeze windows if not clicked on actor !!!!
        vsty = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(vsty)
        self.view_mode = True
        self.gt_view_mode = False
        self.infer_view_mode = False

        ## Callback to update content
        self.iren.RemoveObservers('KeyPressEvent')
        self.iren.AddObserver('LeftButtonPressEvent', self._mouseleft)
        self.iren.AddObserver('KeyPressEvent', self.update)

        self.iren.SetRenderWindow(self.rw)
        self.viewport_frame()
        self.viewport()

    def viewport_frame(self):

        self.xmins=[]
        self.xmaxs=[]
        self.ymins=[]
        self.ymaxs=[]

        # Define viewport ranges
        for size in range(len(self.mode)):
            self.xmins.append(size/len(self.mode))
            self.xmaxs.append(size/len(self.mode) + 1/len(self.mode))
            self.ymins.append(0)
            self.ymaxs.append(1)
        
        if len(self.mode) == 4:
            self.xmins=[0,.5,0,.5]
            self.xmaxs=[0.5,1,0.5,1]
            self.ymins=[0.5,0.5,0,0]
            self.ymaxs=[1,1,0.5,0.5]

        self.xmins=np.asarray(self.xmins)
        self.xmaxs=np.asarray(self.xmaxs)
        self.ymins=np.asarray(self.ymins)
        self.ymaxs=np.asarray(self.ymaxs)

        # Have some fun with colors
        #self.ren_bkg = ['AliceBlue', 'GhostWhite', 'WhiteSmoke', 'Seashell']
        #self.ren_bkg = ['GhostWhite']
        self.ren_bkg = ['Black']
    
    def viewport(self):
        for i in range(len(self.mode)):
            self.ren = vtk.vtkRenderer()
            self.render_list.append(self.ren)
            self.rw.AddRenderer(self.ren)
            self.ren.SetViewport(self.xmins[i], self.ymins[i], self.xmaxs[i], self.ymaxs[i])
            self.actor = self.add_actors(self.mode[i],self.frame)
            if self.actor != None:
                self.ren.AddActor(self.actor)
            if self.mode[i] == 'ray_cast':
                ## button used for the colors changing
                states, c, bc, pos, size, font, bold, italic, alpha, angle = self.button_cast(pos=[0.7, 0.035],states=["State 1", "State 2"])
                self.but = Button(self.buttonfuncMode, states, c, bc, pos, size, font, bold, italic, alpha, angle).status(int(0))
                self.ren.AddActor2D(self.but.actor)
                self.buttons.append(self.but)

                if len(self.axes_list)==0:
                   self.axes = Axes(self.iren)
                   self.axes_list.append(self.axes.axact)
                if len(self.grid_list)==0:
                    self.grid = Grid(self.actor)
                    self.grid_list.append(self.grid.grid)
                    self.ren.AddActor(self.grid.grid)

            elif self.mode[i] == 'iso' :
                ## button used for the slicer 2d
                states, c, bc, pos, size, font, bold, italic, alpha, angle = self.button_cast(pos=[0.7, 0.035],states=["Cutter On", "Cutter Off"])
                self.cutter = Button(self.buttonfuncMode_cutter, states, c, bc, pos, size, font, bold, italic, alpha, angle).status(int(0))
                self.ren.AddActor2D(self.cutter.actor)
                self.buttons.append(self.cutter)

                ## button used for saving
                states, c, bc, pos, size, font, bold, italic, alpha, angle = self.button_cast(pos=[0.5, 0.035],states=["Save"])
                self.saver = Button(self.buttonfuncMode_saving, states, c, bc, pos, size, font, bold, italic, alpha, angle).status(int(0))
                self.ren.AddActor2D(self.saver.actor)
                self.buttons.append(self.saver)

            elif self.mode[i] == 'slicer_2d' :
                # ## button used for the 2D Slicer and change view
                ## button used for the colors changing
                states, c, bc, pos, size, font, bold, italic, alpha, angle = self.button_cast(pos=[0.7, 0.035],states=["Start Slicer Mode", "Stop Slicer Mode"])
                self.but_ = Button(self.buttonviewMode, states, c, bc, pos, size, font, bold, italic, alpha, angle).status(int(0))
                self.ren.AddActor2D(self.but_.actor)
                self.buttons.append(self.but_)

            elif self.mode[i] == 'inference' :
                self.render_score = self.ren
                ## button used for diameter value
                self.valve_dia = Text_2D("Valve diameter :",pos=[0.01, 0.87],s=0.8,c=None,alpha=1,bg=None,font="Montserrat",justify="bottom-left",bold=False,italic=False)
                self.ren.AddActor2D(self.valve_dia)

                self.valve_value_actor = Text_2D(str(self.valve_value),pos=[0.49, 0.87],s=0.8,c=None,alpha=1,bg=None,font="Montserrat",justify="bottom-left",bold=False,italic=False)
                self.ren.AddActor2D(self.valve_value_actor)

                ## button used for inference and GT
                states, c, bc, pos, size, font, bold, italic, alpha, angle = self.button_cast(pos=[0.5, 0.035],states=["Inference (On)","Inference (Off)"])
                self.infer = Button(self.buttonfuncInference, states, c, bc, pos, size, font, bold, italic, alpha, angle).status(int(0))
                self.ren.AddActor2D(self.infer.actor)
                self.buttons.append(self.infer)

                states, c, bc, pos, size, font, bold, italic, alpha, angle = self.button_cast(pos=[0.15, 0.035],states=["Ground Truth (On)","Ground Truth (Off)"])
                self.ground_truth = Button(self.buttonfuncGroundTruth, states, c, bc, pos, size, font, bold, italic, alpha, angle).status(int(0))
                self.ren.AddActor2D(self.ground_truth.actor)
                self.buttons.append(self.ground_truth)

                states, c, bc, pos, size, font, bold, italic, alpha, angle = self.button_cast(pos=[0.8, 0.035],states=["Fitting (On)","Fitting (Off)"])
                self.fitting = Button(self.buttonfuncFitting, states, c, bc, pos, size, font, bold, italic, alpha, angle).status(int(0))
                self.ren.AddActor2D(self.fitting.actor)
                self.buttons.append(self.fitting)

                states, c, bc, pos, size, font, bold, italic, alpha, angle = self.button_cast(pos=[0.12, 0.94],states=["Agatston Score"])
                self.score_ratio = Button(self.buttonfuncAgatston, states, c, bc, pos, size, font, bold, italic, alpha, angle).status(int(0))
                self.ren.AddActor2D(self.score_ratio.actor)
                self.buttons.append(self.score_ratio)
            
            self.ren.SetBackground(self.colors.GetColor3d(self.ren_bkg[0]))
            self.ren.ResetCamera()
            self.camera_position()
    
    def add_actors(self,mode,frame):
        ## Update title : 
        txt = glob.glob(self.data_path[frame] + '/*.txt')
        self.title = self.data_path[frame].split('/')
        self.title = self.title[-2] + '/' + self.title[-1] 
        if len(txt) > 0 :
            txt=txt[0]
            f = open(txt, "r")
            lines = f.readlines()
            #title = lines[0]
            valve = lines[2]
            valve = valve[:-1] ## Remove wrong asci character
            self.rw.SetWindowName(valve)
        else :
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
        else :
           return actor

    def buttonfuncMode(self):
        s = self.volume.mode()
        snew = (s + 1) % 2
        self.volume.mode(snew)
        self.but.switch()

    def buttonviewMode(self):
        self.but_.switch()
        if self.view_mode:
            style = vtk.vtkInteractorStyleImage()
            style.SetInteractionModeToImage3D()
            self.view_mode = not self.view_mode
        else :
            style = vtk.vtkInteractorStyleTrackballCamera()
            self.view_mode = not self.view_mode
        self.iren.SetInteractorStyle(style)
        
    def buttonfuncMode_cutter(self):
        if not self.cutter_tool :
            self.cutter_obj.append(Cutter(self.render_list[-1],self.iren,self.cutter_actor))
            for cut in self.cutter_obj:
                self.widget_cut = cut.boxWidget
                self.widget_cut.On()
            self.cutter_tool = not self.cutter_tool
        else :
            for cut in self.cutter_obj:
                self.widget_cut = cut.boxWidget
                self.widget_cut.Off()
                cut.actor.mapper().RemoveAllClippingPlanes()
            self.cutter_tool = not self.cutter_tool
            self.cutter_obj = []
        self.cutter.switch()

    def buttonfuncMode_saving(self):
        if self.cutter_tool:
            for cut in self.cutter_obj:
                ## Save the remaining points
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
        self.ground_truth.switch()
        if not self.gt_view_mode:
            numpy_3d = glob.glob(os.path.join(self.npy_folder,self.title) + '/*.npy')
            if len(numpy_3d)>0:
                actor_ = self.label_3d(numpy_3d[0],c=[1,0,0])
                self.actor_gt_list.append(actor_)
                for render in self.render_list:
                    if render == self.render_score:
                        render.AddActor(actor_)                        
            self.rw.Render()
            self.ren.ResetCamera()
            self.camera_position()
            self.gt_view_mode = not self.gt_view_mode
        else : 
            for render in self.render_list:
                if render == self.render_score:
                    for actor_ in self.actor_gt_list:
                        render.RemoveActor(actor_)
            self.actor_gt_list = []
            self.rw.Render()
            self.gt_view_mode = not self.gt_view_mode

    def buttonfuncInference(self):
        self.infer.switch()
        if self.model is not None:
            if not self.infer_view_mode:
                #Prediction
                idx = os.path.join(self.data_path[self.frame])
                img = dp.load_scan(idx)
                img = dp.get_pixels_hu(img)
                shape_x = img.shape[1]
                shape_y = img.shape[2]
                img = dp.preprocess_inputs(img)
                pred_list = []
                for i in tqdm(range(img.shape[0])):
                    pred = np.expand_dims(img[i,:,:,:], 0)
                    prediction = self.model.predict(pred)
                    prediction = np.argmax(prediction.squeeze(),axis=-1)
                    prediction = np.rot90(prediction,axes=(1,0)) 
                    prediction = np.expand_dims(prediction, 0)
                    prediction[prediction == 0] = -1 # Validate
                    pred_list.append(prediction)
                
                predictions = np.vstack(pred_list) 
                predictions,_ = dp.resample(predictions,[self.spacing[0],self.spacing[1]],self.spacing[2],[1,1,1])
                vertices,_ = dp.make_mesh(predictions,-1)
                # Clustering
                vertices = dp.clustering(vertices,self.center,ratio=0.3,threshold=3800)
                # Fit with closest true points
                self.predictions_final = dp.boxe_3d(self.all_numpy_nodes,vertices)
                self.predictions_agatston = dp.boxe_3d(self.all_numpy_nodes_agatston,vertices)
                actor_ = self.label_3d(self.predictions_final,c=[0,1,0])
                self.actor_infer_list.append(actor_)
                for render in self.render_list:
                    if render == self.render_score:
                        render.AddActor(actor_)
                self.rw.Render()
                self.ren.ResetCamera()
                self.camera_position()
                self.infer_view_mode = not self.infer_view_mode
            else :
                for render in self.render_list:
                    if render == self.render_score:
                        for actor_ in self.actor_infer_list:
                            render.RemoveActor(actor_)
                self.actor_infer_list = []
                self.rw.Render()
                self.infer_view_mode = not self.infer_view_mode

    def buttonfuncFitting(self):
        """
        w_fit = Direction of the cylinder axis
        C_fit = A point on the cylinder axis
        r_fit = Radius of the cylinder
        fit_err = Fitting error (G function)
        """
        self.fitting.switch()     
        if (self.fitting.status() == "Fitting (Off)") and (self.predictions_final is not None):
            # Cylinder Fit
            print("performing fitting...")
            self.w_fit, self.C_fit, self.r_fit, self.fit_err = fit(self.predictions_final)
            print("fitting done !")  
            actor = Cylinder(pos=tuple(self.C_fit),r=self.r_fit,height=20,axis=tuple(self.w_fit),alpha=0.5,c="white")
            self.actor_fitting_list.append(actor)
            for render in self.render_list:
                if render == self.render_score:
                    render.AddActor(actor)          
            self.rw.Render()
            self.ren.ResetCamera()
            self.camera_position()
            self.fitting_bool = True
        else :
            for render in self.render_list:
                if render == self.render_score:
                    for actor_ in self.actor_fitting_list:
                        render.RemoveActor(actor_)
            self.actor_fitting_list = []
            self.rw.Render()
            self.fitting_bool = False

    def buttonfuncAgatston(self):
        if self.fitting_bool == True:
            #Points in cylinder
            d = []
            for point in tqdm(self.predictions_agatston):
                d.append(dp.point_line_distance(point,self.C_fit,self.w_fit))
            d = np.array(d)
            points_filter = self.predictions_agatston[np.where(d<self.valve_value/2)]
            points_filter [:,0] = np.around(points_filter [:,0]/self.spacing[0])
            points_filter [:,1] = np.around(points_filter [:,1]/self.spacing[1])
            points_filter [:,2] = np.around(points_filter [:,2]/self.spacing[2])
            # Along the z axis : Later Along x axis and y axis ... (3D Object Detection from CT Scans using a Slice-and-fuse Approach)
            slicer = np.unique(points_filter[:,2]).astype(int)

            mask_agatston = self.mask.copy()
            for z_axis in range(self.mask.shape[0]):
                if z_axis in slicer:
                    z = points_filter[points_filter[:,2]==z_axis]
                    x_y = np.ndarray(shape=(z.shape[0],2), dtype=int)
                    x_y[:] = z[:,0:2]
                    x_y = np.unique(x_y, axis=0)
                    for value in x_y :
                        mask_agatston[z_axis,value[0],value[1]] = 1               
                    mask_agatston[z_axis,::] = np.rot90(mask_agatston[z_axis,::])
            
            # Show the score in 2D mode
            Viewer2D(data_path=self.data_path,folder_mask="",frame=self.frame,model="",mask_agatston=mask_agatston,agatston=True,area=self.area)

    def button_cast(self,pos:list=None,states:list=None):
        c=["bb", "gray"]
        bc=["gray", "bb"]  # colors of states
        font="arial"
        size=16
        bold=0
        italic=False
        alpha=1
        angle=0
        return states, c, bc, pos, size, font, bold, italic, alpha, angle

    def camera_position(self,zoom=0.7):
        self.camera = self.ren.GetActiveCamera()
        self.camera.Zoom(zoom)

    def ray_cast(self,data):
        self.img = vtkio.load(data).imagedata()
        if self.init:
            self.volume = Volume(self.img,c='jet',mode=int(0))
            self.volume.jittering(True)
        else : 
            self.volume._update(self.img)
        self.center = self.volume.center()
        
        return self.volume
    
    def iso_surface(self,data):
        self.img = vtkio.load(data)
        self.img = self.img.isosurface()
        ## Following lines used to get the mask 
        self.mask = tuple(reversed(vtkio.load(data).imagedata().GetDimensions()))

        self.mask = np.zeros(self.mask,dtype=int)
        self.spacing = vtkio.load(data).imagedata().GetSpacing()
        self.area = self.spacing[0]*self.spacing[1]

        # Get the all points in isosurface
        self.points = self.img.GetMapper().GetInput()
        self.all_array = self.points.GetPoints()
        self.all_numpy_nodes = vtk_to_numpy(self.all_array.GetData())

        # Get the all points for agatston score
        self.img_agatston = vtkio.load(data)
        self.img_agatston = self.img_agatston.isosurface(threshold=130)
        self.points_agatston = self.img_agatston.GetMapper().GetInput()
        self.all_array_agatston = self.points_agatston.GetPoints()
        self.all_numpy_nodes_agatston = vtk_to_numpy(self.all_array_agatston.GetData())
            
        return self.img

    def slicer_2d(self,data):
        self.image = vtkio.load(data).imagedata()
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

        printc("Slicer Mode:", invert=1, c="m")
        printc(
            """Press  SHIFT+Left mouse    to rotate the camera for oblique slicing
            SHIFT+Middle mouse  to slice perpendicularly through the image
            Left mouse and Drag to modify luminosity and contrast
            X                   to Reset to sagittal view
            Y                   to Reset to coronal view
            Z                   to Reset to axial view
            R                   to Reset the Window/Levels
            Q                   to Quit.""",
            c="m",
        )
        
        return self.ia

    def label_3d(self,data,c=[1,0,0]):
        if isinstance(data,str):
            with open(data, 'rb') as f:
                self.img = np.load(f)  
        if isinstance(data,np.ndarray):
            self.img = data
        self.points = vtk.vtkPoints()
        # Create the topology of the point (a vertex)
        self.vertices = vtk.vtkCellArray()
        # Add points
        for i in range(0, len(self.img)):
            p = self.img[i].tolist()
            point_id = self.points.InsertNextPoint(p)
            self.vertices.InsertNextCell(1)
            self.vertices.InsertCellPoint(point_id)
        # Create a poly data object
        polydata = vtk.vtkPolyData()
        # Set the points and vertices we created as the geometry and topology of the polydata
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
    
    def numpy_saving(self,points):
        ## Will try to fit a rectangle and to retrieve the x,y,z coordinate for segmentation...
        array = points.GetOutput().GetPoints()
        numpy_nodes = vtk_to_numpy(array.GetData())

        if os.path.exists(self.npy_folder):
            directory = self.npy_folder
        else :
            directory = os.getcwd() + self.npy_folder
        os.makedirs(os.path.join(directory,self.title),exist_ok = True) ## Make folder recursively

        with open(os.path.join(directory,self.title) + '/' + 'volume_label.npy', 'wb') as f:
            np.save(f,numpy_nodes)

        numpy_nodes[:,0] = np.around(numpy_nodes[:,0]/self.spacing[0])
        numpy_nodes[:,1] = np.around(numpy_nodes[:,1]/self.spacing[1])
        numpy_nodes[:,2] = np.around(numpy_nodes[:,2]/self.spacing[2])
        # Along the z axis : Later Along x axis and y axis ... (3D Object Detection from CT Scans using a Slice-and-fuse Approach)
        slicer = np.unique(numpy_nodes[:,2]).astype(int)

        if self.multi_label:
            self.all_numpy_nodes[:,0] = np.around(self.all_numpy_nodes[:,0]/self.spacing[0])
            self.all_numpy_nodes[:,1] = np.around(self.all_numpy_nodes[:,1]/self.spacing[1])
            self.all_numpy_nodes[:,2] = np.around(self.all_numpy_nodes[:,2]/self.spacing[2])
            slicer_tot = np.unique(self.all_numpy_nodes[:,2]).astype(int)    

        if os.path.exists(self.label_folder):
            directory = self.label_folder
        else :
            directory = os.getcwd() + self.label_folder
        os.makedirs(os.path.join(directory,self.title),exist_ok = True) ## Make folder recursively

        mask_1 = self.mask.copy()
        mask_2 = self.mask.copy()
        for z_axis in range(self.mask.shape[0]):
            if self.multi_label:
                if z_axis in slicer_tot:
                    #save and diplay the mask (Not magna valve label)
                    nodes_copy = self.all_numpy_nodes
                    z = nodes_copy[self.all_numpy_nodes[:,2]==z_axis]
                    x_y = np.ndarray(shape=(z.shape[0],2), dtype=int)
                    x_y[:] = z[:,0:2]
                    x_y = np.unique(x_y, axis=0)
                    for value in x_y :
                        mask_1[z_axis,value[0],value[1]] = 1
                    self.mask[z_axis,::] = ndi.binary_fill_holes(mask_1[z_axis,::]).astype(int)

            if z_axis in slicer:
                #save and diplay the mask (Magna valve label)
                nodes_copy = numpy_nodes
                z = nodes_copy[numpy_nodes[:,2]==z_axis]
                x_y = np.ndarray(shape=(z.shape[0],2), dtype=int)
                x_y[:] = z[:,0:2]
                x_y = np.unique(x_y, axis=0)
                for value in x_y :
                    mask_2[z_axis,value[0],value[1]] = 1
                if self.multi_label:
                    mask_2[z_axis,::] = ndi.binary_fill_holes(mask_2[z_axis,::]).astype(int)
                self.mask[z_axis,::] += mask_2[z_axis,::]
            
            self.mask[z_axis,::] = np.rot90(self.mask[z_axis,::]) # To be aligned with 2D show. Can be modified in the future !        
            ## Need to fill contour image - Not perfect so will use manual tool correction
            ## https://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html
            ## https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
            #Can not be binary_fill_holes in multi label detection otherwise will fill everything with 1 value
            if not self.multi_label:
                self.mask[z_axis,::] = ndi.binary_fill_holes(self.mask[z_axis,::]).astype(int)
            np.save(os.path.join(directory,self.title) + '/' + str(z_axis),self.mask[z_axis,::])
            #plt.imsave(os.path.join(directory,self.title) + '/' + str(z_axis) + ".png",self.mask[z_axis,::])
        
        print('Labeling done !')

    def create_window(self,render,renderwindow,interactor):
        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetSize(int(self.window_size[0]), int(self.window_size[1]))
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        return renderer,renderWindow,renderWindowInteractor
    
    def close_window(self,iren):
        render_window = iren.GetRenderWindow()
        render_window.Finalize()

    def show(self):
        self.rw.Render()
        self.iren.Start()
    
    def update(self,obj,ev):
        key = self.iren.GetKeySym()
        if key == 'Left':
            if self.frame > 0:
                for render,actor in zip(self.render_list,self.actor_list):
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
                    _ = self.add_actors(mode,self.frame)
                for render,actor in zip(self.render_list,self.actor_list):
                    render.AddActor(actor)
                self.rw.Render()

        elif key == 'Right':
            if self.frame < (len(self.data_path)-1):
                for render,actor in zip(self.render_list,self.actor_list):
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
                   _ = self.add_actors(mode,self.frame)
                for render,actor in zip(self.render_list,self.actor_list):
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
                    self.valve_value_actor = Text_2D(str(self.valve_value),pos=[0.49, 0.87],s=0.8,c=None,alpha=1,bg=None,font="Montserrat",justify="bottom-left",bold=False,italic=False)
                    render.AddActor2D(self.valve_value_actor)
            self.rw.Render()
        elif key == 'minus':
            self.valve_value -= 1
            for render in self.render_list:
                if render == self.render_score:
                    render.RemoveActor(self.valve_value_actor)
                    self.valve_value_actor = Text_2D(str(self.valve_value),pos=[0.49, 0.87],s=0.8,c=None,alpha=1,bg=None,font="Montserrat",justify="bottom-left",bold=False,italic=False)
                    render.AddActor2D(self.valve_value_actor)
            self.rw.Render()
        else :
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