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

class Viewer3D(object):

    def __init__(self,data_path:str,frame=0,mode=1,label='/label_mask/',npy=None):
        
        self.frame = 0
        self.icol = 0
        self.data_path = data_path
        self.colors = vtk.vtkNamedColors()
        self.window_size = (1200, 800)
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
        self.score_list = []
        self.actor_infer_list = []
        self.cutter_tool = False
        self.window_2D = False
        self.window_3Dslice = False
        self.cutter_obj = []
        self.widget_cut = None
        self.mask = None
        self.render_score = None
        self.spacing = None
        self.title = None
        self.label_folder = label
        self.npy_folder = npy

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
                ## button used for score and name 
                self.score_value = Text2D("0.0",pos=8,s=0.8,c=None,alpha=1,bg=None,font="Montserrat",justify="bottom-left",bold=False,italic=False)
                self.ren.AddActor2D(self.score_value)
                self.render_score = self.ren
                self.score_list.append(self.score_value)

                self.score_name = Text2D("Score :",pos=3,s=0.8,c=None,alpha=1,bg=None,font="Montserrat",justify="bottom-left",bold=False,italic=False)
                self.ren.AddActor2D(self.score_name)

                ## button used for inference and GT
                ## button used for inference
                states, c, bc, pos, size, font, bold, italic, alpha, angle = self.button_cast(pos=[0.7, 0.035],states=["Inference (On)","Inference (Off)"])
                self.infer = Button(self.buttonviewMode, states, c, bc, pos, size, font, bold, italic, alpha, angle).status(int(0))
                self.ren.AddActor2D(self.infer.actor)
                self.buttons.append(self.infer)

                states, c, bc, pos, size, font, bold, italic, alpha, angle = self.button_cast(pos=[0.3, 0.035],states=["Ground Truth (On)","Ground Truth (Off)"])
                self.ground_truth = Button(self.buttonfuncGroundTruth, states, c, bc, pos, size, font, bold, italic, alpha, angle).status(int(0))
                self.ren.AddActor2D(self.ground_truth.actor)
                self.buttons.append(self.ground_truth)

            
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
                self.actor_infer_list.append(actor_)
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
                    for actor_ in self.actor_infer_list:
                        render.RemoveActor(actor_)
            self.actor_infer_list = []
            self.rw.Render()
            self.gt_view_mode = not self.gt_view_mode

    def buttoninference(self):
        pass

    def inference(self):
        # Erase the score 
        # for score in self.score_list:
        #     for render in self.render_list:
        #         try : 
        #             render.RemoveActor(score)
        #         except:
        #             pass
        # self.rw.Render()
        # self.score_list = []
        # # Update the score
        # self.score_value = Text2D("10.0",pos=8,s=0.8,c=None,alpha=1,bg=None,font="Montserrat",justify="bottom-left",bold=False,italic=False)
        # for render in self.render_list:
        #     if render == self.render_score:
        #         render.AddActor2D(self.score_value)
        #         self.score_list.append(self.score_value)
        # self.rw.Render()
        pass

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
        self.volume = Volume(self.img,c='jet',mode=int(0))
        self.volume.jittering(True)        
        return self.volume
    
    def iso_surface(self,data):
        self.img = vtkio.load(data)
        self.img = self.img.isosurface()
        ## Following lines used to get the mask 
        self.mask = tuple(reversed(vtkio.load(data).imagedata().GetDimensions()))
        self.mask = np.zeros(self.mask)
        self.spacing = vtkio.load(data).imagedata().GetSpacing()
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
        with open(data, 'rb') as f:
            self.img = np.load(f)  
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
        #actor.GetProperty().SetPointSize(apoint_size)
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
        if os.path.exists(self.label_folder):
            directory = self.label_folder
        else :
            directory = os.getcwd() + self.label_folder
        os.makedirs(os.path.join(directory,self.title),exist_ok = True) ## Make folder recursively
        for z_axis in range(self.mask.shape[0]):
            if z_axis in slicer:
                #save and diplay le mask
                nodes_copy = numpy_nodes
                z = nodes_copy[numpy_nodes[:,2]==z_axis]
                x_y = np.ndarray(shape=(z.shape[0],2), dtype=int)
                x_y[:] = z[:,0:2]
                x_y = np.unique(x_y, axis=0)
                for value in x_y :
                    self.mask[z_axis,value[0],value[1]] = 1
            self.mask[z_axis,::] = np.rot90(self.mask[z_axis,::]) # To be aligned with 2D show. Can be modified in the future !        
            ## Need to fill contour image - Not perfect so will use manual tool correction
            ## https://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html
            ## https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
            # TODO :
            #from scipy import ndimage as ndi
            #self.mask[z_axis,::] = ndi.binary_fill_holes(self.mask[z_axis,::])
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
                        for actor_ in self.actor_infer_list:
                            render.RemoveActor(actor_)
                self.actor_list = []
                self.actor_infer_list = []
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
                        for actor_ in self.actor_infer_list:
                            render.RemoveActor(actor_)
                self.actor_list = []
                self.actor_infer_list = []
                self.frame += 1
                for mode in self.mode:
                    _ = self.add_actors(mode,self.frame)
                for render,actor in zip(self.render_list,self.actor_list):
                    render.AddActor(actor)
                self.rw.Render()

        elif key == 'p':
            for render in self.render_list:
                render.ResetCamera()

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