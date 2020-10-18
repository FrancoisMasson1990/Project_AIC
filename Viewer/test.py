import vtk

# Need to create a class with sphere defined in init !!!!
# Call back function
def boxCallback(obj, event):
    sphere.SetRadius(obj.GetRepresentation().GetValue())

# A renderer and render window
renderer = vtk.vtkRenderer()
renderer.SetBackground(0, 0, 0)

# Create a Cone
sphereSource = vtk.vtkSphereSource()
#cone.SetResolution(20)
coneMapper = vtk.vtkPolyDataMapper()
coneMapper.SetInputConnection(sphereSource.GetOutputPort())
coneActor = vtk.vtkActor()
coneActor.SetMapper(coneMapper)

# A renderer and render window
renderer.AddActor(coneActor)

renwin = vtk.vtkRenderWindow()
renwin.AddRenderer(renderer)
 
# An interactor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(renwin)

sliderRep = vtk.vtkSliderRepresentation2D()
sliderRep.SetMinimumValue(0)
sliderRep.SetMaximumValue(100)
sliderRep.SetValue(50)
sliderRep.SetSliderLength(0.015)
sliderRep.SetSliderWidth(0.05)
sliderRep.SetEndCapLength(0.01)
sliderRep.SetEndCapWidth(0.01)
sliderRep.SetTubeWidth(0.0075)
sliderRep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
sliderRep.GetPoint1Coordinate().SetValue(0.1 ,0.07)
sliderRep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
sliderRep.GetPoint2Coordinate().SetValue(0.9, 0.07)

sliderWidget = vtk.vtkSliderWidget()
sliderWidget.SetInteractor(interactor)
sliderWidget.SetAnimationModeToJump()
sliderWidget.SetRepresentation(sliderRep)
sliderWidget.AddObserver("InteractionEvent", boxCallback)
sliderWidget.EnabledOn()
 
# Start
interactor.Initialize()
interactor.Start()