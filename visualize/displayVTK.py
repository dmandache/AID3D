import numpy as np
import vtk
from vtk.util import numpy_support


def display_volume(numpy_volume, pixel_spacing):

    # Create a VTK image data object
    vtk_image_data = vtk.vtkImageData()
    vtk_image_data.SetDimensions(numpy_volume.shape)
    vtk_image_data.SetSpacing(pixel_spacing)

    # Flatten the NumPy array and set the scalar values in VTK data
    flat_data = numpy_volume.flatten(
        order="F"
    )  # 'F' order is used to match VTK's ordering
    vtk_image_data.GetPointData().SetScalars(numpy_support.numpy_to_vtk(flat_data))

    # Create a VTK volume mapper
    volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()
    volume_mapper.SetInputData(vtk_image_data)

    # This class stores color data and can create color tables from a few color points.
    color = vtk.vtkPiecewiseFunction()
    color.AddPoint(-1024, 0.0)
    color.AddPoint(1023, 1)

    # Create transfer mapping scalar value to opacity
    alpha_channel = vtk.vtkPiecewiseFunction()
    alpha_channel.AddPoint(0, 0.0)
    alpha_channel.AddPoint(768, 1)

    # Create a VTK volume property
    volume_property = vtk.vtkVolumeProperty()
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()
    volume_property.SetColor(color)
    volume_property.SetScalarOpacity(alpha_channel)

    # Create a VTK volume
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    # Create a VTK renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0, 0, 1)  # Set background to white

    # Create a VTK render window
    render_window = vtk.vtkRenderWindow()
    render_window.SetWindowName("3D CT Scan Visualization")
    render_window.SetSize(800, 800)
    render_window.AddRenderer(renderer)

    # Create a VTK render window interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Add the volume to the renderer
    renderer.AddVolume(volume)

    # Set up the camera
    renderer.GetActiveCamera().Azimuth(0)
    renderer.GetActiveCamera().Elevation(45)
    renderer.GetActiveCamera().Roll(270)
    renderer.ResetCamera()

    # Render the scene
    render_window.Render()

    # Start the VTK event loop
    render_window_interactor.Start()


def explore_3d_volume_slices(numpy_volume, pixel_spacing):
    """
    Build a VTK renderer for exploring a 3D volume slice by slice.

    Parameters:
    - numpy_volume (numpy.ndarray): 3D NumPy array representing the volume.
    - pixel_spacing (tuple): Tuple of three values representing pixel spacing in x, y, and z directions.
    """
    # Create a VTK image data object
    vtk_image_data = vtk.vtkImageData()
    vtk_image_data.SetDimensions(numpy_volume.shape)
    vtk_image_data.SetSpacing(pixel_spacing)

    # Flatten the NumPy array and set the scalar values in VTK data
    flat_data = numpy_volume.flatten(
        order="F"
    )  # 'F' order is used to match VTK's ordering
    vtk_image_data.GetPointData().SetScalars(numpy_support.numpy_to_vtk(flat_data))

    # Create a VTK volume mapper
    volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()
    volume_mapper.SetInputData(vtk_image_data)

    # This class stores color data and can create color tables from a few color points.
    color = vtk.vtkPiecewiseFunction()
    color.AddPoint(-1024, 0.0)
    color.AddPoint(1023, 1)

    # Create transfer mapping scalar value to opacity
    alpha_channel = vtk.vtkPiecewiseFunction()
    alpha_channel.AddPoint(-512, 0.0)
    alpha_channel.AddPoint(1024, 1)

    # Create a VTK volume property
    volume_property = vtk.vtkVolumeProperty()
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()
    volume_property.SetColor(color)
    volume_property.SetScalarOpacity(alpha_channel)

    # Create a VTK volume
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    # Create a VTK renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 0, 0)  # Set background to white

    # Create a VTK render window
    render_window = vtk.vtkRenderWindow()
    render_window.SetWindowName("Explore 3D Volume Slices")
    render_window.SetSize(800, 800)
    render_window.AddRenderer(renderer)

    # Create a VTK render window interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Add the volume to the renderer
    renderer.AddVolume(volume)

    # Set up the camera
    renderer.GetActiveCamera().Azimuth(0)
    renderer.GetActiveCamera().Elevation(180)
    renderer.ResetCamera()

    # Create a VTK image actor for displaying 2D slices
    image_actor = vtk.vtkImageActor()
    image_actor.SetInputData(vtk_image_data)

    # Add the image actor to the renderer
    renderer.AddActor(image_actor)

    # Set up a slider widget for exploring slices
    slider_widget = vtk.vtkSliderWidget()
    slider_widget.SetInteractor(render_window_interactor)
    slider_widget.SetRepresentation(vtk.vtkSliderRepresentation2D())
    slider_widget.GetRepresentation().SetMinimumValue(0)
    slider_widget.GetRepresentation().SetMaximumValue(numpy_volume.shape[-1] - 1)
    slider_widget.GetRepresentation().SetValue(0)
    slider_widget.GetRepresentation().SetTitleText("Slice")
    slider_widget.GetRepresentation().GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_widget.GetRepresentation().GetPoint1Coordinate().SetValue(0.1, 0.1)
    slider_widget.GetRepresentation().GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_widget.GetRepresentation().GetPoint2Coordinate().SetValue(0.9, 0.1)
    slider_widget.SetAnimationModeToAnimate()
    slider_widget.EnabledOn()

    # Define a callback function to update the slice position
    def update_slice_position(obj, event):
        slider_value = int(slider_widget.GetRepresentation().GetValue())

        # Extract 2D slice from the 3D volume for display
        slice_data = numpy_volume[:, :, slider_value]
        vtk_slice_data = vtk.vtkImageData()
        vtk_slice_data.SetDimensions(slice_data.shape[0], slice_data.shape[1], 1)
        vtk_slice_data.SetSpacing(pixel_spacing[0], pixel_spacing[1], 1.0)
        flat_slice_data = slice_data.flatten(order="F")
        vtk_slice_data.GetPointData().SetScalars(
            numpy_support.numpy_to_vtk(flat_slice_data)
        )
        image_actor.SetInputData(vtk_slice_data)

        render_window.Render()

    # Connect the callback function to the slider widget
    slider_widget.AddObserver("InteractionEvent", update_slice_position)

    # Render the scene
    render_window.Render()

    # Start the VTK event loop
    render_window_interactor.Start()
