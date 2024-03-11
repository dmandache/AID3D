import vtk
from vtk.util import numpy_support
import numpy as np


def create_3d_volume_renderer(numpy_volume, pixel_spacing):
    """
    Create a VTK renderer for displaying a 3D volume.

    Parameters:
    - numpy_volume (numpy.ndarray): 3D NumPy array representing the volume with dimensions (width, height, channels).
    - pixel_spacing (tuple): Tuple of three values representing pixel spacing in x, y, and z directions.

    Returns:
    - vtk.vtkRenderer: VTK renderer for 3D volume.
    """
    # Ensure the input array has the correct shape (width, height, channels)
    if numpy_volume.shape[2] <= 1:
        raise ValueError(
            "The input array should have more than one channel for 3D volume visualization."
        )

    # Create a VTK image data object
    vtk_image_data = vtk.vtkImageData()
    vtk_image_data.SetDimensions(numpy_volume.shape)
    vtk_image_data.SetSpacing(pixel_spacing)

    # Flatten the NumPy array and set the scalar values in VTK data
    flat_data = numpy_volume.flatten(
        order="F"
    )  # Transpose to (height, width, channels)
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
    renderer.SetBackground(1, 1, 1)  # Set background to white
    renderer.AddVolume(volume)

    return renderer


def create_slice_explorer_renderer(numpy_volume, pixel_spacing):
    """
    Create a VTK renderer for exploring 3D volume slice by slice.

    Parameters:
    - numpy_volume (numpy.ndarray): 3D NumPy array representing the volume with dimensions (width, height, channels).
    - pixel_spacing (tuple): Tuple of three values representing pixel spacing in x, y, and z directions.

    Returns:
    - vtk.vtkRenderer: VTK renderer for slice exploration.
    """
    # Ensure the input array has the correct shape (width, height, channels)
    if numpy_volume.shape[2] <= 1:
        raise ValueError(
            "The input array should have more than one channel for slice exploration."
        )

    # Create a VTK renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 0, 0)  # Set background to white

    # Create a VTK image actor for displaying 2D slices
    image_actor = vtk.vtkImageActor()
    renderer.AddActor(image_actor)

    # Set up a slider widget for exploring slices
    slider_widget = vtk.vtkSliderWidget()
    slider_widget.SetRepresentation(vtk.vtkSliderRepresentation2D())
    slider_widget.GetRepresentation().SetMinimumValue(0)
    slider_widget.GetRepresentation().SetMaximumValue(numpy_volume.shape[2] - 1)
    slider_widget.GetRepresentation().SetValue(0)
    slider_widget.GetRepresentation().SetTitleText("Slice")
    slider_widget.GetRepresentation().GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_widget.GetRepresentation().GetPoint1Coordinate().SetValue(0.1, 0.1)
    slider_widget.GetRepresentation().GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider_widget.GetRepresentation().GetPoint2Coordinate().SetValue(0.9, 0.1)
    slider_widget.SetAnimationModeToAnimate()

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

        renderer.Render()

    # Connect the callback function to the slider widget
    slider_widget.AddObserver("InteractionEvent", update_slice_position)

    return renderer, slider_widget


def show_side_by_side_renderers(numpy_volume, pixel_spacing):
    """
    Show side-by-side VTK renderers for 3D volume and slice exploration.

    Parameters:
    - numpy_volume (numpy.ndarray): 3D NumPy array representing the volume with dimensions (width, height, channels).
    - pixel_spacing (tuple): Tuple of three values representing pixel spacing in x, y, and z directions.
    """
    # Create a VTK render window
    render_window = vtk.vtkRenderWindow()
    render_window.SetWindowName("Side-by-Side Renderers")
    render_window.SetSize(1600, 800)

    # Create a VTK render window interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Create 3D volume renderer
    volume_renderer = create_3d_volume_renderer(numpy_volume, pixel_spacing)
    volume_renderer.SetViewport(0, 0, 0.5, 1.0)

    # Create slice explorer renderer
    slice_renderer, slider_widget = create_slice_explorer_renderer(
        numpy_volume, pixel_spacing
    )
    slice_renderer.SetViewport(0.5, 0, 1.0, 1.0)

    # Add renderers to the render window
    render_window.AddRenderer(volume_renderer)
    render_window.AddRenderer(slice_renderer)

    # Render the scene
    render_window.Render()

    # Start the VTK event loop
    render_window_interactor.Start()
