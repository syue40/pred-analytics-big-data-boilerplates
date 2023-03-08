# Tensorflow/Keras
from tensorflow.keras.applications.vgg16 import VGG16
from trulens.nn.models import get_model_wrapper
keras_model = VGG16(weights='imagenet')
# Produce a wrapped model from the keras model.
model = get_model_wrapper(keras_model)
# Pytorch
from torchvision.models import vgg16
from trulens.nn.models import get_model_wrapper
pytorch_model = vgg16(pretrained=True)
# Produce a wrapped model from the pytorch model.
model = get_model_wrapper(pytorch_model, input_shape=(3,224,224), device='cpu')

from trulens.nn.attribution import InputAttribution
from trulens.visualizations import MaskVisualizer
# Create the attribution measure.
beagle_bike_input = "beaglebike.jpg"
saliency_map_computer = InputAttribution(model)
# Calculate the input attributions.
input_attributions = saliency_map_computer.attributions(beagle_bike_input)
# Visualize the attributions as a mask on the original image.
visualizer = MaskVisualizer(blur=10, threshold=0.95)
visualization = visualizer(input_attributions, beagle_bike_input)