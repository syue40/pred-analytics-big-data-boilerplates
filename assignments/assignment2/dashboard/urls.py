from django.urls import path
from .views import homePageView, results, homePost

urlpatterns = [
    path('', homePageView, name='home'),
    path('homePost/', homePost, name='homePost'),
    path('results/<str:radius_mean>/<str:texture_mean>/<str:perimeter_mean>/<str:area_mean>/<str:smoothness_mean>/<str:area_se>/<str:smoothness_worst>/<str:concave_points_worst>/<str:symmetry_worst>', results, name='results')
]   