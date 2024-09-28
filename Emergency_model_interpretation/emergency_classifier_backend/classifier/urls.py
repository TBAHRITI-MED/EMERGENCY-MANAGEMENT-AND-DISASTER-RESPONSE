from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict_emergency, name='predict_emergency'),
    path('predict-fire-damage-estimation/', views.predict_fire_damage_map, name='predict_fire_damage_map'),
    path('predict-flood-damage/', views.predict_flood_damage, name='predict_flood_damage'),
    path('convert_tensor/<str:filename>/', views.tensor_to_image, name='tensor_to_image'),
    path('real_time_detection/', views.real_time_detection, name='real_time_detection'),
    path('video_feed/', views.video_feed, name='video_feed'),
]