from django.contrib import admin
from django.urls import path,include
from . import views
app_name="mainapp"
urlpatterns = [
    path('', views.home,name='home'),
    path('login/', views.login,name='login'),
    path('register/', views.register,name='register'),
     path('logout/', views.logout,name='logout'),
  
    path('notes/', views.notes,name='notes'),
    path('screen/', views.screen,name='screen'),
    path('gen', views.gen,name='gen'),
    path('video_feed', views.video_feed,name='video_feed'),
    path('screen/', views.screen,name='screen'),
    

    
]
