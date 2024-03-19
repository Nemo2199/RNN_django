from django.urls import path
import dga_identification.views as views

urlpatterns = [
    path('', views.home, name='home'),
]
