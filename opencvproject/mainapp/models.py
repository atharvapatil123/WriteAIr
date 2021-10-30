from django.db import models
from django.contrib.auth.models import User
# Create your models here.
class NotesOfUser(models.Model):
    created=models.DateTimeField(auto_now_add=True)
    name=models.CharField(max_length=200)
    
    author=models.ForeignKey(User,on_delete=models.CASCADE)