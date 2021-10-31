from django import forms
from .models import NotesOfUser
class UploadForm(forms.ModelForm):
    class Meta:
        model=NotesOfUser
        fields=["name","image1",]
