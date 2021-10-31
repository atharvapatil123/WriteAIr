from django.shortcuts import render,redirect
from django.contrib import messages
from django.contrib.auth.models import User,auth
from .forms import UploadForm


from mainapp.models import NotesOfUser

# Create your views here.
def home(request):
    return render(request,'mainapp/home.html')

def register(request):
    if request.method=="POST":
        first_name=request.POST['first_name']
        last_name=request.POST['last_name']
        email=request.POST['email']
        username=request.POST['username']
        password1=request.POST['password1']
        password2=request.POST['password2']
        if password1==password2:
            if User.objects.filter(username=username).exists():
                messages.info(request,"Username already exists!")
                return redirect('mainapp:register')
            elif User.objects.filter(email=email).exists():
                messages.info(request,"Email already exists!")
                return redirect('mainapp:register')
            else:
                user=User.objects.create_user(username=username,password=password1,email=email,first_name=first_name,last_name=last_name)
                user.save()
                user1=auth.authenticate(username=username,password=password1)
                auth.login(request,user1)
                return redirect('mainapp:screen')
        else:
            #print("Not matching")
            messages.info(request,'Password is not matching..')
            return redirect('mainapp:register')
    return render(request,"mainapp/register.html")
def login(request):
    if request.method=="POST":
        username=request.POST.get('username')
        password=request.POST.get('password')
        user=auth.authenticate(username=username,password=password)
        if user is not None:
            auth.login(request,user)
            return redirect('mainapp:screen')
        else:
            messages.info(request,'Invalid credentials..Try again')
            return redirect('mainapp:login')
    return render(request,'mainapp/login.html')

def postSubmit(request):
    obj=NotesOfUser.objects.get(author=request.user)
    
    form=UploadForm(request.POST or None,request.FILES or None,instance=obj)
    if request.method=='POST':
        if form.is_valid():
            form.save()
    
    return render(request,'mainapp/notes.html',{'form':form})


    
def logout(request):
    auth.logout(request)
    return redirect('mainapp:home')

def notes(request):
    user=request.user
    notes=NotesOfUser.objects.all()
    obj=NotesOfUser.objects.get(author=request.user)
    
    form=UploadForm(request.POST or None,request.FILES or None,instance=obj)
    if request.method=='POST':
        if form.is_valid():
            form.save()

    return render(request,'mainapp/notes.html',{'notes':notes})

def screen(request):
    return render(request,'mainapp/screen.html')