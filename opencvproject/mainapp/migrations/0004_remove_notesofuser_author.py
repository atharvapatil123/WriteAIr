# Generated by Django 3.2.8 on 2021-10-31 10:24

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('mainapp', '0003_alter_notesofuser_image1'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='notesofuser',
            name='author',
        ),
    ]
