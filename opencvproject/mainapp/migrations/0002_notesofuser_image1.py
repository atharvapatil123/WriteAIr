# Generated by Django 3.2.8 on 2021-10-30 14:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mainapp', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='notesofuser',
            name='image1',
            field=models.ImageField(blank=True, upload_to=''),
        ),
    ]
