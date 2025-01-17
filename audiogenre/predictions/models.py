# predictions/models.py
from django.db import models

class AudioFile(models.Model):
    file = models.FileField(upload_to='audio/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class SongRecommendation(models.Model):
    title = models.CharField(max_length=200)
    artist = models.CharField(max_length=200)
    genre = models.CharField(max_length=100)
    youtube_link = models.URLField(max_length=200, blank=True, null=True)


    def __str__(self):
        return f"{self.title} by {self.artist} - {self.youtube_link}"
