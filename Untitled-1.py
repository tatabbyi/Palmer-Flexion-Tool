

Class HeartBeat(object):
    id = Model.UUIDField(primary_key=True);
    timestamp = Model.DateTimeField(auto_now_add=True);
    value = Model.IntegerField();
    previous_value = Model.IntegerField(null=True, blank=True);
    patient = Model.ForeignKey(patient, on_delete=Model.CASCADE);
    device = Model.ForeignKey(device, on_delete=Model.CASCADE);

    def __str__(self):
        return f"HeartBeat {self.id} at {self.timestamp} with value {self.value}"


    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        if self.value < 40 or self.value > 160:
            raise Warning("Abnormal heart rate detected!")
        else:
            return super().save(*args, **kwargs)
        
    def update_value(self, new_value):
        self.previous_value = self.value
        self.value = new_value
        self.save()