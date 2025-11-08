from torch.utils.tensorboard import SummaryWriter

class Logger:

    def __init__(self, period = 50, logDir = None):
        self.writer = SummaryWriter(logDir)
        self.scalars = {}
        self.images = {}
        self.period = period
        self.steps = 0

    def log(self, **kwargs):
        if 'scalars' in kwargs:
            for key in kwargs['scalars'].keys():
                self.scalars[key] = self.scalars.get(key, 0.) + kwargs['scalars'][key]
        if 'images' in kwargs:
            self.images = kwargs['images']

        self.steps += 1
        if self.steps % self.period == 0:
            for key in self.scalars.keys():
                self.writer.add_scalar(key, self.scalars[key] / self.period, self.steps)
            for key in self.images.keys():
                self.writer.add_image(key, self.images[key], self.steps)
            self.clear()

    def clear(self, clearSteps = False):
        self.scalars = {}
        self.images = {}
        if clearSteps:
            self.steps = 0