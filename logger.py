class AverageMeter(object):
  """
  Computes and stores the average and current value.
  """

  def __init__(self, precision=4):
    self.dph = "%.{}f".format(precision) # display placeholder
    self.epsilon = 1*10**(-precision)
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / (self.epsilon + self.count)

  def __str__(self):
    """String representation for logging"""
    # for values that should be recorded exactly e.g. iteration number
    if self.count == 0:
      return str(self.val)
    # for stats
    return '{} ({})'.format(self.dph, self.dph) % (self.val, self.avg)