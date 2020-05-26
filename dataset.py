import torch,os, math
import random
from PIL import Image
import torchvision.transforms as transforms
import csv

class RandomDataset(torch.utils.data.IterableDataset):

    def __init__(self, total):
        super(RandomDataset).__init__()
        assert total > 0, ' The total number must be greater than 0.'
        self.start = 0
        self.end = total

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator in a worker process
            iter_start = self.start
            iter_end = self.end
        else:
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        return iter([(torch.rand((3, 224, 224)), torch.randint(low=0, high=2, size=(1,)).squeeze(0).long()) for _ in range(iter_start, iter_end)])


class BalancedClusterDataset(torch.utils.data.IterableDataset):

    def __init__(self, path):
        super(BalancedClusterDataset).__init__()
        f_p = os.path.abspath(path)
        assert os.path.exists(f_p), "The directory does not exist"
        self.path = f_p

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        with open(self.path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            faces = [row for row in reader][1:]
            
        if worker_info is None:
            iter_start = 0
            iter_end = len(faces) - 1
        else:
            per_worker = int(math.ceil((len(faces) - 0) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(faces) - 1)

        return map(lambda x: format_row(x), faces[iter_start:iter_end])


def format_row(item):
    img = Image.open(item[-1])
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
    # transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    transform = transforms.ToTensor()
    # TODO: Should normalization be removed?
    # If needed for regression the negative label can be changed here instead of in the middle of the running code
    label = int(item[1])
    return transform(img), torch.tensor(label)