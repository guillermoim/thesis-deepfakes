from datasets.ff_dataset import get_loader, read_dataset, CompositeDataset

class AverageMeter(object):
    '''
        Code taken from torchreid repository
    '''

    def __init__(self):
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
        self.avg = self.sum / self.count if self.count > 0 else 0.

def read_training_dataset(data_dir, transform, normalization, filter = lambda x: 'original' in x or 'neural' in x, max_images_per_video=10, max_videos=10000, window_size=1, splits_path='ff_splits'):
    '''

     This function was taken from the TUM repository. In order to have control over the datasets (i.e. original, face2face,
     faceswaps, etc.). I have added the filter parameter that takes in a function to filter the keays. Default is all
     originals & NeuralTextures at all levels of compression (raw, c23, c40).

    '''
    datasets = read_dataset(data_dir, normalization = normalization, transform=transform, max_images_per_video=max_images_per_video, max_videos=max_videos,
                            window_size=window_size, splits_path=splits_path)

    # only neural textures and original
    datasets = {
        k: v for k, v in datasets.items()
        if filter(k)
    }
    print('Using training data: ')
    print('\n'.join(sorted(datasets.keys())))

    trains, vals, tests = [], [], []
    for data_dir_name, dataset in datasets.items():
        train, val, test = dataset
        # repeat original data multiple times to balance out training data
        compression = data_dir_name.split('_')[-1]
        num_tampered_with_same_compression = len({x for x in datasets.keys() if compression in x}) - 1
        count = 1 if 'original' not in data_dir_name else num_tampered_with_same_compression
        for _ in range(count):
            trains.append(train)
        vals.append(val)
        tests.append(test)

    return CompositeDataset(*trains), CompositeDataset(*vals), CompositeDataset(*tests)


