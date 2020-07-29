import pandas as pd


def create_datset_csv(preliminary_dataset, final_originals):

    pre = pd.read_csv(preliminary_dataset)
    final = pd.read_csv(final_originals)

    videos_filtered = final.video.unique().tolist()

    v = pre[(pre.label == 1) & (pre.original.isin(videos_filtered))]

    print(pre, v)

    res = pd.concat((final, v), ignore_index=True)

    print(pre.shape, res.shape)




if __name__ == '__main__':

    create_datset_csv('sample/sample/dataset.csv', 'sample/sample/finals_dataset.csv')