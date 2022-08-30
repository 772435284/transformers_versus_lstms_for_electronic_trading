'''
data_factory.py
Based on: https://github.com/thuml/Autoformer/blob/main/data_provider/data_factory.py
'''
from data_provider.data_loader import Dataset_Custom
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
}


def data_provider(args, flag, all_data):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
        scale = True
    elif flag == 'backtest':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        scale = False
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
        scale = True

    data_set = Data(
        all_data = all_data,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale = scale,
        timeenc=timeenc,
        freq=freq,
        label = args.horizon,
        product = args.product
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
