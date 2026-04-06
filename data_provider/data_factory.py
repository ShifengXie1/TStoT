from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_M4, Dataset_Solar
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Weather': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Electricity': Dataset_Custom,
    'ILI': Dataset_Custom,
    'm4': Dataset_M4,
    'Solar': Dataset_Solar,
}

def data_provider(args, flag):
    # 1. 获取对应的数据集类
    Data = data_dict[args.data] 
    
    # 获取时间编码标志 (虽然视觉模型不用，但底层 Dataset 初始化还需要这个参数占位)
    timeenc = 0 if getattr(args, 'embed', 'timeF') != 'timeF' else 1

    # 2. 根据运行阶段 (train/val/test/pred) 设置 DataLoader 的控制参数
    if flag == 'test':
        shuffle_flag = False
        drop_last = True # 丢弃最后一个不完整的 batch，保证形状严格一致
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1 
        freq = getattr(args, 'detail_freq', args.freq)
        Data = Dataset_Pred # 预测模式下强制使用 Pred 数据集类
    else: # train 和 val
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size 
        freq = args.freq

    if args.data == 'm4':
        drop_last = False

    # 3. 实例化数据集 (Dataset)
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        # 【极其重要】：中间的 0 完美替换了报错的 args.label_len
        size=[args.seq_len, 0, args.pred_len], 
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        # 使用 getattr 防止外层 argparse 没传 seasonal_patterns 导致报错
        seasonal_patterns=getattr(args, 'seasonal_patterns', None) 
    )
    
    print(f"Loaded {flag} dataset: {len(data_set)} samples")
    
    # 4. 实例化数据加载器 (DataLoader)
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=getattr(args, 'num_workers', 0), # 默认用 0 单线程加载，最稳不会报错
        drop_last=drop_last
    )
    
    return data_set, data_loader