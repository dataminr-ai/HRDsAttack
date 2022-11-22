from torch.utils.data import Dataset


class T5Dataset(Dataset):
    """
    Customized Pytorch Dataset class
    """

    def __init__(self, all_input_ids, all_mask_ids,
                 all_target_ids, all_feature_idx):
        """
        :param all_input_ids: input token ids
        :param all_mask_ids:  input mask ids
        :param all_target_ids: output token ids
        :param all_feature_idx: index of the data point
        """
        self.all_input_ids = all_input_ids
        self.all_mask_ids = all_mask_ids
        self.all_target_ids = all_target_ids
        self.all_feature_idx = all_feature_idx

    def __getitem__(self, idx):
        return self.all_input_ids[idx], self.all_mask_ids[idx], \
               self.all_target_ids[idx], self.all_feature_idx[idx]

    def __len__(self):
        return len(self.all_input_ids)
