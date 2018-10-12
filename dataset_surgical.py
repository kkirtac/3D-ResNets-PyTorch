from datasets.surgical import SurgicalDataset


phase_list = ['Preparation',
              'CalotTriangleDissection',
              'ClippingCutting',
              'GallbladderDissection',
              'GallbladderPackaging',
              'CleaningCoagulation',
              'GallbladderRetraction']

def get_training_set(opt, spatial_transform, temporal_transform, target_transform):

    assert opt.dataset in ['surgical']

    training_data = SurgicalDataset(
            opt.root_path,
            phase_list,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)

    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform, target_transform):

    assert opt.dataset in ['surgical']

    validation_data = SurgicalDataset(
            opt.root_path,
            phase_list,
            'validation',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)

    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):

    assert opt.dataset in ['surgical']

    test_data = SurgicalDataset(
            opt.root_path,
            phase_list,
            'testing',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=opt.sample_duration)

    return test_data
