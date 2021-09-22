from src.augmentations.inference_scheme import get_inference_scheme

def generate_validation_transform(args, do_prefetch=True):
    validation_transform = get_inference_scheme(input_size=args.input_size,
                                                transform_type=args.transform_type, is_prefetch=do_prefetch)
    return validation_transform
