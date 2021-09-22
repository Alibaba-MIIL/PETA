import torch
import argparse
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pickle
from src.models import create_model
from src.utils.utils import create_dataloader, validate
from src.plot_results import plot_results
from src.loss_functions.asymmetric_loss import AsymmetricLoss


# ----------------------------------------------------------------------
# Parameters
# parser = argparse.ArgumentParser(description='Photo album Event recognition using Transformers Attention.')
parser = argparse.ArgumentParser(description='PyTorch PETA ML-CUFED Inference')
parser.add_argument('--model_path', type=str, default='./models_local/peta_32.pth')
# parser.add_argument('--album_path', type=str, default='./albums/Graduation/0_92024390@N00')
parser.add_argument('--val_dir', type=str, default='./albums') #  /Graduation') # /0_92024390@N00')
parser.add_argument('--num_classes', type=int, default=23)
parser.add_argument('--classes_path', type=str, default='./data/ML_CUFED/classes.pickle')
parser.add_argument('--model_name', type=str, default='mtresnetaggregate')
parser.add_argument('--transformers_pos', type=int, default=1)
# parser.add_argument('--num_rows', type=int, default=7)
# parser.add_argument('--pretrain-backbone', type=int, default=0)
# parser.add_argument('--wordvec_dim', type=int, default=300)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--transform_type', type=str, default='squish')
parser.add_argument('--album_sample', type=str, default='rand_permute')
parser.add_argument('--dataset_path', type=str, default='./data/ML_CUFED')
parser.add_argument('--dataset_type', type=str, default='ML_CUFED')
parser.add_argument('--path_output', type=str, default='./outputs')
parser.add_argument('--use_transformer', type=int, default=1)
parser.add_argument('--album_clip_length', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--top_k', type=int, default=3)
parser.add_argument('--remove_model_jit', type=int, default=None)


def main():
    print('PETA demo of inference code on a single album.')

    # ----------------------------------------------------------------------
    # Preliminaries
    args = parser.parse_args()


    # setup model
    print('creating and loading the model...')
    state = torch.load(args.model_path, map_location='cpu')
    # args.num_classes = state['num_classes']
    model = create_model(args).cuda()
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    # classes_list = np.array(list(state['idx_to_class'].values()))
    print('done\n')

    # setup data loader
    print('creating data loader...')
    val_loader = create_dataloader(args)
    print('done\n')

    # actual validation process
    print('loading album and doing inference...')
    map = validate(model, val_loader)
    print("final validation map: {:.2f}".format(map))

    # setup data
    # with open("data/NUS_WIDE/classes.pickle", 'rb') as fp:
    #     classes = pickle.load(fp)
    # with open(os.path.join(args.dataset_path, "classes.pickle"), 'rb') as fp:
    #     classes = pickle.load(fp)

    # ----------------------------------------------------------------------
    # Inference
    # print('loading album and doing inference...')
    # imgs_path = os.listdir(args.album_path)


    # im = Image.open(args.album_path)
    # im_resize = im.resize((args.input_size, args.input_size))
    # np_img = np.array(im_resize, dtype=np.uint8)
    # tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
    # tensor_batch = torch.unsqueeze(tensor_img, 0).cuda()
    # output = model(tensor_batch)
    # np_output = output.cpu().detach().numpy()
    # print('done\n')

    '''
    # Extract predicted tags
    x = np_output.reshape(args.wordvec_dim, args.num_rows)
    # p = np.dot(x.T, wordvec_array)
    dist = -np.max(p, axis=0)
    sorted_ids = np.argsort(dist)
    top_k_ids = sorted_ids[:args.top_k]
    pred_classes = [classes[i] for i in top_k_ids]

    # Visualize results
    img_result = plot_results(im_resize, pred_classes, classes, unseen_classes, args.path_output,
                              os.path.split(args.album_path)[1])

    # ----------------------------------------------------------------------
    # Example of loss calculation
    output = model(tensor_batch)
    loss_func = AsymmetricLoss(args)
    batch_siz = 1
    target = torch.zeros(batch_siz, args.num_classes)
    target[0, 40] = 1
    loss = loss_func.forward(output.cpu(), target.cpu())
    print("Example loss: %0.2f" % loss)
    '''
    print('Done\n')


if __name__ == '__main__':
    main()
