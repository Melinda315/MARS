import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():

    parser = argparse.ArgumentParser(description="Run Recommender")

    parser.add_argument('--recommender', nargs='?', default='MARS', help='Choose a recommender.', required=False)
    parser.add_argument('--dataset', nargs='?', default='ciao', help='Choose a dataset.', required=False)
    parser.add_argument('--lRate', type=float, default=0.01, help='Learning rate.', required=False)
    parser.add_argument('--mode', nargs='?', help='Validation or Test (Val, Test)', required=False)
    parser.add_argument('--early_stop', type=int, default=50, help='Early stop iteration.')
    parser.add_argument('--topK', nargs='?', default='[5,10,20]', help="topK")
    parser.add_argument('--numEpoch', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--num_negatives', type=int, default=100, help='Number of negative samples.')
    parser.add_argument('--margin', type=float, default=0.5, help='Margin.')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size.')
    parser.add_argument('--batchSize_test', type=int, default=2000, help='Batch size for test.')
    parser.add_argument('--cuda', type=int, default=0, help='Speficy GPU number')
    parser.add_argument('--pop_reg', type=float, default=0.75, help='Popularity Regularization')
    parser.add_argument('--reg1', type=float, default=0.1, help='Distance Regularization.')
    parser.add_argument('--reg2', type=float, default=0.01, help='Neighborhood Regularization.')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Number of embedding dimensions.')
    parser.add_argument('--k', type=int, default=4, help='Number of projections.')
    parser.add_argument('--rand_seed', type=int, default=34567, help='Random seed.')
    parser.add_argument('--eta', type=float, default=0.1, help='eta')
    return parser.parse_known_args()

def printConfig(args):
    common_elems = ['recommender', 'dataset', 'numEpoch', 'lRate', 'num_negatives', 'embedding_dim', 'early_stop', 'batch_size', 'reg1', 'reg2', 'rand_seed', 'margin', 'k', 'pop_reg']

    rec = args.recommender

    st = []
    for elem in common_elems:
        s = str(elem + ": " + str(getattr(args, elem)))
        st.append(s)
    print(st)

if __name__ == '__main__':
    args, unknown = parse_args()
    printConfig(args)

    from models.model import MARS
    recommender = MARS(args)
    recommender.training()
