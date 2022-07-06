import torch
from inputs import inputs
from anntosnn import anntosnn
from snntraining import snntraining
import argparse
import distutils.util

def train(seed, split_ratio, min_freq, BATCH_SIZE, GloVe_name, GloVe_dim, directConversion,\
            N_EPOCHS, lr, T, thres, xbar, MaxN, RTolerance, Readout, Vread, Vpw, readnoise, w, b,\
            Ap, An, a0p, a0n, a1p, a1n, tp, tn, Rinit, Rvar, dt, Rmax, Rmin, \
            pos_pulselist, neg_pulselist):

    SEED = seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

# datasets and pre-trained word embeddings
    input = inputs()
    train_dataloader, valid_dataloader, test_dataloader = input.dataset(split_ratio, min_freq, BATCH_SIZE)
    glove_matrix = input.word_embeddings(GloVe_name, GloVe_dim)
    print('dataset and word embeddings ready!')
    output_dim = 1
# training
    if directConversion == True: #  conversion from a trained ann into a memristor-based snn
        offset = 25
        print('start conversion!')
        anntosnn(N_EPOCHS, SEED, offset, glove_matrix, output_dim, \
            lr, T, thres, xbar, MaxN, RTolerance, Readout, \
            Vread, Vpw, readnoise, w, b, Ap, An, a0p, a0n, a1p, a1n, tp, \
            tn, Rinit, Rvar, dt, Rmax, Rmin, pos_pulselist, neg_pulselist,\
            train_dataloader, valid_dataloader, test_dataloader)
    else: # directing training a memristor-based snn
        print('start training!')
        snntraining(N_EPOCHS, SEED, glove_matrix, output_dim, T, thres, lr, xbar, \
            MaxN, RTolerance, Readout, Vread, Vpw, readnoise, w, b, \
            Ap, An, a0p, a0n, a1p, a1n, tp, \
            tn, Rinit, Rvar, dt, Rmax, Rmin, pos_pulselist, neg_pulselist,\
            train_dataloader, valid_dataloader, test_dataloader)


if __name__=='__main__':
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()

# parameters for conversion from an ANN into an SNN:
    parser.add_argument('--seed', type=int, default=77, help='Random seed')
    parser.add_argument('--split_ratio', type=float, default=0.7, help='training/validation sets split ratio')
    parser.add_argument('--min_freq', type=int, default=10, help='minimum frequency of words in the vocabulary')
    parser.add_argument('--BATCH_SIZE', type=int, default=1, help='batch size')
    parser.add_argument('--GloVe_name', type=str, default='6B', help='GloVe name')
    parser.add_argument('--GloVe_dim', type=int, default=100, help='GloVe dimensions')
    parser.add_argument('--directConversion', type=lambda x:bool(distutils.util.strtobool(x)), default=True, help='conversion of an ANN into an SNN')
    parser.add_argument('--N_EPOCHS', type=int, default=5, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--T', type=int, default=1000, help='spike train length for representing a single continuous value')
    parser.add_argument('--thres', type=float, default=50, help='threshold')
    parser.add_argument('--xbar', type=lambda x:bool(distutils.util.strtobool(x)), default=True, help='enable virtual memristor arrays')
    parser.add_argument('--MaxN', type=int, default=5, help='maximum iteration times in weight updating')
    parser.add_argument('--RTolerance', type=float, default=0.005, help='R tolerance')
    parser.add_argument('--Readout', type=lambda x:bool(distutils.util.strtobool(x)), default=False, help='enable read-out bias voltage')
    parser.add_argument('--Vread', type=float, default=0.5, help='read-out bias voltage magnitude')
    parser.add_argument('--Vpw', type=float, default=1e-6, help='read-out bias voltage pulse-width')
    parser.add_argument('--readnoise', type=float, default=0, help='readnoise')
    parser.add_argument('--w', type=int, default=10, help='number of wordlines in a virtual memristor array')
    parser.add_argument('--b', type=int, default=10, help='number of bitlines in a virtual memristor array')
    parser.add_argument('--Ap', type=float, default=0.21388644421061628, help='memristor parameter')
    parser.add_argument('--An', type=float, default=-0.813018367268805, help='memristor parameter')
    parser.add_argument('--a0p', type=float, default=37086.67218413958, help='memristor parameter')
    parser.add_argument('--a0n', type=float, default=43430.02023698205, help='memristor parameter')
    parser.add_argument('--a1p', type=float, default=-20193.23957579438, help='memristor parameter')
    parser.add_argument('--a1n', type=float, default=34332.85303661032, help='memristor parameter')
    parser.add_argument('--tp', type=float, default=1.6590989889370842, help='memristor parameter')
    parser.add_argument('--tn', type=float, default=1.5148294827972748, help='memristor parameter')
    parser.add_argument('--Rinit', type=int, default=11000, help='memristor initial resistance')
    parser.add_argument('--Rvar', type=int, default=8000, help='memristor initial resistance variation')
    parser.add_argument('--dt', type=float, default=1e-6, help='time step')
    parser.add_argument('--Rmax', type=int, default=18900, help='memristor resistance upper boundary')
    parser.add_argument('--Rmin', type=int, default=2200, help='memristor resistance lower boundary')

    args = parser.parse_args()
# pulse options hardwired here. Please change the values according to the R tolerance you choose.
    if args.directConversion == True:
        pos_pulselist = torch.FloatTensor([[0.9, 0.9, 0.9, 0.9, 0.9, 0.9], [1e-6, 2e-6, 1e-5, 2e-5, 5e-5, 1e-4]]).t().to(device)
        neg_pulselist = torch.FloatTensor([[-1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2, -1.2], [1e-6, 2e-6, 1e-5, 2e-5, 1e-4, 1e-3, 2e-3, 5e-3]]).t().to(device)
    else:
        pos_pulselist = torch.FloatTensor([[0.9, 0.9, 0.9, 0.9, 0.9], [1e-6, 2e-6, 1e-5, 2e-5, 5e-5]]).t().to(device)
        neg_pulselist = torch.FloatTensor([[-1.2, -1.2, -1.2, -1.2, -1.2], [1e-6, 2e-6, 1e-5, 2e-5, 1e-4]]).t().to(device)

    train(seed = args.seed,
            split_ratio = args.split_ratio,
            min_freq = args.min_freq,
            BATCH_SIZE = args.BATCH_SIZE,
            GloVe_name = args.GloVe_name,
            GloVe_dim = args.GloVe_dim,
            directConversion = args.directConversion,
            N_EPOCHS = args.N_EPOCHS,
            lr = args.lr,
            T = args.T,
            thres = args.thres,
            xbar = args.xbar,
            MaxN = args.MaxN,
            RTolerance = args.RTolerance,
            Readout = args.Readout,
            Vread = args.Vread,
            Vpw = args.Vpw,
            readnoise = args.readnoise,
            w = args.w,
            b = args.b,
            Ap = args.Ap,
            An = args.An,
            a0p = args.a0p,
            a0n = args.a0n,
            a1p = args.a1p,
            a1n = args.a1n,
            tp = args.tp,
            tn = args.tn,
            Rinit = args.Rinit,
            Rvar = args.Rvar,
            dt = args.dt,
            Rmax = args.Rmax,
            Rmin = args.Rmin,
            pos_pulselist = pos_pulselist,
            neg_pulselist = neg_pulselist)
