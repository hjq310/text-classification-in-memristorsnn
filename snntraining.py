import torch
import torch.nn as nn
from torch import nn
import torch.nn.functional as F
import time
from rram_array import rram_array, WtoRS, RStoW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SNN(nn.Module):
    def __init__(self, wordemb_matrix, output_dim, T, thres, lr, xbar, MaxN, RTolerance, Readout, Vread, Vpw, \
                    readnoise, w, b, Ap, An, a0p, a0n, a1p, a1n, tp, tn, Rinit, Rvar, dt, Rmax, Rmin, \
                    pos_pulselist, neg_pulselist):

        super().__init__()
        self.device = device
        self.vocab_size, self.embedding_dim = wordemb_matrix.shape[0], wordemb_matrix.shape[1]
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding.load_state_dict({'weight': wordemb_matrix})
        self.embedding.weight.requires_grad = False
        self.output_dim = output_dim
        self.fc = nn.Linear(self.embedding_dim, self.output_dim, bias = False)
        self.fc.weight.requires_grad = False
        self.T = T
        self.lr = lr
        self.fc_s = 0
        self.emb_s = 0
        self.ep = 1e-10
        self.thres= thres
        self.xbar = xbar
        self.MaxN = MaxN
        self.RTolerance = RTolerance
        self.readout = Readout
        self.Vread = Vread
        self.Vpw = Vpw
        self.readnoise = readnoise
        self.w = w
        self.b = b
        self.Ap = Ap
        self.An = An
        self.a0p = a0p
        self.a0n = a0n
        self.a1p = a1p
        self.a1n = a1n
        self.tp = tp
        self.tn = tn
        self.Rinit = Rinit
        self.Rvar = Rvar
        self.dt = dt
        self.Rmax = Rmax
        self.Rmin = Rmin
        self.pos_pulselist = pos_pulselist
        self.neg_pulselist = neg_pulselist

        self.rramArray = rram_array(self.w, self.b, self.Ap, self.An, self.a0p, self.a0n, self.a1p, self.a1n, \
                                        self.tp, self.tn, self.Rinit, self.Rvar, self.dt)
        self.memristorRS = self.rramArray.read(Readout = self.readout, Vread = self.Vread, \
                                                Vpw = self.Vpw, readnoise = self.readnoise)
        self.memristorWeight = RStoW(self.memristorRS.flatten(), self.Rmax, self.Rmin).reshape(-1, self.fc.weight.shape[1])
        self.memristorWeight.clamp_(0, 1)
        #torch.nn.init.uniform_(self.memristorWeight).to(self.device)
        self.fc.load_state_dict({'weight': self.memristorWeight})

    def initVariables(self, batch_size, output_dim, TLen):
        self.membraneV = torch.zeros(batch_size, output_dim, TLen).to(device)
        self.spikes = torch.zeros(batch_size, output_dim, TLen).to(device)

    def forward(self, text):
        self.text = text
        self.batch_size, self.sentLen = self.text.shape[0], self.text.shape[1]
        self.embedded = self.embedding(self.text) # [batch size, sent len, emb dim]
        self.pooled = F.avg_pool2d(self.embedded, (self.embedded.shape[1], 1)).squeeze(1) # [batch size, embedding_dim]

        self.rand_matrix = torch.rand(self.batch_size, self.embedding_dim, self.T).to(device) # [batch size, embedding_dim, T]
        self.inputSpike = (self.pooled.unsqueeze(2).expand(-1, -1, self.T) > self.rand_matrix).float() # [batch size, embedding_dim, T]
        self.initVariables(self.batch_size, self.output_dim, self.T)

        if self.xbar:
            for t in range(self.T):
                self.memristorRS = self.rramArray.read(Readout = self.readout, Vread = self.Vread, \
                                                        Vpw = self.Vpw, readnoise = self.readnoise)
                self.memristorWeight = RStoW(self.memristorRS.flatten(), self.Rmax, self.Rmin).reshape(-1, self.fc.weight.shape[1])
                self.memristorWeight.clamp_(0, 1)
                if t == 0:
                    self.membraneV[:, :, t] = torch.mm(self.inputSpike[:, :, t], self.memristorWeight.t())
                else:
                    self.membraneV[:, :, t] = self.membraneV[:, :, t-1]  - self.spikes[:, :, t-1] * self.thres + \
                                                    torch.mm(self.inputSpike[:, :, t], self.memristorWeight.t())
                self.spikes = (self.membraneV > self.thres).float()
        else:
            for t in range(self.T):
                if t == 0:
                    self.membraneV[:, :, t] = torch.mm(self.inputSpike[:, :, t], self.fc.weight.t())
                else:
                    self.membraneV[:, :, t] = self.membraneV[:, :, t-1]  - self.spikes[:, :, t-1] * self.thres + \
                                                torch.mm(self.inputSpike[:, :, t], self.fc.weight.t())
                self.spikes = (self.membraneV > self.thres).float()

        self.spikeRate = torch.sum(self.spikes, dim = 2) / self.T
        self.result = self.embedding_dim * (self.spikeRate.squeeze(1) - 0.5) / 2

        return self.result

    def plast(self, label):
        self.label = label
        self.delta = (torch.sigmoid(self.result) - label).unsqueeze(1) / self.batch_size

        self.fc_grad = torch.sum(torch.bmm(self.delta.unsqueeze(2), self.pooled.unsqueeze(1)), dim=0)   # [emb_dim, output_dim]
        self.fc_s += self.fc_grad ** 2

        if self.xbar:
            self.emb_grad = torch.mm(self.delta, self.memristorWeight)
            self.memristorWeight_expected = self.memristorWeight - self.lr * self.fc_grad / (self.fc_s ** 0.5  + self.ep)
            self.memristorRS_expected = WtoRS(self.memristorWeight_expected, self.Rmax, self.Rmin)
            print('expected R:', float(self.memristorRS_expected[0, 0]))
            self.rramArray.write(self.memristorRS_expected.reshape(self.w, self.b), self.pos_pulselist, self.neg_pulselist, \
                                    MaxN = self.MaxN, RTolerance = self.RTolerance, Readout = self.readout, Vread = self.Vread, \
                                    Vpw = self.Vpw, readnoise = self.readnoise)
            print('updated R:', float(self.rramArray.R[0, 0]))
            print('R diff:', float(self.rramArray.R[0, 0]) - float(self.memristorRS_expected[0, 0]))
        else:
            self.emb_grad = torch.mm(self.delta, self.fc.weight)
            self.fc.weight.data = self.fc.weight.data - self.lr * self.fc_grad / (self.fc_s ** 0.5  + self.ep)
            self.fc.weight.data.clamp_(0, 1)

        self.input = torch.bincount(self.text.flatten(), minlength = self.vocab_size).float() / (self.sentLen * self.batch_size)
        self.emb_grad = torch.sum(torch.mm(self.emb_grad.reshape(-1, 1), self.input.reshape(1, -1)).reshape(self.batch_size, self.embedding_dim, -1), dim = 0).t()

        self.emb_s += self.emb_grad ** 2
        self.embedding.weight.data = self.embedding.weight.data - self.lr * self.emb_grad / (self.emb_s ** 0.5  + self.ep)
        self.embedding.weight.data.clamp_(0, 1)

def network_init(seed, wordemb_matrix, output_dim, T, thres, lr, xbar, MaxN, RTolerance, Readout, Vread, Vpw, readnoise, \
                    w, b, Ap, An, a0p, a0n, a1p, a1n, tp, tn, Rinit, Rvar, dt, Rmax, Rmin, pos_pulselist, neg_pulselist):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    modelSNN = SNN(wordemb_matrix, output_dim, T, thres, lr, xbar, MaxN, RTolerance, Readout, Vread, Vpw, readnoise, w, b, \
                    Ap, An, a0p, a0n, a1p, a1n, tp, tn, Rinit, Rvar, dt, Rmax, Rmin, pos_pulselist, neg_pulselist)
    print('initial weight:', float(modelSNN.memristorWeight[0, 0]))
    modelSNN = modelSNN.to(device)
    criterionSNN = nn.BCEWithLogitsLoss()

    return modelSNN, criterionSNN

def binary_accuracySNN(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum()/len(correct)
    return acc

def trainSNN(model, iterator, criterionSNN):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for _, (label, text) in enumerate(iterator):
            predictions = model(text)
            model.plast(label)

            loss = criterionSNN(predictions, label.float())
            acc = binary_accuracySNN(predictions, label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaSNN(model, iterator, criterionSNN):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for _, (label, text) in enumerate(iterator):
            predictions = model(text)
            loss = criterionSNN(predictions, label.float())
            acc = binary_accuracySNN(predictions, label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def snntrain(N_EPOCHS, train_dataloader, valid_dataloader, modelSNN, criterionSNN):
    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = trainSNN(modelSNN, train_dataloader, criterionSNN)
        valid_loss, valid_acc = evaSNN(modelSNN, valid_dataloader, criterionSNN)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(modelSNN.state_dict(), 'best-snntrainingmodel.pt')

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\t Val. Loss: {train_loss:.3f} |  Train. Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

def snntest(test_dataloader, modelSNN, criterionSNN):
    start_time = time.time()
    modelSNN.load_state_dict(torch.load('best-snntrainingmodel.pt'))
    test_loss, test_acc = evaSNN(modelSNN, test_dataloader, criterionSNN)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Time: {epoch_mins}m {epoch_secs}s')
    print(f'\t test. Loss: {test_loss:.3f} |  test. Acc: {test_acc*100:.2f}%')

def snntraining(N_EPOCHS, seed, wordemb_matrix, output_dim, T, thres, lr, xbar, \
                    MaxN, RTolerance, Readout, Vread, Vpw, readnoise, w, b, Ap, An, a0p, a0n, \
                    a1p, a1n, tp, tn, Rinit, Rvar, dt, Rmax, Rmin, pos_pulselist, neg_pulselist,\
                    train_dataloader, valid_dataloader, test_dataloader):

    modelSNN, criterionSNN = network_init(seed, wordemb_matrix, output_dim, T, thres, lr, \
                                            xbar, MaxN, RTolerance, Readout, Vread, Vpw, readnoise, \
                                            w, b, Ap, An, a0p, a0n, a1p, a1n, tp, tn, Rinit, Rvar,\
                                            dt, Rmax, Rmin, pos_pulselist, neg_pulselist)
    print('snn initialised!')
    snntrain(N_EPOCHS, train_dataloader, valid_dataloader, modelSNN, criterionSNN)
    print('snn trained!')
    snntest(test_dataloader, modelSNN, criterionSNN)
    print('snn tested!')
