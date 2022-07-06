import torch
import torch.nn as nn
from torch import nn
from rram_array import rram_array, WtoRS, RStoW
import torch.nn.functional as F
import time
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WordAVGModel(nn.Module):
    def __init__(self, offset, wordemb_matrix, output_dim):
        super().__init__()
        self.device = device
        self.vocab_size, self.embedding_dim = wordemb_matrix.shape[0], wordemb_matrix.shape[1]
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding.load_state_dict({'weight': wordemb_matrix})
        self.output_dim = output_dim
        self.fc = nn.Linear(self.embedding_dim, self.output_dim, bias = False)
        torch.nn.init.uniform_(self.fc.weight).to(self.device)
        self.offset = offset

    def forward(self, text):
        self.fc.weight.data.clamp_(0, 1)
        self.embedding.weight.data.clamp_(0, 1)
        self.embedded = self.embedding(text) # [sent len, batch size, emb dim]
        self.pooled = F.avg_pool2d(self.embedded, (self.embedded.shape[1], 1)).squeeze(1) # [batch size, embedding_dim]

        return self.fc(self.pooled).squeeze(1) - self.offset

class ANNtoSNNModel(nn.Module):
    def __init__(self, fc_weight, emb_weight, output_dim, T, thres, xbar, MaxN, RTolerance, \
                    Readout, Vread, Vpw, readnoise, w, b, Ap, An, a0p, a0n, a1p, a1n, \
                    tp, tn, Rinit, Rvar, dt, Rmax, Rmin, pos_pulselist, neg_pulselist):
        super().__init__()
        self.device = device
        self.vocab_size, self.embedding_dim = emb_weight.shape[0], emb_weight.shape[1]
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding.load_state_dict({'weight': emb_weight})
        self.embedding.weight.requires_grad = False
        self.output_dim = output_dim
        self.fc_weight = fc_weight
        self.fc = nn.Linear(self.embedding_dim, self.output_dim, bias = False)
        self.fc.load_state_dict({'weight': self.fc_weight})
        self.fc.weight.requires_grad = False
        self.T = T
        self.xbar = xbar
        self.thres= thres
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

        if self.xbar:
            self.rramArray = rram_array(self.w, self.b, self.Ap, self.An, self.a0p, self.a0n, self.a1p, \
                                        self.a1n, self.tp, self.tn, self.Rinit, self.Rvar, self.dt)
            self.memristorRS_expected = WtoRS(fc_weight.reshape(self.w, self.b), self.Rmax, self.Rmin)
            self.rramArray.write(self.memristorRS_expected, self.pos_pulselist, self.neg_pulselist, \
                                    MaxN = self.MaxN, RTolerance = self.RTolerance, Readout = self.readout, \
                                    Vread = self.Vread, Vpw = self.Vpw, readnoise = self.readnoise)
            self.memristorRS = self.rramArray.read(Readout = self.readout, Vread = self.Vread, \
                                                    Vpw = self.Vpw, readnoise = self.readnoise)
            self.memristorWeight = RStoW(self.memristorRS.flatten(), self.Rmax, self.Rmin).reshape(-1, fc_weight.shape[1])
            self.memristorWeight.clamp_(0, 1)

    def initVariables(self, batch_size, output_dim, TLen):
        self.membraneV = torch.zeros(batch_size, output_dim, TLen).to(device)
        self.spikes = torch.zeros(batch_size, output_dim, TLen).to(device)

    def forward(self, text):
        with torch.no_grad():
            self.embedded = self.embedding(text) # [sent len, batch size, emb dim]
        self.pooled = F.avg_pool2d(self.embedded, (self.embedded.shape[1], 1)).squeeze(1) # [batch size, embedding_dim]

        self.batch_size, self.emb_size = self.pooled.shape[0], self.pooled.shape[1]
        self.rand_matrix = torch.rand(self.batch_size, self.emb_size, self.T).to(device) # [batch size, embedding_dim, T]

        self.inputSpike = (self.pooled.unsqueeze(2).expand(-1, -1, self.T) > self.rand_matrix).float() # [batch size, embedding_dim, T]
        self.initVariables(self.batch_size, self.output_dim, self.T)

        if self.xbar:
            for t in range(self.T):
                self.memristorRS = self.rramArray.read(Readout = self.readout, Vread = self.Vread, \
                                                        Vpw = self.Vpw, readnoise = self.readnoise)
                self.memristorWeight = RStoW(self.memristorRS.flatten(), self.Rmax, self.Rmin).reshape(-1, self.fc_weight.shape[1])
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
                    self.membraneV[:, :, t] = self.membraneV[:, :, t-1]  - self.spikes[:, :, t-1] * self.thres +\
                                                torch.mm(self.inputSpike[:, :, t], self.fc.weight.t())
                self.spikes = (self.membraneV > self.thres).float()

        self.spikeRate = torch.sum(self.spikes, dim = 2) / self.T

        return self.spikeRate.squeeze(1)

def anninit(seed, offset, wordemb_matrix, OUTPUT_DIM, lr):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = WordAVGModel(offset, wordemb_matrix, OUTPUT_DIM).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    return optimizer, criterion, model

def anntosnninit(seed, fc_weight, emb_weight, output_dim, T, thres, xbar, MaxN, RTolerance, \
        Readout, Vread, Vpw, readnoise, w, b, Ap, An, a0p, a0n, a1p, a1n, tp, tn, Rinit,\
        Rvar, dt, Rmax, Rmin, pos_pulselist, neg_pulselist):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    modelANNtoSNN = ANNtoSNNModel(fc_weight, emb_weight, output_dim, T, thres, \
                            True, MaxN, RTolerance, Readout, Vread, Vpw, readnoise, w,\
                             b, Ap, An, a0p, a0n, a1p, a1n, tp, tn, Rinit, Rvar, \
                             dt, Rmax, Rmin, pos_pulselist, neg_pulselist)
    criterionSNN = nn.BCELoss()

    return modelANNtoSNN, criterionSNN

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum()/len(correct)
    return acc

def binary_accuracySNN(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    acc = correct.sum()/len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for _, (label, text) in enumerate(iterator):
        optimizer.zero_grad()
        predictions = model(text)

        loss = criterion(predictions, label.float())
        acc = binary_accuracy(predictions, label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for _, (label, text) in enumerate(iterator):
            predictions = model(text)
            loss = criterion(predictions, label.float())
            acc = binary_accuracy(predictions, label)
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

def anntrain(N_EPOCHS, train_dataloader, valid_dataloader, optimizer, criterion, model):

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_dataloader, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'wordavg-model.pt')

        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    return model.fc.weight, model.embedding.weight

def anntest(model, test_dataloader, criterion):
    start_time = time.time()
    model.load_state_dict(torch.load('wordavg-model.pt'))
    test_loss, test_acc = evaluate(model, test_dataloader, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Time: {epoch_mins}m {epoch_secs}s')
    print(f'\t test. Loss: {test_loss:.3f} |  test. Acc: {test_acc*100:.2f}%')

def snntest(test_dataloader, modelANNtoSNN, criterionSNN):
    start_time = time.time()
    test_loss, test_acc = evaSNN(modelANNtoSNN, test_dataloader, criterionSNN)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(epoch_mins, epoch_secs)
    print(f'\t Val. Loss: {test_loss:.3f} |  Val. Acc: {test_acc*100:.2f}%')

def anntosnn(N_EPOCHS, seed, offset, wordemb_matrix, output_dim, \
                lr, T, thres, xbar, MaxN, RTolerance, Readout, Vread, Vpw, \
                readnoise, w, b, Ap, An, a0p, a0n, a1p, a1n, tp, \
                tn, Rinit, Rvar, dt, Rmax, Rmin, pos_pulselist, neg_pulselist,\
                train_dataloader, valid_dataloader, test_dataloader):
    optimizer, criterion, model = anninit(seed, offset, wordemb_matrix, output_dim, lr)
    print('ann initialised!')
    fc_weight, emb_weight= anntrain(N_EPOCHS, train_dataloader, valid_dataloader, optimizer, criterion, model)
    print('ann trained!')
    anntest(model, test_dataloader, criterion)
    print('ann tested!')
    modelANNtoSNN, criterionSNN = anntosnninit(seed, fc_weight, emb_weight, output_dim, T, thres, xbar, MaxN, RTolerance, Readout, \
                                    Vread, Vpw, readnoise, w, b, Ap, An, a0p, a0n, a1p, a1n, tp, tn, Rinit, Rvar,\
                                    dt, Rmax, Rmin, pos_pulselist, neg_pulselist)
    print('snn initialised!')
    snntest(test_dataloader, modelANNtoSNN, criterionSNN)
    print('snn tested!')
