import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class rram_array:
    def __init__(self, w, b, Ap, An, a0p, a0n, a1p, a1n, tp, tn, Rinit, Rvar, dt):
        self.w = w
        self.b = b
        self.R = ((torch.rand(self.w, self.b) - 0.5) * Rvar + Rinit).to(device)  # initial the weights

        self.Ap = Ap
        self.An = An
        self.a0p = a0p
        self.a0n = a0n
        self.a1p = a1p
        self.a1n = a1n
        self.tp = tp
        self.tn = tn

        self.dt = dt

    def pulse(self, R_cur, V):
        v_pos = torch.where(V > 0, 1, 0)
        v_neg = torch.where(V <= 0, 1, 0)

        s_v_pos = v_pos * self.Ap * (torch.exp(v_pos * V / self.tp) - 1)
        s_v_neg = v_neg * self.An * (torch.exp(-1 * v_neg * V / self.tn) - 1)
        s_v = s_v_pos + s_v_neg

        r_pos = (self.a0p + self.a1p * V )* v_pos
        r_neg = (self.a0n + self.a1n * V )* v_neg
        r_diff_pos = r_pos - R_cur * v_pos
        r_diff_neg = R_cur * v_neg - r_neg
        r_diff = r_diff_pos + r_diff_neg

        stepF = (r_diff > 0).float()
        f_v = stepF * (r_diff ** 2)
        dR = s_v * f_v * self.dt
        R_new = dR + R_cur
        return R_new

    def read(self, Readout, Vread, Vpw, readnoise):
        if Readout:
            for i in range(int(Vpw / self.dt)):
                self.R = self.pulse(self.R, torch.ones_like(self.R) * Vread)

        return self.R * (1 + readnoise * 2 * (torch.rand(self.w, self.b) - 0.5)).to(device)

    def weight_update(self, R_cur, R_expected, pos_pulselist, neg_pulselist): #  pulselist shape: [option nums, param nums]
        R_diff = R_expected - R_cur
        R_diff_min = torch.abs(R_diff)
        R_diff_min_v_pos = torch.zeros(self.w, self.b).to(device)
        R_diff_min_pw_pos = torch.zeros(self.w, self.b).to(device)
        R_diff_min_v_neg = torch.zeros(self.w, self.b).to(device)
        R_diff_min_pw_neg = torch.zeros(self.w, self.b).to(device)
        R_diff_pos = torch.where(R_diff > 0, 1, 0)
        R_diff_neg = torch.where(R_diff <= 0, 1, 0)

        optionN_pos = pos_pulselist.shape[0]
        optionN_neg = neg_pulselist.shape[0]

        for i in range(optionN_pos):
            R_cal = R_cur * R_diff_pos
            for pulseNum in range(int(pos_pulselist[i, 1] / self.dt)):
                R_cal = self.pulse(R_cal, R_diff_pos * pos_pulselist[i, 0])
            R_diff = torch.abs(R_expected * R_diff_pos - R_cal)
            R_diff_min = torch.min(R_diff_min * R_diff_pos, R_diff)
            R_diff_min_update = (R_diff_min == R_diff).float() * R_diff_pos
            R_diff_min_keep = (R_diff_min != R_diff).float() * R_diff_pos
            R_diff_min_v_pos = pos_pulselist[i, 0] * R_diff_min_update + R_diff_min_v_pos * R_diff_min_keep
            R_diff_min_pw_pos = pos_pulselist[i, 1] * R_diff_min_update + R_diff_min_pw_pos * R_diff_min_keep

        R_diff = R_expected - R_cur
        R_diff_min = torch.abs(R_diff)
        for i in range(optionN_neg):
            R_cal = R_cur * R_diff_neg
            for pulseNum in range(int(neg_pulselist[i, 1] / self.dt)):
                R_cal = self.pulse(R_cal, R_diff_neg * neg_pulselist[i, 0])
            R_diff = torch.abs(R_expected * R_diff_neg  - R_cal)
            R_diff_min = torch.min(R_diff_min * R_diff_neg, R_diff)
            R_diff_min_update = (R_diff_min == R_diff).float() * R_diff_neg
            R_diff_min_keep = (R_diff_min != R_diff).float() * R_diff_neg
            R_diff_min_v_neg = neg_pulselist[i, 0] * R_diff_min_update + R_diff_min_v_neg * R_diff_min_keep
            R_diff_min_pw_neg = neg_pulselist[i, 1] * R_diff_min_update + R_diff_min_pw_neg * R_diff_min_keep

        R_diff_min_v = R_diff_min_v_pos + R_diff_min_v_neg
        R_diff_min_pw = R_diff_min_pw_pos + R_diff_min_pw_neg

        return R_diff_min_v, R_diff_min_pw

    def write(self, R_expected, pos_pulselist, neg_pulselist, MaxN, RTolerance, Readout, Vread, Vpw, readnoise):
        for i in range(MaxN):
            print('current R:', float(self.R[0, 0]))
            self.Read = self.read(Readout, Vread, Vpw, readnoise)
            print('measured R:', float(self.Read[0, 0]))
            update_v, update_pw = self.weight_update(self.Read, R_expected, pos_pulselist, neg_pulselist)
            update_enable = (torch.abs(R_expected - self.Read) / R_expected > RTolerance).float()
            update_pulseN = (update_pw / self.dt)
            for j in range(int(torch.max(update_pulseN * update_enable))):
                update_valid = (update_pulseN > 0).float()
                self.R = self.pulse(self.R, update_v * update_enable * update_valid)
                self.Read = self.read(Readout, Vread, Vpw, readnoise)
                update_pulseN -= 1
            print('final R:', float(self.R[0, 0]))
            print('final measured R:', float(self.Read[0, 0]))

def WtoRS(w, Rmax, Rmin):
    Cmin = 1 / Rmax # mapped to 0
    Cmax = 1 / Rmin # mapped to 1
    R = torch.reciprocal(w * (Cmax - Cmin) + Cmin)

    return R.to(device)

def RStoW(R, Rmax, Rmin):
    Cmin = 1 / Rmax # mapped to 0
    Cmax = 1 / Rmin # mapped to 1
    w = (torch.reciprocal(R) - Cmin) / (Cmax - Cmin)

    return w.to(device)
