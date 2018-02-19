from tqdm import tnrange

def step(model, crit, optim, data, train=True):
    model.train() if train else model.eval()
    optim.zero_grad()
    x, y = [V(d) for d in data]
    y_ = model(x)
    loss = crit(y_, y)
    if train:
        loss.backward()
        optim.step()
    
    return loss.data[0]

def train(epochs, dl_train, model, crit, optim, dl_val=None):
    result = {'loss_train': []}
    if dl_val: result['loss_val'] = []

    for epoch in tnrange(epochs):
        print(f'{epoch+1}/{epochs}:', end='')
        for k in result:
            train = k == 'loss_train'
            dl = dl_train if train else dl_val
            running_loss = 0.0
            for i, data in enumerate(dl):
                loss = step(model, crit, optim, data, train)
                running_loss += loss * data[0].size(0)
                
            result[k] += [running_loss / len(dl.dataset)]
            print(f' {"train" if train else "val"}({result[k][-1]:0.4f})', end='')
        print()
    return result

def plot_train(results, skip=0, fs=(7,4)):
    plt.figure(figsize=fs)
    for k in results: plt.plot(results[k][skip:], label=k)
    plt.legend()
    plt.show()

class LrFinder:
    def __init__(self, model, dl, optim, crit, start_lr=1e-5, end_lr=10, f=1):
        '''f: factor that slows the lr increase.'''
        self.model, self.dl, self.optim, self.crit = model, dl, optim, crit
        self.lr_log, self.losses, self.iterations = [], [], []
        self.lr = self.start_lr = start_lr
        nb = len(dl) * f
        self.lr_mult = (end_lr/start_lr)**(1/nb)
    
    def update_lr(self, iteration):
        new_lr = self.start_lr * (self.lr_mult**iteration)
        for g in self.optim.param_groups: g['lr'] = new_lr
        return new_lr
    
    def find(self):
        best = 1e9
        iteration = 0
        self.model.train()
        for data in self.dl:
            self.lr_log += [self.update_lr(iteration)]
            loss = step(self.model, self.crit, self.optim, data) / len(self.dl.dataset)
            self.losses += [loss]
            iteration += 1
            if math.isnan(loss) or loss>best*4: break
            if loss < best: best = loss
                
        self.best = best
    
    def plot(self, skips=2, xlim=None, ylim=None, fs=(12,4)):
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=fs)
        ax1.set_title('loss vs lr')
        ax1.set_ylabel('loss')
        ax1.set_xlabel('learning rate (log scale)')
        ax1.plot(self.lr_log[skips:], self.losses[skips:])
        ax1.set_xscale('log')
        if xlim: ax1.set_xlim(right=xlim)
        if ylim: ax1.set_ylim(top=ylim)
        ax2.set_title('learning rate')
        ax2.set_ylabel('lr')
        ax2.set_xlabel('steps')
        ax2.plot(self.lr_log)

def train_cosine(epochs, dl_train, model, crit, optim, sched_lens=None, dl_val=None):
    '''sched_mul can be a single int or a list'''
    n = len(dl_train.dataset)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n)
    result = {'loss_train': []}
    if dl_val: result['loss_val'] = []
    lr_history = []
    
    if sched_lens is None: sched_lens = 1
    if isinstance(sched_lens, int): sched_lens = [sched_lens]*epochs
    if len(sched_lens) < epochs: sched_lens += [sched_lens[-1]] * (epochs-len(sched_lens))

    for epoch in tnrange(epochs):
        print(f'{epoch+1}/{epochs}:', end='')
        for k in result:
            train = k == 'loss_train'
            dl = dl_train if train else dl_val
            running_loss = 0.0
            cycles = sched_lens[epoch] if train else 1
            sched.T_max = n * cycles
            for _ in range(cycles):
                for data in dl:
                    if train:
                        sched.step()
                        lr_history += sched.get_lr()
                    loss = step(model, crit, optim, data, train)
                    running_loss += loss * data[0].size(0)
                
            result[k] += [running_loss / len(dl.dataset) / cycles]
            print(f' {"train" if train else "val"}({result[k][-1]:0.4f})', end='')
        print()
        if sched:
            sched.last_epoch = -1
        
    return result, lr_history

def eval_classification(model, dl, name):
    model.eval()
    correct = 0
    for x,y in dl:
        y_ = model(V(x, volatile=True))
        correct += (y == torch.max(y_.data, 1)[1]).sum()
        
    acc = correct / len(dl.dataset)
    print(f'{name} accuracy: {acc:0.4f}')
    return acc