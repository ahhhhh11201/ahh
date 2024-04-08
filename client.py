import models, torch, copy


class Client(object):

    def __init__(self, conf, model,train_dataset, id=-1):

        self.conf = conf

        self.local_model = models.get_model(self.conf["model_name"])

        self.client_id = id

        self.train_dataset = train_dataset

        all_range = list(range(len(self.train_dataset)))  # 生成的数据集的长度
        data_len = int(len(self.train_dataset) / self.conf['no_models'])  # 按照客户端个数平方
        train_indices = all_range[id * data_len: (id + 1) * data_len]

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"],
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                            train_indices))

    def get_accuracy(self,logit, target, batch_size):
        ''' Obtain accuracy for training round '''
        corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects / batch_size
        return accuracy.item()
    def local_train(self, model):

        # for name, param in model.state_dict().items():
        #     self.local_model.state_dict()[name].copy_(param.clone())
        self.local_model.load_state_dict(model.state_dict())
        # print(id(model))
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
                                    momentum=self.conf['momentum'])
        # print(id(self.local_model))
        train_running_loss = 0.0
        train_acc = 0.0
        self.local_model.train()
        for e in range(self.conf["local_epochs"]):

            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                self.local_model.hidden = model.init_hidden()

                inputs = data.view(-1, 28, 28)
                # data_x = data.view(-1, 28, 28)
                output = self.local_model(inputs)

                # loss = torch.nn.functional.cross_entropy(output, target)
                loss = torch.nn.functional.cross_entropy(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # print("Epoch %d done." % e)
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - model.state_dict()[name])
        # print(diff[name])

        return diff
