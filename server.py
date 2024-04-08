from collections import OrderedDict

import models, torch


class Server(object):

    def __init__(self, conf, eval_dataset):

        self.conf = conf

        self.global_model = models.get_model(self.conf["model_name"])

        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)

    def model_aggregate(self, weight_accumulator):
        for name, data in self.global_model.state_dict().items():

            update_per_layer = weight_accumulator[name] * self.conf["lambda"]

            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    def get_accuracy(self,logit, target, batch_size):
        ''' Obtain accuracy for training round '''
        corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects / batch_size
        return accuracy.item()
    def model_eval(self):
        self.global_model.eval()

        total_loss = 0.0
        test_acc = 0.0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            inputs = data.view(-1, 28, 28)
            output = self.global_model(inputs)

            total_loss += torch.nn.functional.cross_entropy(output, target,
                                                            reduction='sum').item()  # sum up batch loss
            # pred = output.data.max(1)[1]  # get the index of the max log-probability
            # correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            test_acc += self.get_accuracy(output, target, 32)

        acc = 100.0 * (float(test_acc) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc, total_l