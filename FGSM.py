import torch
from torch.autograd import Variable


class FGSM:
    def __init__(self, model, criterion, epsilon):
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        assert isinstance(model, torch.nn.Module), "Input parameter model is not nn.Module. Check the model"
        assert isinstance(criterion, torch.nn.Module), "Input parameter criterion is no Loss. Check the criterion"
        assert (0 <= epsilon <= 1), "episilon must be 0 <= epsilon <= 1"
        self.model.eval()


    def __call__(self, input, labels):
        # For calculating gradient
        input_for_gradient = Variable(input, requires_grad=True).to(self.device)
        out = self.model(input_for_gradient)
        loss = self.criterion(out, Variable(labels))

        # Calculate gradient
        loss.backward()

        # Calculate sign of gradient
        signs = torch.sign(input_for_gradient.grad.data)

        # Add
        input_for_gradient.data = input_for_gradient.data + (self.epsilon * signs)

        return input_for_gradient
