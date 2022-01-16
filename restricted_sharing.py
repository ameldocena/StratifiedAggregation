import torch
import numpy as np
from collections import OrderedDict

def get_uploaded_params(args, idx, epoch, save=True):
    """
    Creates the model file that will be shared with the aggregator when parameter sharing is restricted
    Parameters with the smallest gradients will be replaced with numpy NaNs
    This is a novel contribution beyond the Tolpegin et al framework

    :param args: experiment arguments
    :type args: Arguments
    :param idx: worker ID
    :type idx: int
    :param epoch: epoch number
    :type epoch: int
    :param portion: portion of parameters that should be shared; e.g. 0.5 means 50% of parameters with highest gradients will be shared
    :return: OrderedDict
    """
    folder = args.get_save_model_folder_path()

    #The way this works is that a start model and end model are stored for each client at every epoch
    end_path = f"{folder}/model_{idx}_{epoch}_end.model"
    end_model = torch.load(end_path)

    #If all parameters are shared, we return the entire end model
    portion = args.get_portion_uploaded()
    if portion == 1:
        if save:
            torch.save(end_model, f"{folder}/model_{idx}_{epoch}_uploaded.model")
        return end_model

    #Else, we return only a portion of the end model
    #We set some parameters for sharing by sorting them based on their gradient values.
    #We replace those restricted parameters for sharing by NaNs
    start_path = f"{folder}/model_{idx}_{epoch}_start.model" #Loads model of client idx at given epoch
    start_model = torch.load(start_path)

    grad_dict = OrderedDict()

    for i in start_model.keys():
        grad_dict[i] = end_model[i] - start_model[i]
    end_model_portioned = OrderedDict()

    for i in grad_dict.keys(): #Ordered keys
        end_model_portioned[i] = torch.empty(grad_dict[i].shape)
        if len(grad_dict[i].shape) > 1:
            for j in range(len(grad_dict[i])):
                topx = round(portion * grad_dict[i][j].numel())
                if topx < 1:
                    grad_dict[i][j].fill_(np.NaN)
                    end_model_portioned[i][j] = torch.full_like(grad_dict[i][j], np.NaN)
                else:
                    grad_dict[i][j][torch.abs(grad_dict[i][j]) <
                                    torch.sort(torch.abs(grad_dict[i][j]).flatten(), descending=True)[0][topx - 1]] = np.NaN
                    end_model_portioned[i][j] = torch.full_like(grad_dict[i][j], np.NaN)
                    end_model_portioned[i][j][~torch.isnan(grad_dict[i][j])] = end_model[i][j][~torch.isnan(grad_dict[i][j])].cpu()
        elif len(grad_dict[i].shape) == 1:
            j = grad_dict[i]
            topx = round(portion * j.numel())
            if topx < 1:
                j.fill_(np.NaN)
                end_model_portioned[i] = torch.full_like(grad_dict[i], np.NaN)
            else:
                j[torch.sort(torch.abs(j).flatten(), descending=True)[1][topx:]] = np.NaN
                end_model_portioned[i] = torch.full_like(grad_dict[i], np.NaN)
                end_model_portioned[i][~torch.isnan(grad_dict[i])] = end_model[i][~torch.isnan(grad_dict[i])]
        else:
            end_model_portioned[i] = end_model[i]
    if save:
        torch.save(end_model_portioned, f"{folder}/model_{idx}_{epoch}_uploaded.model")
    #We return the portioned (or restrictedly shared) end model
    return end_model_portioned

def get_downloadable_params(args, epoch, save=True):
    """
    Creates the model file that will be shared from aggregator to local models when download is restricted
    Parameters with the smallest gradients will be replaced with numpy NaNs
    This is a novel contribution beyond the Tolpegin et al framework

    :param args: experiment arguments
    :type args: Arguments
    :param epoch: epoch number
    :type epoch: int
    :param portion: portion of parameters that should be shared; e.g. 0.5 means 50% of parameters with highest gradients will be shared
    :return: OrderedDict
    """
    folder = args.get_save_model_folder_path()
    end_path = f"{folder}/model_GLOBAL_{epoch}_end.model"
    end_model = torch.load(end_path)
    portion = args.get_portion_downloaded()
    if portion == 1:
        if save:
            torch.save(end_model, f"{folder}/model_GLOBAL_{epoch}_shared.model")
        return end_model

    start_path = f"{folder}/model_GLOBAL_{epoch}_start.model"
    start_model = torch.load(start_path)

    grad_dict = OrderedDict()

    for i in start_model.keys():
        grad_dict[i] = end_model[i] - start_model[i]

    end_model_portioned = OrderedDict()

    for i in grad_dict.keys():
        end_model_portioned[i] = torch.empty(grad_dict[i].shape)
        if len(grad_dict[i].shape) > 1:
            for j in range(len(grad_dict[i])):
                topx = round(portion * grad_dict[i][j].numel())
                if topx < 1:
                    grad_dict[i][j].fill_(np.NaN)
                    end_model_portioned[i][j] = torch.full_like(grad_dict[i][j], np.NaN)
                else:
                    grad_dict[i][j][torch.abs(grad_dict[i][j]) <
                                    torch.sort(torch.abs(grad_dict[i][j]).flatten(), descending=True)[0][topx - 1]] = np.NaN
                    end_model_portioned[i][j] = torch.full_like(grad_dict[i][j], np.NaN)
                    end_model_portioned[i][j][~torch.isnan(grad_dict[i][j])] = end_model[i][j][~torch.isnan(grad_dict[i][j])].cpu()
        elif len(grad_dict[i].shape) == 1:
            j = grad_dict[i]
            topx = round(portion * j.numel())
            if topx < 1:
                j.fill_(np.NaN)
                end_model_portioned[i] = torch.full_like(grad_dict[i], np.NaN)
            else:
                j[torch.sort(torch.abs(j).flatten(), descending=True)[1][topx:]] = np.NaN
                end_model_portioned[i] = torch.full_like(grad_dict[i], np.NaN)
                end_model_portioned[i][~torch.isnan(grad_dict[i])] = end_model[i][~torch.isnan(grad_dict[i])]
        else:
            end_model_portioned[i] = end_model[i]
    if save:
        torch.save(end_model_portioned, f"{folder}/model_GLOBAL_{epoch}_shared.model")
    return end_model_portioned
