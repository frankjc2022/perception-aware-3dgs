import torch


def depth_smoothness_loss(image, depth):

    dx = depth[:, :, 1:] - depth[:, :, :-1]
    dy = depth[:, 1:, :] - depth[:, :-1, :]

    # for edge
    grad_x = torch.mean(torch.abs(image[:, :, 1:] - image[:, :, :-1]), dim=0, keepdim=True)
    grad_y = torch.mean(torch.abs(image[:, 1:, :] - image[:, :-1, :]), dim=0, keepdim=True)

    beta = 1.0
    weight_x = torch.exp(-beta * grad_x)
    weight_y = torch.exp(-beta * grad_y)

    dx *= weight_x
    dy *= weight_y

    loss = dx.abs().mean() + dy.abs().mean()
    return loss


def total_variation_loss(image: torch.Tensor) -> torch.Tensor:

    dx = image[..., :-1] - image[..., 1:]
    dy = image[..., :, :-1] - image[..., :, 1:]

    return dx.abs().mean() + dy.abs().mean()
