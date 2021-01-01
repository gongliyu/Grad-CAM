from typing import Union, List, Tuple

import torch
import torch.nn.functional as F
import pickle


def _find_layer(model: torch.nn.Module,
                target_layer: Union[str, torch.nn.Module]) -> torch.nn.Module:
    """Find the specified layer in a model.

    Args:
        model: a neural network model as a PyTorch Module.
        target_layer: the target layer in the model. If a str, will be used as the name
            of layer to look up in the model. If a Module, will check if it exist in the model.
    """
    ret = None
    if not isinstance(target_layer, (str, torch.nn.Module)):
        raise ValueError(f'target_layer must be a str or a module instance.')
    if isinstance(target_layer, torch.nn.Module):
        if not any(target_layer is x for x in model.modules()):
            raise ValueError(f'Specified target layer: {target_layer} is not in the model.')
    elif isinstance(target_layer, str):
        for name, layer in model.named_modules():
            if target_layer in name:
                if ret is not None:
                    raise ValueError(f'Multiple matches for target layer name prefix: {target_layer}')
                ret = layer
        if ret is None:
            raise ValueError(f'No matches for target layer name prefix: {target_layer}')
    return ret


def grad_cam(model: torch.nn.Module,
             input_image: torch.Tensor,
             feature_layer: Union[str, torch.nn.Module],
             score_layer: Union[str, torch.nn.Module, None] = None,
             class_index: Union[int, List[int], Tuple[int], None] = None,
             clip_negative: bool = True,
             normalize: bool = True,
             return_intermediate_results: bool = False
             ) -> torch.Tensor:
    """Grad-CAM: gradient class activation map

    Implement the Grad-CAM algorithm:
    Selvaraja et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", IJCV, 2019.
    https://arxiv.org/abs/1610.02391

    Args:
        model: a neural network model.
        input_image: input image(s), with shape (C, H, W) or (N, C, H, W).
        feature_layer: feature layer or name of feature layer.
        score_layer: score (logits) layer or name of score layer. If None, the output of the model will
            be used as score. Default to None.
        class_index: index to the target class whose activation map will calculated. If None, the class with
            largest confidence will be calculated. Default to None.
        clip_negative: whether clip negative values of class activation (apply ReLU). Default to True.
        normalize: whether normalize values of class activation to [0, 1]. Default to True.
        return_intermediate_results: whether return the intermediate results, i.e., feature maps, logits, gradients etc.

    Returns:
        class activation map (CAM) of the input image(s) for the specified class if return_intermediate_results
        if False, otherwise, return a tuple contain CAM and intermediate results.

    """
    assert input_image.ndim in {3, 4}, 'input_image must has shape (C, H, W) or (N, C, H, W)'
    if input_image.ndim == 3:
        single_input_image = True
        input_image = input_image.unsqueeze(0)
    else:
        single_input_image = False
    # register hook to record feature map
    feature_layer = _find_layer(model, feature_layer)
    fea_map = None

    def _get_fea_map_ref(_, __, x):
        nonlocal fea_map
        fea_map = x

    h_fea = feature_layer.register_forward_hook(_get_fea_map_ref)

    # forward step
    classification_score = None
    if score_layer is None:
        # use output of the whole model as classification score
        classification_score = model(input_image)
    else:
        # register hook to record classification score
        score_layer = _find_layer(score_layer)

        def _get_logit_ref(_, __, x):
            nonlocal classification_score
            classification_score = x

        h_score = score_layer.register_forward_hook(_get_logit_ref)
        model(input_image)
        h_score.remove()  # remove
    h_fea.remove()  # remove the feature map hook

    # register hook to obtain gradients
    grad = None

    def _get_grad_ref(_grad):
        nonlocal grad
        grad = _grad

    h_grad = fea_map.register_hook(_get_grad_ref)
    class_idx = class_index or classification_score.argmax(-1)

    # backward step
    model.zero_grad()
    classification_score[torch.arange(classification_score.shape[0]), class_idx].backward()
    h_grad.remove()

    # calculate CAM
    with torch.no_grad():
        alpha = torch.mean(grad, axis=(-2, -1), keepdims=True)
        cam = torch.sum(alpha * fea_map, dim=-3)
        if clip_negative:
            cam = F.relu(cam)
        if normalize:
            cam_max = cam.max(axis=-1, keepdims=True)[0].max(axis=-2, keepdims=True)[0]
            cam = cam / cam_max
    if single_input_image:
        cam = cam[0]

    if return_intermediate_results:
        if single_input_image:
            fea_map = fea_map[0]
            classification_score = classification_score[0]
            grad = grad[0]
        ret = cam, dict(feature_map=fea_map, classification_score=classification_score, gradient=grad)
    else:
        ret = cam
    return ret
