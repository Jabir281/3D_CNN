import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    """Simple Grad-CAM implementation for 3D CNNs (PyTorch).

    Usage:
        gradcam = GradCAM(model, target_layer_name='conv4')
        cam = gradcam.generate_cam(input_tensor, target_class=1)

    Returns a numpy array with shape (D, H, W) for each sample.
    """

    def __init__(self, model, target_layer_name='conv4'):
        self.model = model
        self.model.eval()
        self.target_layer = getattr(self.model, target_layer_name)

        # Placeholders for activations and gradients
        self.activations = None
        self.gradients = None

        # Register forward hook to capture activations and attach grad hook
        def forward_hook(module, input, output):
            # output is a tensor with shape (B, C, D, H, W)
            self.activations = output.detach()

            # register a hook on the output tensor to capture the gradients
            def _save_grad(grad):
                self.gradients = grad.detach()

            output.register_hook(_save_grad)

        self.fh = self.target_layer.register_forward_hook(forward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        """Generate Grad-CAM for the given input tensor.

        Args:
            input_tensor (torch.Tensor): shape (B, C, D, H, W)
            target_class (int or None): which class to compute gradients for. For
                binary models returning a single probability, use 1 to focus on
                the positive class. If None, uses the model output score.

        Returns:
            cam (numpy.ndarray): shape (B, D, H, W) - values in [0, 1]
        """
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        # Forward
        self.model.zero_grad()
        output = self.model(input_tensor)  # (B, 1) with sigmoid

        # Choose scalar target
        if target_class is None:
            # use the output itself (probability) as the target
            target = output.squeeze()
        else:
            # For binary output (single logit after sigmoid), use the probability
            target = output.squeeze() if target_class == 1 else (1.0 - output.squeeze())

        # Backward on the target sum (for batch support)
        if target.requires_grad:
            target_sum = target.sum()
        else:
            target_sum = output.squeeze().sum()

        target_sum.backward(retain_graph=True)

        # activations: (B, C, D, H, W)
        # gradients: (B, C, D, H, W)
        activations = self.activations  # already detached
        gradients = self.gradients

        if activations is None or gradients is None:
            raise RuntimeError("GradCAM couldn't find activations or gradients. Ensure the target layer name is correct and a forward+backward pass was made.")

        # Compute weights: global-average-pool over D,H,W -> (B, C, 1, 1, 1)
        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)

        # Weighted sum of activations
        cam = (weights * activations).sum(dim=1)  # (B, D, H, W)

        # Apply ReLU
        cam = F.relu(cam)

        # Normalize per-sample and upsample to input size
        cams = []
        input_size = input_tensor.shape[2:]
        for i in range(cam.shape[0]):
            single_cam = cam[i].unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
            # Upsample to input size
            up = F.interpolate(single_cam, size=input_size, mode='trilinear', align_corners=False)
            up = up.squeeze().cpu().numpy()  # (D,H,W)
            # Normalize
            up_min, up_max = up.min(), up.max()
            if up_max - up_min > 1e-6:
                up = (up - up_min) / (up_max - up_min)
            else:
                up = up * 0.0
            cams.append(up)

        cams = np.stack(cams, axis=0)
        return cams

    def close(self):
        try:
            self.fh.remove()
        except Exception:
            pass

