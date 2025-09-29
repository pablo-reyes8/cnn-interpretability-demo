import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def get_module_by_name(model: nn.Module, name: str):
    """ Navega el árbol con rutas tipo 'layer4.2.conv2'. """
    mod = model
    for attr in name.split('.'):
        if attr.isdigit():
            mod = mod[int(attr)]
        else:
            mod = getattr(mod, attr)
    return mod


def _minmax01(x: torch.Tensor, eps: float = 1e-8):
    """ Normaliza por tensor a [0,1]. """
    mn = x.amin(dim=(-1, -2), keepdim=True)
    mx = x.amax(dim=(-1, -2), keepdim=True)
    return (x - mn) / (mx - mn + eps)


def _to_hwc_uint8(img: torch.Tensor) -> np.ndarray:
    """ (C,H,W) o (1,H,W) -> H,W,3 en [0..255] uint8 """
    if img.dim() == 2:
        img = img.unsqueeze(0)
    if img.size(0) == 1:
        img = img.repeat(3, 1, 1)
    img = img.clamp(0,1)
    img = (img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    return img


def _kernel_to_rgb(w: torch.Tensor) -> torch.Tensor:
    """
    w: (C,kh,kw) -> (3,kh,kw) normalizado [0,1]
    - Si C==3 usa RGB nativo
    - Si no, PCA con q=min(3, C, kh*kw); si q<3, replica hasta 3
    - Fallback: promedio a gris
    """
    C, kh, kw = w.shape
    if C == 3:
        return _minmax01(w.unsqueeze(0)).squeeze(0)
    try:
        P = kh*kw
        q = max(1, min(3, C, P))
        w2 = w.reshape(C, P)
        w2 = w2 - w2.mean(dim=1, keepdim=True)
        U, S, V = torch.pca_lowrank(w2, q=q)
        z = (U[:, :q].T @ w2).reshape(q, kh, kw)
        if q == 1:
            z = z.repeat(3,1,1)
        elif q == 2:
            z = torch.cat([z, z[0:1]], dim=0)
        return _minmax01(z.unsqueeze(0)).squeeze(0)
    except Exception:
        z = _minmax01(w.mean(dim=0, keepdim=True).unsqueeze(0)).squeeze(0)
        return z.repeat(3,1,1)


def _render_kernel_tile(w: torch.Tensor, tile_px: int = 140):
    img = _kernel_to_rgb(w)                 
    tile = Image.fromarray(_to_hwc_uint8(img)).resize((tile_px, tile_px), Image.NEAREST)
    return np.array(tile)


def overlay_heatmap_on_rgb(rgb01: torch.Tensor, heat01: torch.Tensor, alpha: float = 0.35):
    """
    rgb01: (1,3,H,W) en [0,1]; heat01: (1,1,H,W) en [0,1].
    Devuelve H,W,3 uint8 mezclando un colormap simple (inferno-like sin matplotlib).
    """
    # Colormap manual simple (negro -> rojo -> amarillo -> blanco)
    h = heat01.squeeze(0).squeeze(0)  # (H,W)
    c0 = torch.clamp(4*h - 1.5, 0, 1)  # R
    c1 = torch.clamp(4*h - 2.5, 0, 1)  # G
    c2 = torch.clamp(4*h - 3.5, 0, 1)  # B
    cmap = torch.stack([c0, c1, c2], dim=0).unsqueeze(0)  # (1,3,H,W)

    out = (1 - alpha)*rgb01 + alpha*cmap
    return _to_hwc_uint8(out[0])


def denormalize(x, mean=[0.48293063044548035, 0.44492557644844055, 0.3957090973854065], 
                std = [0.2592383325099945, 0.25327032804489136, 0.2598187029361725]):
    m = torch.tensor(mean, device=x.device).view(1,3,1,1)
    s = torch.tensor(std,  device=x.device).view(1,3,1,1)
    return x * s + m


def _resize_np(arr: np.ndarray, size: int) -> np.ndarray:
    """ arr: H,W,3 -> resize cuadrado size×size con NEAREST """
    return np.array(Image.fromarray(arr).resize((size, size), Image.NEAREST))



######### Grad Cam ######### 

def _denormalize01(x: torch.Tensor, mean=[0.48293063044548035, 0.44492557644844055, 0.3957090973854065], 
                   std=[0.2592383325099945, 0.25327032804489136, 0.2598187029361725]):
    m = torch.tensor(mean, device=x.device).view(1,3,1,1)
    s = torch.tensor(std,  device=x.device).view(1,3,1,1)
    return (x * s + m).clamp(0,1)  # (1,3,H,W) en [0,1]

def _to_uint8_rgb(img01: torch.Tensor) -> np.ndarray:
    """
    Acepta (3,H,W) o (B,3,H,W). Si B>1, toma el primer elemento.
    Devuelve H,W,3 uint8.
    """
    # Si viene 4D, quedarnos con el primer "batch"
    if img01.dim() == 4:
        # caso típico correcto: (1,3,H,W) o (B,3,H,W)
        if img01.shape[1] == 3:
            img01 = img01[0]  # -> (3,H,W)
        else:
            # caso patológico por broadcasting: (3,3,H,W)
            # interpretamos la 1ª dim como "batch" y tomamos el primero
            img01 = img01[0]  # -> (3,H,W)

    # Si por alguna razón viene (1,H,W), duplicamos a RGB
    if img01.dim() == 3 and img01.shape[0] == 1:
        img01 = img01.repeat(3, 1, 1)

    assert img01.dim() == 3 and img01.shape[0] == 3, \
        f"Esperaba (3,H,W); tengo {tuple(img01.shape)}"

    img01 = img01.detach().clamp(0,1).contiguous()
    return (img01.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)


def _to_uint8_gray(img01: torch.Tensor) -> np.ndarray:
    """
    Acepta (1,H,W) o (H,W). Retorna H,W uint8.
    """
    if img01.dim() == 3 and img01.shape[0] == 1:
        img01 = img01[0]
    elif img01.dim() == 2:
        pass
    else:
        raise ValueError(f"Esperaba (1,H,W) o (H,W); tengo {tuple(img01.shape)}")
    img01 = img01.detach().clamp(0,1)
    return (img01.cpu().numpy() * 255).astype(np.uint8)

def _cmap(gray01, mode="magma"):
    g = gray01.squeeze(0).clamp(0,1)  # (H,W)
    if mode == "gray":
        return torch.stack([g,g,g], dim=0)
    if mode == "viridis":
        r = torch.clamp(12*g - 4.0, 0, 1)
        g2= torch.clamp(12*g - 7.0, 0, 1)
        b = torch.clamp(12*g - 10.0,0, 1)
        return torch.stack([r, g2, b], dim=0)
    if mode == "magma":  # oscuro→naranja→magenta (sin blanco puro)
        r = torch.clamp(2.5*g + 0.1, 0, 1)
        g2= torch.clamp(2.0*g - 0.2, 0, 1)**1.2
        b = torch.clamp(3.0*g - 1.7, 0, 1)
        return torch.stack([r, g2, b], dim=0)
    # fallback tipo "hot"
    r = torch.clamp(3.0*g - 0.5, 0, 1)
    g2= torch.clamp(3.0*g - 1.5, 0, 1)
    b = torch.clamp(3.0*g - 2.5, 0, 1)
    return torch.stack([r, g2, b], dim=0)

def _colormap_simple(gray01: torch.Tensor) -> torch.Tensor:
    """
    gray01: (1,H,W) en [0,1] -> (3,H,W) tipo inferno simple.
    """
    g = gray01.squeeze(0)  # (H,W)
    r = torch.clamp(4*g - 1.5, 0, 1)
    g2= torch.clamp(4*g - 2.5, 0, 1)
    b = torch.clamp(4*g - 3.5, 0, 1)
    return torch.stack([r, g2, b], dim=0)  # (3,H,W)





############## Pixeles mas Importantes ###########


def _denorm01_np(x, mean, std):
    m = np.array(mean, dtype=np.float32).reshape(1,1,3)
    s = np.array(std,  dtype=np.float32).reshape(1,1,3)
    im = x[0].detach().cpu().permute(1,2,0).numpy()
    return (im*s + m).clip(0,1)

def _cmap_np(h, mode="magma"):
    if mode == "gray": return np.stack([h,h,h], -1)
    if mode == "viridis":
        r = np.clip(12*h-4.0,0,1); g = np.clip(12*h-7.0,0,1); b = np.clip(12*h-10.0,0,1)
        return np.stack([r,g,b], -1)
    r = np.clip(2.5*h+0.1,0,1); g = np.clip(2.0*h-0.2,0,1)**1.2; b = np.clip(3.0*h-1.7,0,1)
    return np.stack([r,g,b], -1)

def _to_norm(x01, mean, std):
    """ x01: H,W,3 [0,1] -> (1,3,H,W) normalizado """
    im = torch.from_numpy(((x01 - mean)/std).transpose(2,0,1)).unsqueeze(0).float()
    return im

def _blur_baseline_np(base01, sigma=3.0):
    # blur canal a canal
    b = np.stack([gaussian_filter(base01[...,c], sigma=sigma) for c in range(3)], axis=-1)
    return b.clip(0,1)


def _tv_loss(x):
    dh = (x[:,:,1:,:] - x[:,:,:-1,:]).abs().mean()
    dw = (x[:,:,:,1:] - x[:,:,:,:-1]).abs().mean()
    return dh + dw