import sys, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
from . import vis_utils
import torch
from torch import nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
from scipy.ndimage import gaussian_filter

def kernels_depth_matrix(model: nn.Module, *,
                         rows_config=None,
                         cols: int = 12,
                         tile_px: int = 150,
                         bg=(10,10,10),
                         fg=(235,235,235),
                         pad_out_x: int = 40,   
                         pad_out_y: int = 20,   
                         pad_row: int = 12,     
                         col_gap: int = 20,     
                         row_title_px: int = 20):
    """
    Render 4xN (por defecto) en formato más panorámico.
    - pad_out_x grande y col_gap grande -> panel más ancho sin crecer en alto.
    - pad_out_y y pad_row pequeños -> compacto en vertical.
    """

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", row_title_px)
    except:
        font = ImageFont.truetype("arial.ttf", row_title_px)

    if rows_config is None:
        rows_config = [
            ("Baja (conv1 7×7)",           "conv1",            None),
            ("Intermedia baja (L2.0 c2)",  "layer2.0.conv2",   None),
            ("Intermedia alta (L4.2 c2)",  "layer4.2.conv2",   None),
            ("Muy profunda (L5.2 c2)",     "layer5.2.conv2",   None),]

    row_images = []
    max_row_w, tile_h = 0, tile_px
    for row_title, layer_name, idxs in rows_config:
        layer = vis_utils.get_module_by_name(model, layer_name)
        if not isinstance(layer, nn.Conv2d):
            raise TypeError(f"{layer_name} no es nn.Conv2d")
        W = layer.weight.detach().cpu()
        K = W.shape[0]
        if (idxs is None) or (len(idxs) == 0):
            idxs = np.linspace(0, K-1, num=cols, dtype=int).tolist()
        else:
            idxs = [i for i in idxs if 0 <= i < K]
            if len(idxs) < cols:
                idxs += [i for i in range(K) if i not in idxs][:cols-len(idxs)]
            elif len(idxs) > cols:
                idxs = idxs[:cols]

        tiles = [vis_utils._render_kernel_tile(W[i], tile_px=tile_px) for i in idxs]
        row_w = cols*tile_px + (cols-1)*col_gap
        max_row_w = max(max_row_w, row_w)
        row_images.append((row_title, tiles))

    title_h = row_title_px + 6
    row_h   = title_h + tile_h
    rows    = len(row_images)
    H = pad_out_y + rows*row_h + (rows-1)*pad_row + pad_out_y
    W = pad_out_x + max_row_w + pad_out_x

    panel = Image.new("RGB", (W, H), color=bg)
    draw  = ImageDraw.Draw(panel)

    y = pad_out_y
    for row_title, tiles in row_images:
        tw = draw.textlength(row_title, font=font)
        tx = pad_out_x + (max_row_w - tw)//2
        draw.text((tx, y), row_title, fill=fg, font=font)

        y_tiles = y + title_h
        for j, tile_np in enumerate(tiles):
            x = pad_out_x + j*(tile_px + col_gap)
            panel.paste(Image.fromarray(tile_np), (x, y_tiles))

        y = y + row_h + pad_row

    return np.array(panel)


@torch.no_grad()
def feature_maps_depth_matrix(
    model: nn.Module,
    x: torch.Tensor,*,
    rows_config=None,        
    cols: int = 6,          
    tile_px: int = 140,     
    bg=(10,10,10),
    fg=(235,235,235),
    pad_out_x: int = 60,
    pad_out_y: int = 24,
    pad_row: int = 12,
    col_gap: int = 18,
    row_title_px: int = 24):

    """
    Devuelve un panel HxWx3 (uint8):
      - 1 fila por capa (profundidad)
      - 'cols' canales consecutivos por fila (orden: start_idx, start_idx+1, ...)
    """

    # Capas por defecto 
    if rows_config is None:
        rows_config = [
    ("Muy baja (conv1)",         "conv1",          0),
    ("Intermedia baja (L2.0c2)", "layer2.0.conv2", 0),
    ("Intermedia (L3.1c2)",      "layer3.1.conv2", 0),
    ("Intermedia alta (L4.2c2)", "layer4.2.conv2", 0),
    ("Muy profunda (L5.2c2)",    "layer5.2.conv2", 0),]

    #  Hooks para capturar activaciones de TODAS las capas en 1 forward
    store = {}
    hooks = []
    for _, layer_name, _ in rows_config:
        m = vis_utils.get_module_by_name(model, layer_name)
        hooks.append(m.register_forward_hook(lambda m,i,o,ln=layer_name: store.setdefault(ln, o.detach())))
    model.eval()
    _ = model(x)  # único forward
    for h in hooks: h.remove()

    # Preparar fuente para títulos
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", row_title_px)
    except:
        font = ImageFont.truetype("arial.ttf", row_title_px)

    row_images = []
    max_row_w = 0
    for row_title, layer_name, start_idx in rows_config:
        a = store[layer_name]          
        a = a[0]                       
        C, Hf, Wf = a.shape

        idxs = list(range(start_idx, min(start_idx+cols, C)))
        if len(idxs) < cols:
            extra = [i for i in range(0, C) if i not in idxs][:cols-len(idxs)]
            idxs += extra

        tiles = []
        for i in idxs:
            fmap = a[i].unsqueeze(0)             
            fmap01 = vis_utils._minmax01(fmap)
            rgb = vis_utils._colormap_simple(fmap01)        
            rgb_np = vis_utils._to_uint8_rgb(rgb)
            tiles.append(vis_utils._resize_np(rgb_np, tile_px))  

        row_w = cols*tile_px + (cols-1)*col_gap
        max_row_w = max(max_row_w, row_w)
        row_images.append((row_title, tiles))

    title_h = row_title_px + 8
    row_h   = title_h + tile_px
    rows    = len(row_images)
    H = pad_out_y + rows*row_h + (rows-1)*pad_row + pad_out_y
    W = pad_out_x + max_row_w + pad_out_x

    panel = Image.new("RGB", (W, H), color=bg)
    draw  = ImageDraw.Draw(panel)

    y = pad_out_y
    for row_title, tiles in row_images:
        tw = draw.textlength(row_title, font=font)
        tx = pad_out_x + (max_row_w - tw)//2
        draw.text((tx, y), row_title, fill=fg, font=font)

        y_tiles = y + title_h
        for j, tile_np in enumerate(tiles):
            x0 = pad_out_x + j*(tile_px + col_gap)
            panel.paste(Image.fromarray(tile_np), (x0, y_tiles))
        y = y + row_h + pad_row

    return np.array(panel)

def gradcam(
    model: nn.Module,
    x: torch.Tensor,                         
    layer_name: str,                           
    target_class: int | None = None,*,
    img_mean=(0.48293063044548035, 0.44492557644844055, 0.3957090973854065),
    img_std= (0.2592383325099945, 0.25327032804489136, 0.2598187029361725),
    alpha: float = 0.35):

    """
    Devuelve: (overlay_rgb_uint8 HxWx3, heat_gray_uint8 HxW, class_id, prob_class)
    """

    assert x.dim() == 4 and x.shape[0] == 1 and x.shape[1] == 3, f"x debe ser (1,3,H,W), recibido {tuple(x.shape)}"
    model.eval()

    A_store = {}
    layer = vis_utils.get_module_by_name(model, layer_name)
    
    def fwd_hook(m,i,o): 
        A_store["A"] = o

    h = layer.register_forward_hook(fwd_hook)

    x_in = x.clone().detach().requires_grad_(True)  
    logits = model(x_in)                            
    probs = torch.softmax(logits, dim=1)

    if target_class is None:
        target_class = int(torch.argmax(probs, dim=1).item())
    p = float(probs[0, target_class].item())

    assert "A" in A_store, f"No capturé activación en {layer_name}. ¿El nombre de capa es correcto?"
    A = A_store["A"]                                 
    h.remove()                                     

    score = logits[:, target_class].sum()
    dA = torch.autograd.grad(score, A, retain_graph=False, create_graph=False)[0]  

    # Grad-CAM
    weights = dA.mean(dim=(2,3), keepdim=True)     
    cam = (weights * A).sum(dim=1, keepdim=True)    
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  

    rgb01 = vis_utils._denormalize01(x, mean=img_mean, std=img_std)    
    cmap  = vis_utils._colormap_simple(cam)                           
    overlay01 = (1 - alpha)*rgb01[0] + alpha*cmap             
    overlay_np =vis_utils. _to_uint8_rgb(overlay01)                    
    heat_np    = vis_utils._to_uint8_gray(cam[0])                     

    return overlay_np, heat_np, target_class, p



def gradcam_grid_panel_using_your_fn(
    model, x,
    *, ncols=5, tile_px=256, alpha=0.42,
    use_recolor=True, cmap_mode="magma",
    img_mean=(0.4829306304, 0.4449255764, 0.3957090974),
    img_std =(0.2592383325, 0.2532703280, 0.2598187029),
    bg=(10,10,10), fg=(235,235,235),
    pad_out=20, gap_xy=18, title_px=22,
    show_layer_path=False , items=None):

    import numpy as np
    
    items = [
    ("conv1",        "conv1"),
    ("L2.0 c2",      "layer2.0.conv2"),
    ("L2.1 c2",      "layer2.1.conv2"),
    ("L3.0 c2",      "layer3.0.conv2"),
    ("L3.2 c2",      "layer3.2.conv2"),

    ("L4.0 c2",      "layer4.0.conv2"),
    ("L4.2 c2",      "layer4.2.conv2"),
    ("L4.4 c2",      "layer4.4.conv2"),
    ("L4.6 c2",      "layer4.6.conv2"),
    ("L4.8 c2",      "layer4.8.conv2"),

    ("L4.10 c2",     "layer4.10.conv2"),
    ("L4.12 c2",     "layer4.12.conv2"),
    ("L4.16 c2",     "layer4.16.conv2"),
    ("L5.0 c2",      "layer5.0.conv2"),
    ("L5.2 c2",      "layer5.2.conv2"),] 

    def _short(t, n=28): return t if len(t)<=n else t[:n-1]+"…"

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", title_px)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", title_px)
        except:
            font = ImageFont.load_default()

    def _denorm01_np(x, mean, std):
        import numpy as np
        m = np.array(mean, dtype=np.float32).reshape(1,1,3)
        s = np.array(std,  dtype=np.float32).reshape(1,1,3)
        im = x[0].detach().cpu().permute(1,2,0).numpy()
        return (im*s + m).clip(0,1)

    def _cmap_np(heat01, mode="magma"):
        import numpy as np
        h = heat01
        if mode == "gray": return np.stack([h,h,h], -1)
        if mode == "viridis":
            r = np.clip(12*h-4.0, 0,1); g = np.clip(12*h-7.0, 0,1); b = np.clip(12*h-10.0,0,1)
            return np.stack([r,g,b], -1)
        r = np.clip(2.5*h+0.1,0,1); g = np.clip((2.0*h-0.2),0,1)**1.2; b = np.clip(3.0*h-1.7,0,1)
        return np.stack([r,g,b], -1)

    base01 = _denorm01_np(x, img_mean, img_std)  # H,W,3
    tiles, titles = [], []
    for short_title, lname in items:
        overlay_np, heat_np, cls_id, p = gradcam(
            model, x, layer_name=lname,
            img_mean=img_mean, img_std=img_std, alpha=alpha)
        
        if use_recolor:
            h01   = (heat_np.astype(np.float32)/255.0)
            color = _cmap_np(h01, mode=cmap_mode)
            overlay_np = ((1-alpha)*base01 + alpha*color)
            overlay_np = (overlay_np*255).astype(np.uint8)

        tile = Image.fromarray(overlay_np).resize((tile_px, tile_px), Image.BILINEAR)
        tiles.append(np.array(tile))
        # 👇 título: solo el corto, sin path completo
        titles.append(short_title if not show_layer_path else f"{short_title} • {lname}")

    n = len(tiles); ncols = max(1, ncols); nrows = math.ceil(n/ncols)
    title_h = title_px + 6; cell_h = title_h + tile_px; cell_w = tile_px
    W = pad_out + ncols*cell_w + (ncols-1)*gap_xy + pad_out
    H = pad_out + nrows*cell_h + (nrows-1)*gap_xy + pad_out

    panel = Image.new("RGB", (W, H), color=bg); draw = ImageDraw.Draw(panel)
    for idx, (tile_np, text) in enumerate(zip(tiles, titles)):
        r, c = divmod(idx, ncols)
        x0 = pad_out + c*(cell_w + gap_xy)
        y0 = pad_out + r*(cell_h + gap_xy)
        tw = draw.textlength(text, font=font); tx = x0 + (cell_w - tw)//2
        draw.text((tx, y0), text, fill=fg, font=font)
        panel.paste(Image.fromarray(tile_np), (x0, y0 + title_h))

    return np.array(panel)



def integrated_gradients_overlay(
    model: nn.Module,
    x: torch.Tensor,                      
    *,
    target_class: int | None = None,
    steps: int = 128,
    batch_size: int = 32,
    baseline: str = "mean",               
    smooth_samples: int = 8,             
    smooth_sigma: float = 0.02,        
    img_mean=(0.4829306304, 0.4449255764, 0.3957090974),
    img_std =(0.2592383325, 0.2532703280, 0.2598187029),
    alpha: float = 0.45,
    cmap_mode: str = "magma",
    percentile_clip: float = 99.0,       
    border_suppress: float = 0.02,      
    return_heat: bool = False,):

    assert x.dim()==4 and x.shape[0]==1 and x.shape[1]==3
    model.eval()
    device = x.device
    H, W = x.shape[-2:]


    with torch.enable_grad():
        xx = x.clone().detach().requires_grad_(True)
        logits = model(xx)
        probs  = logits.softmax(dim=1)
        if target_class is None:
            target_class = int(probs.argmax(dim=1).item())
        p = float(probs[0, target_class].item())

    base01 = vis_utils._denorm01_np(x, img_mean, img_std) 
    if baseline == "zeros":
        x0 = torch.zeros_like(x)
    elif baseline == "mean":
        x0 = torch.zeros_like(x)
    elif baseline == "black_pixel":
        black01 = np.zeros_like(base01)
        x0 = vis_utils._to_norm(black01, np.array(img_mean), np.array(img_std)).to(device)
    elif baseline == "blurred":
        blr01 = vis_utils._blur_baseline_np(base01, sigma=3.0)
        x0 = vis_utils._to_norm(blr01, np.array(img_mean), np.array(img_std)).to(device)
    else:
        raise ValueError("baseline inválido")

    deltas = x - x0
    ts = torch.linspace(0, 1, steps+1, device=device)[1:]

    attr_accum = torch.zeros_like(x, dtype=torch.float32)

    for s in range(smooth_samples):
        noise = torch.randn_like(x) * smooth_sigma if smooth_samples>1 else 0.0
        grads_sum = torch.zeros_like(x, dtype=torch.float32)

        for i in range(0, steps, batch_size):
            t_b = ts[i:i+batch_size]
            m = t_b.numel()
            xt = (x0 + t_b.view(m,1,1,1) * (deltas + noise)).detach().requires_grad_(True)
            with torch.enable_grad():
                out = model(xt)
                score = out[:, target_class].sum()
                grads = torch.autograd.grad(score, xt, retain_graph=False, create_graph=False)[0]
            grads_sum += grads.mean(0, keepdim=True)

        avg_grads = grads_sum / (steps / batch_size)
        attr_accum += (deltas * avg_grads)

    attributions = (attr_accum / max(1, smooth_samples)) 

    heat = attributions.clamp(min=0).sum(dim=1, keepdim=False)[0]  
    h_np = heat.detach().cpu().numpy().astype(np.float32)
    if border_suppress > 0:
        b = int(round(border_suppress * min(H,W)))
        if b > 0: h_np[:b,:]=0; h_np[-b:,:]=0; h_np[:,:b]=0; h_np[:,-b:]=0
    hi = np.percentile(h_np, percentile_clip)

    h_np = np.clip(h_np/ (hi+1e-8), 0, 1)

    try:
        h_np = gaussian_filter(h_np, sigma=1.0)
        h_np = np.clip(h_np, 0, 1)
    except Exception:
        pass

    # overlay
    base01 = vis_utils._denorm01_np(x, img_mean, img_std)
    color  = vis_utils._cmap_np(h_np, mode=cmap_mode)
    overlay01 = (1 - alpha)*base01 + alpha*color
    overlay_np = (overlay01*255).astype(np.uint8)

    if return_heat:
        return overlay_np, (h_np*255).astype(np.uint8), target_class, p
    return overlay_np



@torch.no_grad()
def occlusion_sensitivity_overlay(
    model: nn.Module,
    x: torch.Tensor,                  
    *,
    target_class: int | None = None,     
    patch: int = 32,                      
    stride: int = 16,                    
    baseline: str = "mean",              
    batch_size: int = 64,                
    img_mean=(0.4829306304, 0.4449255764, 0.3957090974),
    img_std =(0.2592383325, 0.2532703280, 0.2598187029),
    alpha: float = 0.45,
    cmap_mode: str = "magma",
    agg: str = "prob_drop"):

    """
    Devuelve: overlay_np (H,W,3 uint8), heat_np (H,W uint8), class_id, prob_base
    """
    assert x.dim()==4 and x.shape[0]==1 and x.shape[1]==3, f"x debe ser (1,3,H,W), recibido {tuple(x.shape)}"
    model.eval()
    device = x.device
    _, _, H, W = x.shape

    logits0 = model(x)
    probs0  = logits0.softmax(dim=1)
    if target_class is None:
        target_class = int(probs0.argmax(dim=1).item())
    p0 = float(probs0[0, target_class].item())
    s0 = float(logits0[0, target_class].item())

    if baseline in ("mean", "zeros"):
        base_val = 0.0
    elif baseline == "black":
        # negro 
        base_val = 0.0
    else:
        raise ValueError("baseline debe ser 'mean'|'zeros'|'black'")


    ys = list(range(0, max(1, H - patch + 1), stride))
    xs = list(range(0, max(1, W - patch + 1), stride))
    if ys[-1] != H - patch: ys.append(max(0, H - patch))
    if xs[-1] != W - patch: xs.append(max(0, W - patch))
    windows = [(y, x_) for y in ys for x_ in xs]


    heat = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)


    for i in range(0, len(windows), batch_size):
        batch = windows[i:i+batch_size]
        m = len(batch)
        xb = x.repeat(m, 1, 1, 1).clone()

        for k, (yy, xx) in enumerate(batch):
            xb[k, :, yy:yy+patch, xx:xx+patch] = base_val

        out = model(xb)  
        if agg == "prob_drop":
            prob = out.softmax(dim=1)[:, target_class]  
            drop = (p0 - prob).clamp(min=0.0)           
            drop_np = drop.detach().cpu().numpy()
        else:  
            sc = out[:, target_class]
            drop_np = (s0 - sc).clamp(min=0.0).detach().cpu().numpy()

        for k, (yy, xx) in enumerate(batch):
            heat[yy:yy+patch, xx:xx+patch] += drop_np[k]
            count[yy:yy+patch, xx:xx+patch] += 1.0

    mask = count > 0
    heat[mask] = heat[mask] / count[mask]
    if heat.max() > 0:
        heat01 = heat / heat.max()
    else:
        heat01 = heat

    base01 = vis_utils._denorm01_np(x, img_mean, img_std)    
    color  = vis_utils._cmap_np(heat01.astype(np.float32), mode=cmap_mode)
    overlay01 = (1 - alpha)*base01 + alpha*color
    overlay_np = (overlay01*255).astype(np.uint8)
    heat_np    = (heat01*255).astype(np.uint8)

    return overlay_np, heat_np, target_class, p0


def actmax_single(
    model: nn.Module,
    layer_name: str,
    channel: int,
    *,
    img_size: int = 224,
    steps: int = 200,
    lr: float = 0.1,
    tv_weight: float = 0.005,
    l2_weight: float = 0.0005,
    jitter: int = 8,
    img_mean=(0.4829306304, 0.4449255764, 0.3957090974),
    img_std =(0.2592383325, 0.2532703280, 0.2598187029),
    seed: int | None = 0):
    """
    Devuelve una imagen uint8 (H,W,3) que maximiza el canal indicado.
    Actualización manual con autograd.grad -> evita el error de 'no grad'.
    """
    device = next(model.parameters()).device
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)

    x = torch.randn(1,3,img_size,img_size, device=device) * 0.2
    x = x.clamp(-3, 3)

    A = {}
    layer = vis_utils.get_module_by_name(model, layer_name)
    def fwd_hook(m,i,o): A["val"] = o
    h = layer.register_forward_hook(fwd_hook)

    for t in range(steps):
        x = x.detach().requires_grad_(True)

        if jitter > 0:
            ox = torch.randint(-jitter, jitter+1, ()).item()
            oy = torch.randint(-jitter, jitter+1, ()).item()
            x_shift = torch.roll(x, shifts=(oy, ox), dims=(2,3))
        else:
            x_shift = x

        with torch.enable_grad():
            _ = model(x_shift)
            feat = A.get("val", None)
            if feat is None:
                h.remove()
                raise RuntimeError(f"No se capturó activación en '{layer_name}'")

            ch = min(channel, feat.shape[1]-1)
            obj = feat[:, ch].mean()                  
            loss = -(obj) + l2_weight*(x_shift**2).mean() + tv_weight*_tv_loss(x_shift)

            g = torch.autograd.grad(loss, x_shift, retain_graph=False, create_graph=False)[0]

        if jitter > 0:
            g = torch.roll(g, shifts=(-oy, -ox), dims=(2,3))
        with torch.no_grad():
            x -= lr * g.sign()                     
            x.clamp_(-3, 3)

    h.remove()

    out_np = vis_utils._denorm01_np(x, img_mean, img_std)
    return (out_np*255).astype(np.uint8)


def actmax_grid_panel(
    model: nn.Module,
    items,                           
    *,
    k_per_item: int = 1,               
    ncols: int = 5,
    tile_px: int = 224,
    steps: int = 200,
    lr: float = 0.1,
    tv_weight: float = 0.005,
    l2_weight: float = 0.0005,
    jitter: int = 8,
    img_mean=(0.4829306304, 0.4449255764, 0.3957090974),
    img_std =(0.2592383325, 0.2532703280, 0.2598187029),
    title_px: int = 22,
    bg=(10,10,10), fg=(235,235,235),
    pad_out: int = 20, gap_xy: int = 18):
    """
    Si k_per_item == 1: cada celda muestra 1 imagen (canal start_ch).
    Si k_per_item > 1: dentro de la celda se hace submosaico horizontal de k_per_item canales.
    """
    import math
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", title_px)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", title_px)
        except:
            font = ImageFont.load_default()

    tiles, titles = [], []
    for short_title, lname, start_ch in items:
        imgs = []
        for j in range(k_per_item):
            ch = start_ch + j
            img = actmax_single(
                model, lname, ch,
                img_size=tile_px, steps=steps, lr=lr,
                tv_weight=tv_weight, l2_weight=l2_weight, jitter=jitter,
                img_mean=img_mean, img_std=img_std, seed=j 
            )
            imgs.append(Image.fromarray(img))
        if k_per_item == 1:
            tile = imgs[0]
        else:
            w = tile_px * k_per_item
            canvas = Image.new("RGB", (w, tile_px), color=(0,0,0))
            for j, im in enumerate(imgs):
                canvas.paste(im, (j*tile_px, 0))
            tile = canvas
        tiles.append(np.array(tile))
        titles.append(short_title)

    n = len(tiles); ncols = max(1, ncols); nrows = math.ceil(n/ncols)
    title_h = title_px + 6
    cell_w  = tiles[0].shape[1]
    cell_h  = title_h + tiles[0].shape[0]

    W = pad_out + ncols*cell_w + (ncols-1)*gap_xy + pad_out
    H = pad_out + nrows*cell_h + (nrows-1)*gap_xy + pad_out

    panel = Image.new("RGB", (W, H), color=bg)
    draw  = ImageDraw.Draw(panel)

    for idx, (tile_np, text) in enumerate(zip(tiles, titles)):
        r, c = divmod(idx, ncols)
        x0 = pad_out + c*(cell_w + gap_xy)
        y0 = pad_out + r*(cell_h + gap_xy)
        tw = draw.textlength(text, font=font)
        tx = x0 + (cell_w - tw)//2
        draw.text((tx, y0), text, fill=fg, font=font)
        panel.paste(Image.fromarray(tile_np), (x0, y0 + title_h))

    return np.array(panel)