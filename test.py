import os
import torch
from models.team05_IPCV.model import HMA # Ensure the correct model import

def load_model(device):
    model = HMA(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=16,
        interval_size=4,
        img_range=1.0,
        depths=[6, 6, 6, 6,6,6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6,6,6],
        mlp_ratio=2,
        resi_connection='1conv',
        upsampler='pixelshuffledirect'
    )
    
    model_path = "./model_zoo/team05_ipcv.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def process_images(model, device):
    # changes these paths
    data_root = "/content/drive/MyDrive/NTIRE/Test/"
    save_path = "results/"
    os.makedirs(save_path, exist_ok=True)
    
    for img_name in os.listdir(data_root):
        img_path = os.path.join(data_root, img_name)
        img = util.imread_uint(img_path, n_channels=3)
        img = util.uint2tensor4(img, 1.0).to(device)
        
        with torch.no_grad():
            output = model(img)
        output_img = util.tensor2uint(output, 1.0)
        util.imsave(output_img, os.path.join(save_path, img_name))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    process_images(model, device)
    print("Processing complete. Results saved.")

if __name__ == "__main__":
    main()
