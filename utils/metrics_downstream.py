import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import facer

class MetricsBase(object):
    def __init__(self, num_classes, names):
        pass

    def pixel_accuracy(self):
        raise NotImplementedError

    def pixel_accuracy_class(self):
        raise NotImplementedError

    def mean_intersection_over_union(self):
        raise NotImplementedError

    def frequency_weighted_intersection_over_union(self):
        raise NotImplementedError

    def _generate_matrix(self):
        raise NotImplementedError

    def get_table(self):
        raise NotImplementedError

    def add_batch(self, gt, pred):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

class Metrics(MetricsBase):
    def __init__(self, num_classes, names):
        super(Metrics, self).__init__(num_classes, names)
        assert num_classes == len(names)
        self.num_classes = num_classes
        self.names = names
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)

    def pixel_accuracy(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def pixel_accuracy_class(self):
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        acc = np.nanmean(acc)
        return acc

    def mean_intersection_over_union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def frequency_weighted_intersection_over_union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pred_image):
        mask = (gt_image >= 0) & (gt_image < self.num_classes)
        label = self.num_classes * gt_image[mask].astype('int') + pred_image[mask]
        count = np.bincount(label, minlength=self.num_classes ** 2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def get_table(self):
        eps = 1e-4
        total_elem = np.sum(self.confusion_matrix, axis=None)
        tp = np.diag(self.confusion_matrix)
        fp_plus_tp = np.sum(self.confusion_matrix, axis=0)
        fn_plus_tp = np.sum(self.confusion_matrix, axis=1)

        A = (total_elem - (fp_plus_tp + fn_plus_tp - 2 * tp)) / total_elem
        R = tp / (eps + fn_plus_tp)
        P = tp / (eps + fp_plus_tp)
        F1 = 2 * P * R / (eps + P + R)
        IOU = tp / (eps + fp_plus_tp + fn_plus_tp - tp)

        df = pd.DataFrame(data=np.column_stack([IOU, F1, P, R, A]),
                          columns=['IoU', 'F1', 'Prec', 'recall', 'Acc'])

        df = df.round(4)
        df.index = self.names
        total = df.iloc[:, :].mean()
        total_bg = df.iloc[1:, :].mean()
        df.loc['total'] = total
        df.loc['total(-bg)'] = total_bg

        return df

    def add_batch(self, gt_image, pred_image):
        assert gt_image.shape == pred_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pred_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes,) * 2)


def plot_segmentation_overlay(image: torch.Tensor, logits: torch.Tensor, class_index: int = None, threshold: float = 0.5, alpha: float = 0.5, cmap='viridis', name="mask.png", logits_f=False, new_logits=None):
    """
    Plots the segmentation mask overlayed on the original image with a perceptually uniform colormap.
    
    Parameters:
        image (torch.Tensor): The input image tensor of shape (1, 3, H, W).
        logits (torch.Tensor): The raw logits output from the segmentation model of shape (1, 19, H, W).
        class_index (int): The class index for which the mask should be visualized. 
                           If None, the argmax of logits (most likely class) will be used.
        threshold (float): Threshold for binarizing the logits into a mask. Default is 0.5.
        alpha (float): The transparency level for the mask overlay. Default is 0.5.
        cmap (str): Colormap for overlaying the mask, default is 'viridis'.
    """
    # Remove batch dimension for simplicity
    image = image[0] # Shape becomes (3, H, W)
    logits = logits[0] # Shape becomes (19, H, W)
    
    if not logits.shape[0] == 68:
        # Select the appropriate class mask
        if logits_f:
            if class_index is not None:
                mask = torch.sigmoid(logits[class_index])  # Apply sigmoid for probability
                mask = (mask > threshold).float()  # Binarize mask
                print(mask)
            else:
                mask = logits.argmax(dim=0).float()  # Predicted class mask
            mask_np = mask.cpu().numpy()
        else:
            mask_np = logits.cpu().numpy()
            

    # Convert tensors to NumPy arrays for plotting
    image = ((image+1)*0.5).clamp(0,1)
    image_np = image.permute(1, 2, 0).cpu().numpy()  # Shape becomes (H, W, 3)
   

    # Plot the image and overlay mask
    plt.figure(figsize=(2, 2), dpi=300)
    
    plt.imshow(image_np, alpha=alpha)
    if not logits.shape[0] == 68:
        plt.imshow(mask_np, cmap='tab20', alpha=alpha, interpolation='nearest')  # Use perceptually uniform colormap
    else:
        logits = ((logits*255).clamp(0,255).cpu().numpy())

    # Remove axis ticks and labels but keep grid lines
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_xticks([])  # Completely remove x ticks
    plt.gca().set_yticks([])  # Completely remove y ticks

    # Optimize layout
    plt.tight_layout()

    plt.grid(False)

    # Save and close plot
    plt.savefig(name, bbox_inches='tight')  # Use bbox_inches to ensure legend is fully visible
    plt.close()



def plot_land_overlay(image: torch.Tensor, logits: torch.Tensor, class_index: int = None, threshold: float = 0.5, alpha: float = 0.5, cmap='viridis', name="mask.png", logits_f=False, new_logits=None):
    """
    Plots the segmentation mask overlayed on the original image with a perceptually uniform colormap.
    
    Parameters:
        image (torch.Tensor): The input image tensor of shape (1, 3, H, W).
        logits (torch.Tensor): The raw logits output from the segmentation model of shape (1, 19, H, W).
        class_index (int): The class index for which the mask should be visualized. 
                           If None, the argmax of logits (most likely class) will be used.
        threshold (float): Threshold for binarizing the logits into a mask. Default is 0.5.
        alpha (float): The transparency level for the mask overlay. Default is 0.5.
        cmap (str): Colormap for overlaying the mask, default is 'viridis'.
    """
    image = image[0] # Shape becomes (3, H, W)
    logits = logits[0] # Shape becomes (19, H, W)
    
    if not logits.shape[0] == 68:
        # Select the appropriate class mask
        if logits_f:
            if class_index is not None:
                mask = torch.sigmoid(logits[class_index])  # Apply sigmoid for probability
                mask = (mask > threshold).float()  # Binarize mask
            else:
                mask = logits.argmax(dim=0).float()  # Predicted class mask
            mask_np = mask.cpu().numpy()
        else:
            mask_np = logits.cpu().numpy()
            
    image = ((image+1)*0.5).clamp(0,1)
    image_np = image.permute(1, 2, 0).cpu().numpy()  # Shape becomes (H, W, 3)
   
    plt.figure(figsize=(2, 2), dpi=300)

    plt.imshow(image_np, alpha=alpha)
    if not logits.shape[0] == 68:
        plt.imshow(mask_np, cmap='tab20', alpha=alpha)  # Use perceptually uniform colormap
    else:
        logits = ((logits*255).clamp(0,255).cpu().numpy())
   
    plt.scatter(logits[:, 0], logits[:, 1], color="#9A4942", s=7, label='Prediction')
    if new_logits is not None:
        new_logits = new_logits.clamp(0, 255).cpu().numpy()
        plt.scatter(new_logits[:, 0], new_logits[:, 1], color="#6DA7CA", s=7, label='Ground-Truth')

    # Set a grid with slightly denser sampling
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(32))  # Major ticks every 32 pixels
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(32))  # Major ticks every 32 pixels
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(16))  # Minor ticks every 16 pixels
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(16))  # Minor ticks every 16 pixels

    # Enable minor grid lines with a lighter style
    plt.grid(which='minor', linestyle=':', linewidth=0.5)
    

    # Customize legend
    plt.legend(frameon=True, loc='upper left', fontsize=18, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    frame = plt.gca().legend().get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('gray')
    frame.set_linewidth(0.8)
    frame.set_alpha(0.8)  # Semi-transparent legend background

    # Remove axis ticks and labels but keep grid lines
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_xticks([])  # Completely remove x ticks
    plt.gca().set_yticks([])  # Completely remove y ticks

    # Optimize layout
    plt.tight_layout()

    # Save and close plot
    plt.savefig(name, bbox_inches='tight')  # Use bbox_inches to ensure legend is fully visible
    plt.close()


def load_attributes(path, images):
    face_attr = facer.face_attr("farl/celeba/224", device="cuda")
    names = [n.split("/")[-1][:-4] for n in images]
    attributes_gt = []
    images_names = []

    with open(path, 'r') as file:
        for line in file:
            try:
                a = line.split()
                name = a[0][:-4]
                if name in names:
                    images_names.append(name)
                    list_att = a[1:]
                    list_att_float = [float(char) for char in list_att]
                    tensor_list_att = attributes_gt.append(torch.tensor(list_att_float))

            except:
                print("Line with less attributes")		

    return torch.stack(attributes_gt)

def expand_faces(num_items):

    faces = {'points': torch.tensor([[[97.1746, 115.4525],
                                    [163.3413, 116.1134],
                                    [134.1400, 160.9091],
                                    [103.5579, 189.0539],
                                    [154.8323, 189.5556]],
                                    [[97.1746, 115.4525],
                                    [163.3413, 116.1134],
                                    [134.1400, 160.9091],
                                    [103.5579, 189.0539],
                                    [154.8323, 189.5556]]], device='cuda:0'),
                                    'scores': torch.tensor([0.9994, 0.9994], device='cuda:0'),
                                    'image_ids': torch.tensor([0, 1], device='cuda:0')}
    num_repeats = num_items // faces['points'].size(0)
    
    # Repeat 'points' and 'scores' tensors
    expanded_faces = {
        'points': faces['points'].repeat(num_repeats, 1, 1),
        'scores': faces['scores'].repeat(num_repeats)
    }
    
    # Generate sequential 'image_ids' to match the new count
    expanded_faces['image_ids'] = torch.arange(num_items, device=faces['image_ids'].device)
    
    return expanded_faces

def detect_faces(input_image, detect_model):
    # during training mobilenet was problematic so we can just kind of assume the face is centralized, for the ffhq case.
    # faces = face_detector(input_gt) #TODO: fix the real face detection model during training.
    return expand_faces(input_image.shape[0])


