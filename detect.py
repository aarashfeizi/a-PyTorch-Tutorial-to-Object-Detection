from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = 'checkpoint_ssd300_2.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(original_image, min_score, max_overlap, top_k, suppress=None, max_predictions=False):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')
    det_labels = det_labels[0].to('cpu')

    if max_predictions:
        det_scores = det_scores[0].cpu().detach().numpy()

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    if max_predictions:
        all_detections = list(zip(det_labels, det_scores, det_boxes))

        predictions = {}
        for lbl, scr, bx in all_detections:
            if lbl not in predictions.keys():
                predictions[lbl] = [(scr, bx)]
            else:
                if predictions[lbl][0][0] < scr:
                    predictions[lbl] = [(scr, bx)]
                elif predictions[lbl][0][0] == scr:
                    predictions[lbl].append((scr, bx))

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image, []

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)

    crops = []

    # if max_predictions:

    # cropping images
    for k, vs in predictions.items():
        for v in vs:
            box_location = v[1].tolist()
            width = box_location[2] - box_location[0]
            height = box_location[3] - box_location[1]
            box_location = [max(box_location[0] - int(0.13 * width), 0),
                            max(box_location[1] - int(0.13 * height), 0),
                            min(box_location[2] + int(0.13 * width), original_image.size[0]),
                            min(box_location[3] + int(0.13 * height), original_image.size[1])]
            crops.append(original_image.crop(box_location))

    # Suppress specific classes, if needed
    for k, vs in predictions.items():
        for v in vs:

            # Boxes
            box_location = v[1].tolist()
            draw.rectangle(xy=box_location, outline=label_color_map[k])
            draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
                k])  # a second rectangle at an offset of 1 pixel to increase line thickness
            # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
            #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
            # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
            #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

            # Text
            text_size = font.getsize((k + ' ' + str(v[0])).upper())
            text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
            textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                                box_location[1]]
            draw.rectangle(xy=textbox_location, fill=label_color_map[k])
            draw.text(xy=text_location, text=(k + ' ' + str(v[0])).upper(), fill='white',
                      font=font)

    del draw
    # else:
    #
    #     # cropping images
    #     for i in range(det_boxes.size(0)):
    #         if suppress is not None:
    #             if det_labels[i] in suppress:
    #                 continue
    #
    #         box_location = det_boxes[i].tolist()
    #         width = box_location[2] - box_location[0]
    #         height = box_location[3] - box_location[1]
    #         box_location = [max(box_location[0] - int(0.2 * width), 0),
    #                         max(box_location[1] - int(0.2 * height), 0),
    #                         min(box_location[2] + int(0.2 * width), original_image.size[0]),
    #                         min(box_location[3] + int(0.2 * height), original_image.size[1])]
    #         crops.append(original_image.crop(box_location))
    #
    #     # Suppress specific classes, if needed
    #     for i in range(det_boxes.size(0)):
    #         if suppress is not None:
    #             if det_labels[i] in suppress:
    #                 continue
    #
    #         # Boxes
    #         box_location = det_boxes[i].tolist()
    #         draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
    #         draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
    #             det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
    #         # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
    #         #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
    #         # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
    #         #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness
    #
    #         # Text
    #         det_scores_numpy = det_scores[0].cpu().detach().numpy()
    #         text_size = font.getsize((det_labels[i] + str(det_scores_numpy[i])).upper())
    #         text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
    #         textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
    #                             box_location[1]]
    #         draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
    #         draw.text(xy=text_location, text=(det_labels[i] + str(det_scores_numpy[i])).upper(), fill='white',
    #                   font=font)

    #   del draw

    return annotated_image, crops, predictions


if __name__ == '__main__':
    # folder_path = f'/home/mila/f/feiziaar/projects/a-PyTorch-Tutorial-to-Object-Detection/mine/real_images'
    # folder_path = f'/home/mila/f/feiziaar/projects/a-PyTorch-Tutorial-to-Object-Detection/mine/gen_dataset/images_TEST/'
    # folder_path = f'/home/mila/f/feiziaar/projects/a-PyTorch-Tutorial-to-Object-Detection/mine/img_results/'
    # folder_path = f'/home/mila/f/feiziaar/projects/a-PyTorch-Tutorial-to-Object-Detection/mine/img_results/'
    folder_path = f'/home/mila/f/feiziaar/projects/a-PyTorch-Tutorial-to-Object-Detection/model_cropped_2/'
    # folder_path = f'/home/mila/f/feiziaar/projects/a-PyTorch-Tutorial-to-Object-Detection/mine/xy_dataset/images_TEST/'
    files = os.listdir(folder_path)
    files = [f for f in files if f.endswith('png') or f.endswith('.jpeg') or f.endswith('.jpg')]
    for f in files:
        print(f)
        img_path = os.path.join(folder_path, f)
        original_image = Image.open(img_path, mode='r')
        original_image = original_image.convert('RGB')
        # annotaded_image, crops = detect(original_image, min_score=0.999, max_overlap=0.5, top_k=200)
        # annotaded_image.save(f'./model_annotated/{f}')
        # for i, c in enumerate(crops):
        #     c.save(f'./model_cropped/c{i}_{f}')


        annotaded_image, crops, preds = detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200, max_predictions=True)
        annotaded_image.save(f'./model_annotated_points/{f}')
