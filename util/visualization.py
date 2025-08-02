# -----------------------------------------------------------------------
# S2Tab official code : util/visualization.py
# -----------------------------------------------------------------------
from PIL import Image, ImageDraw
import imageio, glob
import torchvision

def draw_cell(res, img_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    color_map = {1: 'blue', 2: 'green', 3: 'red', 4: 'darkviolet',
                 5: 'skyblue', 6: 'lime', 7: 'pink', 8: 'violet'}
    color_map[0] = 'gray'

    if 'cross_attn' in list(res.values())[0].keys():
        attn_output_dir = output_dir.parent / 'attn'
        attn_output_dir.mkdir(parents=True, exist_ok=True)
    if 'self_attn' in list(res.values())[0].keys():
        self_attn_output_dir = output_dir.parent / 'self_attn'
        self_attn_output_dir.mkdir(parents=True, exist_ok=True)
    if 'segm_map' in list(res.values())[0].keys():
        segm_map_output_dir = output_dir.parent / 'segm_map'
        segm_map_output_dir.mkdir(parents=True, exist_ok=True)

    assert img_dir.exists()

    for k, v in res.items():
        boxes = v['boxes'] # (x,y,w,h)
        labels = v['labels'].cpu().numpy()
        if 'all_boxes' in v:
            all_boxes = v['all_boxes']
        else:
            all_boxes = []

        if 'tokens' in v:
            tokens = v['tokens']
        else:
            tokens = [None] * len(boxes)

        img = Image.open(img_dir / v['filename']).convert('RGB')
        draw = ImageDraw.Draw(img)

        if 'cross_attn' in v.keys():
            attns = v['cross_attn'] # ATTN VIS
        if 'self_attn' in v.keys():
            self_attns = v['self_attn'] # ATTN VIS

        if len(boxes) == 0:
            continue
        valid_boxes = [(box, label) for box, label in zip(boxes, labels)]
        box_coord = boxes.copy()
        box_coord[:, 2:] += box_coord[:, :2]

        init_color = (255, 0, 0)
        fin_color = (0, 0, 255)
        num = len(valid_boxes)
        if num > 0:
            step = 255 / num
        else:
            step = 0

        for i in range(num):
            box, _ = valid_boxes[i]
            box = list(map(int, box))
            if i == num-1:
                fill_color = fin_color
            else:
                fill_color = tuple(map(int, (init_color[0]-step*i, init_color[1], init_color[2]+step*i)))
            draw.rectangle((box[0], box[1], box[0] + box[2], box[1] + box[3]), fill=None, outline=(0,0,255), width=2)
            # if tokens[i] is not None:
            #     try:
            #         draw.text((box[0], box[1]), tokens[i], fill=(0,0,0))
            #     except:
            #         draw.text((box[0], box[1]), tokens[i].encode('utf-8').decode('iso-8859-1'), fill=(0,0,0))

        for j in range(len(all_boxes)):
            box = all_boxes[j]
            if 'cross_attn' in v.keys():
                # ATTN VIS
                img_attn = Image.open(img_dir / v['filename'])
                draw_attn = ImageDraw.Draw(img_attn)
                draw_attn.rectangle((box[0], box[1], box[2], box[3]), fill=None, outline=fill_color, width=2)
                max_rgb = attns[j].max(1)[0].max(1)[0]
                attn = attns[j] / (max_rgb[:, None, None] + 1e-15)
                attn = torchvision.transforms.functional.to_pil_image(attn)
                ################# shared mode ##############
                # draw_attn.rectangle((box[0], box[1], box[0] + box[2], box[1] + box[3]), fill=None, outline=fill_color, width=2)
                # attn = np.asarray(attns[i])
                # attn = (attn / (np.max(attn)+1e-15) * 255).astype(np.uint8)
                # attn = cv2.applyColorMap(attn, cv2.COLORMAP_HSV)
                # attn = Image.fromarray(attn)
                ###############################################
                attn = attn.resize(img.size)
                attn = attn.convert('RGB')
                blend = Image.blend(img_attn, attn, 0.3)
                blend.save(attn_output_dir / v['filename'].replace('.png', f'_{j}.png').replace('.jpg', f'_{j}.jpg'))
                
            if 'self_attn' in v.keys():
                img_attn = Image.open(img_dir / v['filename'])
                draw_attn = ImageDraw.Draw(img_attn)
                img_self_attn = Image.new('RGB', img.size)
                draw_self_attn = ImageDraw.Draw(img_self_attn)
                self_attn = self_attns[j]
                self_attn = self_attn / self_attn.max(0)[0][None,:]
                box_j = all_boxes[j]
                for k, val in enumerate(self_attn): # val: r,g,b
                    box_k = all_boxes[k]
                    color = tuple((255 * val.to(int)).tolist())
                    draw_self_attn.rectangle((box_k[0], box_k[1], box_k[2], box_k[3]), fill=color, outline=None, width=2)
                draw_attn.rectangle((box_j[0], box_j[1], box_j[2], box_j[3]), fill=None, outline=(0,0,255), width=2)
                blend = Image.blend(img_attn, img_self_attn, 0.3)
                blend.save(self_attn_output_dir / v['filename'].replace('.png', f'_{j}.png').replace('.jpg', f'_{j}.jpg'))

        img.save(output_dir / v['filename'])


def make_gif(name, output_dir):
    paths = glob.glob(f'{name}*')
    paths = sorted(paths, key=lambda x: int(x.split('_')[-1].replace('.png','')))
    paths = [Image.open(x) for x in paths]
    imageio.mimsave(f'{output_dir}/{name}.gif', paths, fps=2)