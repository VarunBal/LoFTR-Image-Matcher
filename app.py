from flask import Flask, render_template, request, jsonify
import numpy as np
import kornia
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from time import time

app = Flask(__name__)

ts = time()
loft_outdoor = kornia.feature.LoFTR(pretrained='outdoor')
loft_indoor = kornia.feature.LoFTR(pretrained='indoor')
loft_indoor_new = kornia.feature.LoFTR(pretrained='indoor_new')
print(f"Total time to load models: {time()-ts}")

loft_models = {'outdoor': loft_outdoor, 'indoor': loft_indoor, 'indoor_new': loft_indoor_new}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/match', methods=['POST'])
def match_images():
    # Uploaded images
    img1_raw = request.files['image1']
    img2_raw = request.files['image2']

    # Get the selected pretrained model
    pretrained_model = request.form['model']

    output_img_path = loftr_matching(img1_raw, img2_raw, pretrained_model)

    return render_template('result.html', image_path=output_img_path)


@app.route('/match_api', methods=['POST'])
def match_api():
    # Uploaded images
    img1_raw = request.files['image1']
    img2_raw = request.files['image2']

    # Get the selected pretrained model
    pretrained_model = request.form['model']

    output_img_path = loftr_matching(img1_raw, img2_raw, pretrained_model)

    output_img_url = request.host_url + output_img_path

    return jsonify(output_img_url=output_img_url)


def loftr_matching(img1_raw, img2_raw, pretrained_model='output'):

    # Load input images
    img1 = np.asarray(Image.open(img1_raw).convert('L'), dtype=np.uint8)
    img2 = np.asarray(Image.open(img2_raw).convert('L'), dtype=np.uint8)

    # Convert images to tensors
    img1_tensor = kornia.image_to_tensor(img1/255).unsqueeze(0).float()
    img2_tensor = kornia.image_to_tensor(img2/255).unsqueeze(0).float()

    # Initialize LOFTR model
    loftr = loft_models[pretrained_model]

    input = {"image0": img1_tensor, "image1": img2_tensor}

    ts = time()
    matches = loftr(input)
    print(f"Forward pass time: {time()-ts}")
    # print(matches)

    # Create colour map
    color = cm.jet(matches['confidence'], alpha=0.7)

    # Save the output image
    output_img_path = 'static/output.jpg'

    # Print text on image
    text = [
        'LoFTR',
        'Matches: {}'.format(len(matches['keypoints0'])),
    ]

    # Visualize matches
    make_matching_figure(
        img1,
        img2,
        matches['keypoints0'], matches['keypoints1'], color,
        text=text, path=output_img_path)

    return output_img_path


def make_matching_figure(img0, img1, mkpts0, mkpts1, color, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    # Draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                             (fkpts0[i, 1], fkpts1[i, 1]),
                                             transform=fig.transFigure, c=color[i], linewidth=1)
                     for i in range(len(mkpts0))]

        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # Put texts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # Save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig


if __name__ == '__main__':
    app.run(debug=True)
