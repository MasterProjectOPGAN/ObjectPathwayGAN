import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

import pickle
from miscc.utils import weights_init, load_params, copy_G_params
import torch
from miscc.config import cfg
from datasets import prepare_data
from model import RNN_ENCODER, CNN_ENCODER
from torch.autograd import Variable
import model
import torchvision.transforms as transforms
from datasets import TextDataset
from PIL import Image
from tqdm import tqdm
import numpy as np
import glob
from flask import Flask, request, send_from_directory

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='./static'))

netG = model.G_NET()
netG.apply(weights_init)
netG.to()
state_dict = torch.load('../output/coco_glu-gan2_2021_04_06_19_01_43_9609/Model/checkpoint_0019.pth', map_location=lambda storage, loc: storage)
#state_dict = torch.load('../models/op-gan.pth', map_location=lambda storage, loc: storage)

netG.load_state_dict(state_dict["netG"])
for p in netG.parameters():
    p.requires_grad = False
netG.to('cuda:0')
netG.eval()


        #text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
cfg.DEVICE = 'cuda:0'
print(cfg)

text_encoder = RNN_ENCODER(27297, nhidden=cfg.TEXT.EMBEDDING_DIM)
state_dict = torch.load('../models/coco/text_encoder100.pth', map_location=lambda storage, loc: storage)
text_encoder.load_state_dict(state_dict)
text_encoder = text_encoder.to('cuda:0')
text_encoder.eval()

print(text_encoder)

batch_size = 2
nz = cfg.GAN.GLOBAL_Z_DIM
noise = Variable(torch.FloatTensor(batch_size, nz)).to('cuda:0')
local_noise = Variable(torch.FloatTensor(batch_size, cfg.GAN.LOCAL_Z_DIM)).to('cuda:0')

max_objects = 3
image_transform = transforms.Compose([
                transforms.Resize((268, 268)),
                            transforms.ToTensor()])
dataset = TextDataset('../data/', 'test/val2014', 'test', base_size=cfg.TREE.BASE_SIZE,
                                transform=image_transform, eval=True, use_generated_bboxes=True)

assert dataset
dataset_to_load = (
                    torch.utils.data.Subset(dataset, list(range(cfg.DEBUG_NUM_DATAPOINTS))) if cfg.DEBUG else dataset
                            )
dataloader = torch.utils.data.DataLoader(dataset_to_load, batch_size=2,
                                                         drop_last=True, shuffle=False, num_workers=int(cfg.WORKERS))

noise.data.normal_(0, 1)
local_noise.data.normal_(0, 1)


#s_tmp = model_dir[:model_dir.rfind('.pth')].split("/")[-1]
#save_dir = '%s/%s/%s' % ("../output", s_tmp, split_dir)
#mkdir_p(save_dir)

save_dir = './static'

data_iter = iter(dataloader)

path = Path(__file__).parent

#export_file_url = 'https://www.dropbox.com/s/6bgq8t6yextloqp/export.pkl?raw=1'
#export_file_name = 'export.pkl'
#
#classes = ['black', 'grizzly', 'teddys']
#path = Path(__file__).parent
#
#app = Starlette()
#app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
#app.mount('/static', StaticFiles(directory='app/static'))
#
#
#async def download_file(url, dest):
#    if dest.exists(): return
#    async with aiohttp.ClientSession() as session:
#        async with session.get(url) as response:
#            data = await response.read()
#            with open(dest, 'wb') as f:
#                f.write(data)
#
#
#async def setup_learner():
#    await download_file(export_file_url, path / export_file_name)
#    try:
#        learn = load_learner(path, export_file_name)
#        return learn
#    except RuntimeError as e:
#        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
#            print(e)
#            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
#            raise RuntimeError(message)
#        else:
#            raise
#
#
#loop = asyncio.get_event_loop()
#tasks = [asyncio.ensure_future(setup_learner())]
#learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
#loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/file')
async def getfile(path):
    return send_from_directory('js', path)


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    #img_data = await request.form()
    #img_bytes = await (img_data['file'].read())
    #img = open_image(BytesIO(img_bytes))
    #prediction = learn.predict(img)[0]
    i=0
    print('analyze...')
    for step in tqdm(range(1)):
        if i>0:
            break
        data = data_iter.next()
        imgs, captions, cap_lens, class_ids, keys, transformation_matrices, label_one_hot, _ = prepare_data(
                                    data, eval=True)
        transf_matrices = transformation_matrices[0]
        transf_matrices_inv = transformation_matrices[1]
        hidden = text_encoder.init_hidden(2)
        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
        mask = (captions == 0)
        num_words = words_embs.size(2)
        if mask.size(1) > num_words:
            mask = mask[:, :num_words]
        inputs = (noise, local_noise, sent_emb, words_embs, mask, transf_matrices, transf_matrices_inv, label_one_hot, max_objects)
        inputs = tuple((inp.to(cfg.DEVICE) if isinstance(inp, torch.Tensor) else inp) for inp in inputs)
        with torch.no_grad():
            fake_imgs, _, mu, logvar = netG(*inputs)
        for batch_idx, j in enumerate(range(batch_size)):
            s_tmp = '%s/%s' % (save_dir, keys[j])
            k = -1
            im = fake_imgs[k][j].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            fullpath = '%s_s%d.png' % (s_tmp, step*batch_size+batch_idx)
            im.save(fullpath)
        i+=1
        print('done...')
    return JSONResponse({'result': fullpath})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=8000, log_level="info")
