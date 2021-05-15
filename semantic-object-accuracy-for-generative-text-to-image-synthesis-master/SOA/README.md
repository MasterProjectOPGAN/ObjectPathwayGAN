# Details for calculating the SOA Scores

## Below code snippet is the formula to convert the caption idx to image index of corresponding object category. 
## This logic is placed inside the __getitem__ function of TextDataSet class
To load the captions: load a caption file, get the captions, and generate images:

We used ``captions.pickle`` file from the [AttnGAN](https://github.com/taoxugit/AttnGAN) and their dataloader you can use the provided ``idx`` to load the file directly from the file:

```python
import pickle
import my_model

# load the AttnGAN captions file
with open(captions.pickle, "rb") as f:
    attngan_captions = pickle.load(f)
test_captions = attngan_captions[1]

# load the caption file
with open(label_01_bicycle.pkl, "rb") as f:
    captions = pickle.load(f)

    def __getitem__(self, index):
        #
        from random import randrange
        if not cfg.TRAIN.FLAG and cfg.TRAIN.DEPLOY_FLAG:
            with open(cfg.TRAIN.CAPTION_PATH+cfg.CURRENT_LABEL+".pkl", "rb") as f:
                captions = pickle.load(f)
        new_ix = randrange(len(captions)-1)
        new_idx = 1
        current_caption_idx = captions[new_ix]["idx"]
        new_sent_ix = current_caption_idx[0]*5+current_caption_idx[1]

        key = self.filenames[current_caption_idx[0]]
        index = current_caption_idx[0]
        cls_id = self.class_id[index]
```
 
