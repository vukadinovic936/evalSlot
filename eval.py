import clevr as clevr
import tensorflow as tf
from data import build_test_iterator
import numpy as np
import sys
from utils import adjusted_rand_index
import numpy as np
from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# When executing this notebook out of a subfolder, use the command below to
# change to the project's root folder (required for imports):
# %cd ..
import data as data_utils
import model as model_utils
from tqdm import tqdm


def load_model(checkpoint_dir, num_slots=11, num_iters=3, batch_size=16):
  resolution = (128, 128)
  model = model_utils.build_model(
      resolution, batch_size, num_slots, num_iters,
      model_type="object_discovery")

  ckpt = tf.train.Checkpoint(network=model)
  ckpt_manager = tf.train.CheckpointManager(
      ckpt, directory=checkpoint_dir, max_to_keep=5)

  if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    logging.info("Restored from %s", ckpt_manager.latest_checkpoint)

  return model

def renormalize(x):
  """Renormalize from [-1, 1] to [0, 1]."""
  return x / 2. + 0.5

def get_prediction(model, batch, idx=0):
  recon_combined, recons, masks, slots = model(batch["image"])
  image = renormalize(batch["image"])[idx]
  recon_combined = renormalize(recon_combined)[idx]
  recons = renormalize(recons)[idx]
  masks = masks[idx]
  return image, recon_combined, recons, masks, slots

if __name__ == "__main__":

    tf_records_path = 'clevr_with_masks_train.tfrecords'
    batch_size = 1
    data_iterator = build_test_iterator(batch_size = batch_size, apply_crop=True)

    score = 1
    ##Build dataset iterators, optimizers and model.
    ckpt_path = "/home/ubuntu/slot_attention/slot_attention/pretrained"
    model = load_model(ckpt_path, num_slots=7, num_iters=3, batch_size=1)

    ## TODO WRITE TO A FILE ON 100 iterations
    score = np.array([])
    # evaluate on 70k images
    for i in tqdm(range(70000)):

      batch = next(data_iterator)
      image, recon_combined, recons, pred_masks, slots = get_prediction(model, batch)
      pred_masks = tf.squeeze(pred_masks)
      pred_mask = tf.zeros((128,128),tf.int32)

      pred_mask = tf.zeros((128,128),tf.int32)
      pred_mask = tf.math.argmax(pred_masks,0)

      #for i in range(len(pred_masks)):
      #    pred_mask += (i+1) * (tf.cast(pred_masks[i]>0.65, tf.int32))

      mask = batch['mask']
      p_mask = pred_mask

      ## batch one and for each point we have background and object
      mask = tf.one_hot(mask, depth= len(pred_masks))
      mask = tf.reshape(mask,[batch_size,128*128,len(pred_masks)])
      
      p_mask = tf.one_hot(p_mask, depth= len(pred_masks))
      p_mask = tf.reshape(p_mask,[batch_size,128*128,len(pred_masks)])
    
      score = np.append(score, adjusted_rand_index(mask[..., 1:],p_mask))

      if(i%1000==0):
        f = open('log.txt','a')
        temp = f.write(f"it {i}:                                 {np.nanmean(score)}\n")
        f.close()

    print(np.nanmean(score))
