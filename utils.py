# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Misc. utilities."""
import numpy as np
import scipy.optimize
import tensorflow as tf
from scipy.special import comb

def adjusted_rand_index(true_mask, pred_mask, name='ari_score'):

  r"""Computes the adjusted Rand index (ARI), a clustering similarity score.
  This implementation ignores points with no cluster label in `true_mask` (i.e.
  those points for which `true_mask` is a zero vector). In the context of
  segmentation, that means this function can ignore points in an image
  corresponding to the background (i.e. not to an object).
  Args:
    true_mask: `Tensor` of shape [batch_size, n_points, n_true_groups].
      The true cluster assignment encoded as one-hot.
    pred_mask: `Tensor` of shape [batch_size, n_points, n_pred_groups].
      The predicted cluster assignment encoded as categorical probabilities.
      This function works on the argmax over axis 2.
    name: str. Name of this operation (defaults to "ari_score").
  Returns:
    ARI scores as a tf.float32 `Tensor` of shape [batch_size].
  Raises:
    ValueError: if n_points <= n_true_groups and n_points <= n_pred_groups.
      We've chosen not to handle the special cases that can occur when you have
      one cluster per datapoint (which would be unusual).
  References:
    Lawrence Hubert, Phipps Arabie. 1985. "Comparing partitions"
      https://link.springer.com/article/10.1007/BF01908075
    Wikipedia
      https://en.wikipedia.org/wiki/Rand_index
    Scikit Learn
      http://scikit-learn.org/stable/modules/generated/\
      sklearn.metrics.adjusted_rand_score.html
  """
  with tf.name_scope(name):
    _, n_points, n_true_groups = true_mask.shape.as_list()
    n_pred_groups = pred_mask.shape.as_list()[-1]
    n_pred_groups = pred_mask.shape.as_list()[-1]
    if n_points <= n_true_groups and n_points <= n_pred_groups:
      # This rules out the n_true_groups == n_pred_groups == n_points
      # corner case, and also n_true_groups == n_pred_groups == 0, since
      # that would imply n_points == 0 too.
      # The sklearn implementation has a corner-case branch which does
      # handle this. We chose not to support these cases to avoid counting
      # distinct clusters just to check if we have one cluster per datapoint.
      raise ValueError(
          "adjusted_rand_index requires n_groups < n_points. We don't handle "
          "the special cases that can occur when you have one cluster "
          "per datapoint.")

    true_group_ids = tf.argmax(true_mask, -1)
    pred_group_ids = tf.argmax(pred_mask, -1)
    # We convert true and predicted clusters to one-hot ('oh') representations.
    true_mask_oh = tf.cast(true_mask, tf.float32)  # already one-hot
    pred_mask_oh = tf.one_hot(pred_group_ids, n_pred_groups)  # returns float32

    n_points = tf.cast(tf.reduce_sum(true_mask_oh, axis=[1, 2]), tf.float32)

    nij = tf.einsum('bji,bjk->bki', pred_mask_oh, true_mask_oh)
    a = tf.reduce_sum(nij, axis=1)
    b = tf.reduce_sum(nij, axis=2)

    rindex = tf.reduce_sum(nij * (nij - 1), axis=[1, 2])
    aindex = tf.reduce_sum(a * (a - 1), axis=1)
    bindex = tf.reduce_sum(b * (b - 1), axis=1)
    expected_rindex = aindex * bindex / (n_points*(n_points-1))
    max_rindex = (aindex + bindex) / 2
    ari = (rindex - expected_rindex) / (max_rindex - expected_rindex)

    # The case where n_true_groups == n_pred_groups == 1 needs to be
    # special-cased (to return 1) as the above formula gives a divide-by-zero.
    # This might not work when true_mask has values that do not sum to one:
    both_single_cluster = tf.logical_and(
        _all_equal(true_group_ids), _all_equal(pred_group_ids))
    return tf.where(both_single_cluster, tf.ones_like(ari), ari)

def l2_loss(prediction, target):
  return tf.reduce_mean(tf.math.squared_difference(prediction, target))


def hungarian_huber_loss(x, y):
  """Huber loss for sets, matching elements with the Hungarian algorithm.

  This loss is used as reconstruction loss in the paper 'Deep Set Prediction
  Networks' https://arxiv.org/abs/1906.06565, see Eq. 2. For each element in the
  batches we wish to compute min_{pi} ||y_i - x_{pi(i)}||^2 where pi is a
  permutation of the set elements. We first compute the pairwise distances
  between each point in both sets and then match the elements using the scipy
  implementation of the Hungarian algorithm. This is applied for every set in
  the two batches. Note that if the number of points does not match, some of the
  elements will not be matched. As distance function we use the Huber loss.

  Args:
    x: Batch of sets of size [batch_size, n_points, dim_points]. Each set in the
      batch contains n_points many points, each represented as a vector of
      dimension dim_points.
    y: Batch of sets of size [batch_size, n_points, dim_points].

  Returns:
    Average distance between all sets in the two batches.
  """
  pairwise_cost = tf.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)(
      tf.expand_dims(y, axis=-2), tf.expand_dims(x, axis=-3))
  indices = np.array(
      list(map(scipy.optimize.linear_sum_assignment, pairwise_cost)))

  transposed_indices = np.transpose(indices, axes=(0, 2, 1))

  actual_costs = tf.gather_nd(
      pairwise_cost, transposed_indices, batch_dims=1)

  return tf.reduce_mean(tf.reduce_sum(actual_costs, axis=1))


def average_precision_clevr(pred, attributes, distance_threshold):
  """Computes the average precision for CLEVR.

  This function computes the average precision of the predictions specifically
  for the CLEVR dataset. First, we sort the predictions of the model by
  confidence (highest confidence first). Then, for each prediction we check
  whether there was a corresponding object in the input image. A prediction is
  considered a true positive if the discrete features are predicted correctly
  and the predicted position is within a certain distance from the ground truth
  object.

  Args:
    pred: Tensor of shape [batch_size, num_elements, dimension] containing
      predictions. The last dimension is expected to be the confidence of the
      prediction.
    attributes: Tensor of shape [batch_size, num_elements, dimension] containing
      ground-truth object properties.
    distance_threshold: Threshold to accept match. -1 indicates no threshold.

  Returns:
    Average precision of the predictions.
  """

  [batch_size, _, element_size] = attributes.shape
  [_, predicted_elements, _] = pred.shape

  def unsorted_id_to_image(detection_id, predicted_elements):
    """Find the index of the image from the unsorted detection index."""
    return int(detection_id // predicted_elements)

  flat_size = batch_size * predicted_elements
  flat_pred = np.reshape(pred, [flat_size, element_size])
  sort_idx = np.argsort(flat_pred[:, -1], axis=0)[::-1]  # Reverse order.

  sorted_predictions = np.take_along_axis(
      flat_pred, np.expand_dims(sort_idx, axis=1), axis=0)
  idx_sorted_to_unsorted = np.take_along_axis(
      np.arange(flat_size), sort_idx, axis=0)

  def process_targets(target):
    """Unpacks the target into the CLEVR properties."""
    coords = target[:3]
    object_size = tf.argmax(target[3:5])
    material = tf.argmax(target[5:7])
    shape = tf.argmax(target[7:10])
    color = tf.argmax(target[10:18])
    real_obj = target[18]
    return coords, object_size, material, shape, color, real_obj

  true_positives = np.zeros(sorted_predictions.shape[0])
  false_positives = np.zeros(sorted_predictions.shape[0])

  detection_set = set()

  for detection_id in range(sorted_predictions.shape[0]):
    # Extract the current prediction.
    current_pred = sorted_predictions[detection_id, :]
    # Find which image the prediction belongs to. Get the unsorted index from
    # the sorted one and then apply to unsorted_id_to_image function that undoes
    # the reshape.
    original_image_idx = unsorted_id_to_image(
        idx_sorted_to_unsorted[detection_id], predicted_elements)
    # Get the ground truth image.
    gt_image = attributes[original_image_idx, :, :]

    # Initialize the maximum distance and the id of the groud-truth object that
    # was found.
    best_distance = 10000
    best_id = None

    # Unpack the prediction by taking the argmax on the discrete attributes.
    (pred_coords, pred_object_size, pred_material, pred_shape, pred_color,
     _) = process_targets(current_pred)

    # Loop through all objects in the ground-truth image to check for hits.
    for target_object_id in range(gt_image.shape[0]):
      target_object = gt_image[target_object_id, :]
      # Unpack the targets taking the argmax on the discrete attributes.
      (target_coords, target_object_size, target_material, target_shape,
       target_color, target_real_obj) = process_targets(target_object)
      # Only consider real objects as matches.
      if target_real_obj:
        # For the match to be valid all attributes need to be correctly
        # predicted.
        pred_attr = [pred_object_size, pred_material, pred_shape, pred_color]
        target_attr = [
            target_object_size, target_material, target_shape, target_color]
        match = pred_attr == target_attr
        if match:
          # If a match was found, we check if the distance is below the
          # specified threshold. Recall that we have rescaled the coordinates
          # in the dataset from [-3, 3] to [0, 1], both for `target_coords` and
          # `pred_coords`. To compare in the original scale, we thus need to
          # multiply the distance values by 6 before applying the norm.
          distance = np.linalg.norm((target_coords - pred_coords) * 6.)

          # If this is the best match we've found so far we remember it.
          if distance < best_distance:
            best_distance = distance
            best_id = target_object_id
    if best_distance < distance_threshold or distance_threshold == -1:
      # We have detected an object correctly within the distance confidence.
      # If this object was not detected before it's a true positive.
      if best_id is not None:
        if (original_image_idx, best_id) not in detection_set:
          true_positives[detection_id] = 1
          detection_set.add((original_image_idx, best_id))
        else:
          false_positives[detection_id] = 1
      else:
        false_positives[detection_id] = 1
    else:
      false_positives[detection_id] = 1
  accumulated_fp = np.cumsum(false_positives)
  accumulated_tp = np.cumsum(true_positives)
  recall_array = accumulated_tp / np.sum(attributes[:, :, -1])
  precision_array = np.divide(accumulated_tp, (accumulated_fp + accumulated_tp))

  return compute_average_precision(
      np.array(precision_array, dtype=np.float32),
      np.array(recall_array, dtype=np.float32))


def compute_average_precision(precision, recall):
  """Computation of the average precision from precision and recall arrays."""
  recall = recall.tolist()
  precision = precision.tolist()
  recall = [0] + recall + [1]
  precision = [0] + precision + [0]

  for i in range(len(precision) - 1, -0, -1):
    precision[i - 1] = max(precision[i - 1], precision[i])

  indices_recall = [
      i for i in range(len(recall) - 1) if recall[1:][i] != recall[:-1][i]
  ]

  average_precision = 0.
  for i in indices_recall:
    average_precision += precision[i + 1] * (recall[i + 1] - recall[i])
  return average_precision

def _all_equal(values):
  """Whether values are all equal along the final axis."""
  return tf.reduce_all(tf.equal(values, values[..., :1]), axis=-1)
