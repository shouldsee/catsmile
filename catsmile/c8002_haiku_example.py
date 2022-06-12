
import os,shutil,sys
import haiku as hk
import jax.numpy as jnp
import jax
# import tensorflow_datasets as tfds
import numpy as np
from pprint import pprint

def load_dataset(
    split: str,
    is_training: bool,
    batch_size: int,
):
  PKL = __file__+'.npy'
  if not os.path.exists(PKL):
      from keras.datasets import mnist
      v = mnist.load_data()
      np.save(PKL+'.temp.npy',v)
      shutil.move(PKL+'.temp.npy',PKL)
  else:
      v = np.load(PKL,allow_pickle=True)

  (x_train, y_train), (x_test, y_test) = v
  x   = x_train
  y   = y_train
  x   = x.reshape((-1,28**2))
  y   = y[:,None]
  if is_training:
      idx = np.random.permutation(range(len(x)))
      x   = x[idx]
      y   = y[idx]
  x   = jnp.array(x,dtype='float')
  y   = jnp.array(y,dtype='int8')

  def giter(it=(x,y),batch_size=batch_size):
      x,y = it
      L = len(x)
      i=-1
      while True:
          i+=1
          i = i %(L//batch_size)
          idx = slice(i*batch_size,(i+1)*batch_size)
          tup = x[idx],y[idx]
          yield tup
  return giter()


def softmax_cross_entropy(logits, labels):
  one_hot = jax.nn.one_hot(labels, logits.shape[-1])
  return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)

def loss_fn(images, labels):
  mlp = hk.Sequential([
      hk.Linear(300), jax.nn.relu,
      hk.Linear(100), jax.nn.relu,
      hk.Linear(10),
  ])
  logits = mlp(images)
  return jnp.mean(softmax_cross_entropy(logits, labels))

#### core routine for training
loss_fn_t = hk.transform(loss_fn)
loss_fn_t = hk.without_apply_rng(loss_fn_t)

input_dataset = load_dataset("train", is_training=True, batch_size=100)
rng = jax.random.PRNGKey(42)
dummy_images, dummy_labels = next(input_dataset)
params = loss_fn_t.init(rng, dummy_images, dummy_labels)

def update_rule(param, update):
  return param - 0.001 * update

max_iter = 100
print_interval = 10
i = -1
for images, labels in input_dataset:
  i+= 1
  grads = jax.grad(loss_fn_t.apply)(params, images, labels)
  params = jax.tree_map(update_rule, params, grads)
  loss = loss_fn_t.apply(params,images,labels)
  if i%print_interval==0:
      print(f'[B{i}]loss={loss:.3f}')
  if i>= max_iter:
      break
