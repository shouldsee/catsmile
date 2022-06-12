
import haiku as hk
import jax.numpy as jnp
import jax

def load_dataset(
    split: str,
    *,
    is_training: bool,
    batch_size: int,
):
  """Loads the dataset as a generator of batches."""
  ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
  if is_training:
    ds = ds.shuffle(10 * batch_size, seed=0)
  ds = ds.batch(batch_size)
  return iter(tfds.as_numpy(ds))

input_dataset = load_dataset
train = load_dataset("train", is_training=True, batch_size=100)

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

rng = jax.random.PRNGKey(42)
dummy_images, dummy_labels = next(input_dataset)
params = loss_fn_t.init(rng, dummy_images, dummy_labels)

def update_rule(param, update):
  return param - 0.01 * update

for images, labels in input_dataset:
  grads = jax.grad(loss_fn_t.apply)(params, images, labels)
  params = jax.tree_map(update_rule, params, grads)
