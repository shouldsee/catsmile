#! https://zhuanlan.zhihu.com/p/500856805
# 8800: PyTorch调试小工具 20220418

<http://catsmile.info/8800-pytorch-debug-utils.html>

## 前言:

顺手的工具可以极大地提高生产效率.本篇希望能分享一些实用的模型调试工具

是个人做pytorch实验时积累的一些小工具

## 用grad函数手动计算梯度

```python
# !TEST! !RUN!
def _stub_grad():
    x = x1 ['embeddings.word_embeddings' ]
    y = x1[ 'encoder.layer.9.output']
    del x1

    ### 对于非标量的y,设置需要VJP迭代的初始向量
    grad_outputs = torch.ones_like(y)

    ### 根据计算图给出梯度
    xo = torch.autograd.grad(y,x,retain_graph=True,grad_outputs=grad_outputs)[0]

    print(xo.shape)
```

## 基于回调钩子的调试 Debugging by inserting callback hooks.

### 朴素的hook

```python
def hook(mod, input,output,):
    print(mod.__class__,input.shape, output.shape)
    return output
model.register_forward_hook(hook)
```

### 层级化增加hook

因为模型的层级对于组分模块并不可见,所以需要在修饰时注入全局信息

```python

def add_hook_for_output_tensor(model,tokenizer,CONFIG_EXPAND_LEVEL,CONFIG_DETACH, CONFIG_PRINT_CLASS):
    '''
    Adds to model a method  `model.get_tensors(sent,xdict)`
    which returns a dict of tensors outputted at each nn.Module with maximum depth of CONFIG_EXPAND_LEVEL.
    '''
    xdict = OrderedDict()

    def get_forward_hook(name,xdict=xdict):
        def hook(model, input, output):
            recur_detach(xdict,name,output,CONFIG_DETACH,'/')
            # xdict[name] = output
        return hook

    '注入灵魂'
    for k,v in model.named_modules():
        if k.count('.')<=CONFIG_EXPAND_LEVEL:
            if CONFIG_PRINT_CLASS:
                print(fws(k,60) ,v.__class__);
            v.register_forward_hook(get_forward_hook(k))

    def get_all_tensors(sent):
        '''
        Convert a sentence to a dict of internal representations
        '''
        xdict.clear()
        inputs = tokenizer(sent, return_tensors="pt")
        outputs = model(**inputs)
        return xdict.copy()
    model.get_all_tensors = get_all_tensors

    return model

def recur_detach(xdict,name,output,detach=True,sep='/'):
    '''
    Recursively detach the Tensor and store into xdict.

    '''
    if isinstance(output,tuple):
        for k,v in enumerate(output):
            recur_detach(xdict, name+sep+str(k), v, detach,sep)
    elif isinstance(output,torch.Tensor):
        if detach:
            output = output.detach()
        xdict[name] = output
    elif isinstance(output,dict):
        for k,v in output.items():
            recur_detach(xdict,name+sep+str(k),v, detach,sep)
    else:
        print(name,output.__class__)
        assert 0,'Unknown type %s'%output.__class__
        # output=output.detach()
    return output
```

## 用自动微分算一些不是梯度的东西

Trying to hack `torch.autograd.grad` to calculate Frobenius Norm of Jacobian, proved impossible,

由于`backprop`并不维护一个二阶的形式,可以理解为在不断的回传过程中,只存在一阶的状态形式,
因此,尽管`backprop`通过计算图高效地完成了基于`Jacobian Product`的回传,朴素的`backprop`
并不允许直观地改造得出`\sum_{ij}|J_{ij}|^2`.要想高效地估计FrobeniusNorm,意味着要试着去扩展
`backprop`.



Call chain

```
torch.autograd.grad -> autograd.variable.Variable._execution_engine.run_backward
```

```python
'''
# Excerpt from https://github.com/pytorch/pytorch/blob/master/torch/autograd/__init__.py
# at https://github.com/pytorch/pytorch/commit/0a1bc5f501bb571b6f4275b6ca863e68bf6cda02
'''

def grad(
    outputs: _TensorOrTensors,
    inputs: _TensorOrTensors,
    grad_outputs: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    only_inputs: bool = True,
    allow_unused: bool = False,
    is_grads_batched: bool = False
) -> Tuple[torch.Tensor, ...]:
    r"""Computes and returns the sum of gradients of outputs with respect to
    the inputs.
    ``grad_outputs`` should be a sequence of length matching ``output``
    containing the "vector" in vector-Jacobian product, usually the pre-computed
    gradients w.r.t. each of the outputs. If an output doesn't require_grad,
    then the gradient can be ``None``).
    .. note::
        If you run any forward ops, create ``grad_outputs``, and/or call ``grad``
        in a user-specified CUDA stream context, see
        :ref:`Stream semantics of backward passes<bwd-cuda-stream-semantics>`.
    .. note::
        ``only_inputs`` argument is deprecated and is ignored now (defaults to ``True``).
        To accumulate gradient for other parts of the graph, please use
        ``torch.autograd.backward``.
    Args:
        outputs (sequence of Tensor): outputs of the differentiated function.
        inputs (sequence of Tensor): Inputs w.r.t. which the gradient will be
            returned (and not accumulated into ``.grad``).
        grad_outputs (sequence of Tensor): The "vector" in the vector-Jacobian product.
            Usually gradients w.r.t. each output. None values can be specified for scalar
            Tensors or ones that don't require grad. If a None value would be acceptable
            for all grad_tensors, then this argument is optional. Default: None.
        retain_graph (bool, optional): If ``False``, the graph used to compute the grad
            will be freed. Note that in nearly all cases setting this option to ``True``
            is not needed and often can be worked around in a much more efficient
            way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If ``True``, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Default: ``False``.
        allow_unused (bool, optional): If ``False``, specifying inputs that were not
            used when computing outputs (and therefore their grad is always zero)
            is an error. Defaults to ``False``.
        is_grads_batched (bool, optional): If ``True``, the first dimension of each
            tensor in ``grad_outputs`` will be interpreted as the batch dimension.
            Instead of computing a single vector-Jacobian product, we compute a
            batch of vector-Jacobian products for each "vector" in the batch.
            We use the vmap prototype feature as the backend to vectorize calls
            to the autograd engine so that this computation can be performed in a
            single call. This should lead to performance improvements when compared
            to manually looping and performing backward multiple times. Note that
            due to this feature being experimental, there may be performance
            cliffs. Please use ``torch._C._debug_only_display_vmap_fallback_warnings(True)``
            to show any performance warnings and file an issue on github if warnings exist
            for your use case. Defaults to ``False``.
    """
    t_outputs = cast(Tuple[torch.Tensor, ...], (outputs,) if is_tensor_like(outputs) else tuple(outputs))
    t_inputs = cast(Tuple[torch.Tensor, ...], (inputs,) if is_tensor_like(inputs) else tuple(inputs))
    overridable_args = t_outputs + t_inputs
    if has_torch_function(overridable_args):
        return handle_torch_function(
            grad,
            overridable_args,
            outputs,
            inputs,
            grad_outputs=grad_outputs,
            retain_graph=retain_graph,
            create_graph=create_graph,
            only_inputs=only_inputs,
            allow_unused=allow_unused,
            is_grads_batched=is_grads_batched,
        )

    if not only_inputs:
        warnings.warn("only_inputs argument is deprecated and is ignored now "
                      "(defaults to True). To accumulate gradient for other "
                      "parts of the graph, please use torch.autograd.backward.")

    grad_outputs_ = _tensor_or_tensors_to_tuple(grad_outputs, len(t_outputs))
    grad_outputs_ = _make_grads(t_outputs, grad_outputs_, is_grads_batched=is_grads_batched)

    if retain_graph is None:
        retain_graph = create_graph

    # The reason we repeat same the comment several times below is because
    # some Python versions print out the first line of multi-line function
    # calls in the traceback and some print out the last line
    if is_grads_batched:
        def vjp(gO):
            return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
                t_outputs, gO, retain_graph, create_graph, t_inputs,
                allow_unused, accumulate_grad=False)  # Calls into the C++ engine to run the backward pass
        return _vmap_internals._vmap(vjp, 0, 0, allow_none_pass_through=True)(grad_outputs)
    else:
        return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
            t_outputs, grad_outputs_, retain_graph, create_graph, t_inputs,
            allow_unused, accumulate_grad=False)  # Calls into the C++ engine to run the backward pass

  def _make_grads(outputs: Sequence[torch.Tensor], grads: Sequence[_OptionalTensor],
                  is_grads_batched: bool) -> Tuple[_OptionalTensor, ...]:
                  '''
                  Take care of grad shape
                  '''
                  assert 0
```


```cpp
// from https://github.com/pytorch/pytorch/blob/1bea49c716b8e6e748e902fe06daf66210fbc836/torch/csrc/autograd/python_engine.cpp

// Implementation of torch._C._EngineBase.run_backward
PyObject *THPEngine_run_backward(PyObject *self, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  PyObject *tensors = nullptr;
  PyObject *grad_tensors = nullptr;
  unsigned char keep_graph = 0;
  unsigned char create_graph = 0;
  PyObject *inputs = nullptr;
  unsigned char allow_unreachable = 0;
  unsigned char accumulate_grad = 0; // Indicate whether to accumulate grad into leaf Tensors or capture
  const char *accepted_kwargs[] = { // NOLINT
      "tensors", "grad_tensors", "keep_graph", "create_graph", "inputs",
      "allow_unreachable", "accumulate_grad", nullptr
  };

  // NOTE:parsing arguments

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OObb|Obb", (char**)accepted_kwargs,
              &tensors, &grad_tensors, &keep_graph, &create_graph, &inputs, &allow_unreachable, &accumulate_grad))
          return nullptr;
        THPUtils_assert(PyTuple_Check(tensors), "tensors argument is expected to "
            "be a tuple, but got %s", THPUtils_typename(tensors));
        THPUtils_assert(PyTuple_Check(grad_tensors), "grad_tensors argument is "
            "expected to be a tuple, but got %s", THPUtils_typename(grad_tensors));

        Py_ssize_t num_tensors = PyTuple_GET_SIZE(tensors);
        Py_ssize_t num_gradients = PyTuple_GET_SIZE(grad_tensors);
        THPUtils_assert(num_tensors == num_gradients, "got %ld tensors and %ld "
            "gradients", num_tensors, num_gradients);

        // The user either called autograd.backward(...) or autograd.grad(...) to get here
        bool backward_api_called = accumulate_grad;
        TORCH_CHECK(!backward_api_called || at::impl::VmapMode::current_vmap_level() == 0,
            "backward() called inside torch.vmap. This is not supported, "
            "please call backward() outside torch.vmap or instead use "
            "torch.autograd.grad inside torch.vmap");

  // ALLOCATION

  edge_list roots;
  roots.reserve(num_tensors);
  variable_list grads;
  grads.reserve(num_tensors);
  for(const auto i : c10::irange(num_tensors)) {


    PyObject *_tensor = PyTuple_GET_ITEM(tensors, i);

    THPUtils_assert(THPVariable_Check(_tensor), "element %d of tensors "
        "tuple is not a Tensor", i);
    const auto& variable = THPVariable_Unpack(_tensor);
    TORCH_CHECK(!isBatchedTensor(variable),
        "torch.autograd.grad(outputs, inputs, grad_outputs) called inside ",
        "torch.vmap. We do not support the case where any outputs are ",
        "vmapped tensors (output ", i, " is being vmapped over). Please "
        "call autograd.grad() outside torch.vmap or file a bug report "
        "with your use case.")

    auto gradient_edge = torch::autograd::impl::gradient_edge(variable);
    THPUtils_assert(gradient_edge.function,
        "element %d of tensors does not require grad and does not have a grad_fn", i);


    roots.push_back(std::move(gradient_edge));

    PyObject *grad = PyTuple_GET_ITEM(grad_tensors, i);
    if (THPVariable_Check(grad)) {
      const Variable& grad_var = THPVariable_Unpack(grad);
      if (grad_var.has_names()) {
        TORCH_WARN(
            "Autograd was passed a named grad tensor with dims ", grad_var.names(),
            ". Autograd does not yet support named tensor semantics, so all names ",
            "will be ignored. In practice all computed gradients will still be correct "
            "according to regular tensor semantics.");
      }
      grads.push_back(grad_var);
    } else {
      THPUtils_assert(grad == Py_None,
          "element %d of gradients tuple is not a Tensor or None", i);
      THPUtils_assert(!variable.requires_grad(),
          "element %d of gradients tuple is None, but the corresponding Tensor requires grad");
    }
  }

  std::vector<Edge> output_edges;
  if (inputs != nullptr) {
    int num_inputs = PyTuple_GET_SIZE(inputs);
    output_edges.reserve(num_inputs);
    for (const auto i : c10::irange(num_inputs)) {
      PyObject *input = PyTuple_GET_ITEM(inputs, i);
      THPUtils_assert(THPVariable_Check(input),
          "all inputs have to be Tensors, but got %s", THPUtils_typename(input));
      const auto& tensor = THPVariable_Unpack(input);
      TORCH_CHECK(!isBatchedTensor(tensor),
          "torch.autograd.grad(outputs, inputs, grad_outputs) called inside ",
          "torch.vmap. We do not support the case where any inputs are ",
          "vmapped tensors (input ", i, " is being vmapped over). Please "
          "call autograd.grad() outside torch.vmap or file a bug report "
          "with your use case.")
      const auto output_nr = tensor.output_nr();
      auto grad_fn = tensor.grad_fn();
      if (!grad_fn) {
        grad_fn = torch::autograd::impl::try_get_grad_accumulator(tensor);
      }
      if (accumulate_grad) {
        tensor.retain_grad();
      }
      THPUtils_assert(tensor.requires_grad(),
          "One of the differentiated Tensors does not require grad");
      if (!grad_fn) {
        // NOTE [ Autograd Unreachable Input ]
        // Since input has no grad_accumulator, its guaranteed to be unreachable.
        // We initialize an edge pointing to a non-nullptr Node so nodes in the graph
        // (e.g., mul when an operand is scalar) that have edges pointing to nullptr
        // don't get erroneously assigned `needed = True` in exec_info.
        output_edges.emplace_back(std::make_shared<Identity>(), 0);
      } else {
        output_edges.emplace_back(grad_fn, output_nr);
      }
    }
  }

  variable_list outputs;
  {
    pybind11::gil_scoped_release no_gil;
    auto& engine = python::PythonEngine::get_python_engine();
    outputs = engine.execute(roots, grads, keep_graph, create_graph, accumulate_grad, output_edges);
  }

  if (!backward_api_called && inputs != nullptr) {
    int num_inputs = PyTuple_GET_SIZE(inputs);
    THPObjectPtr py_outputs {PyTuple_New(num_inputs)};
    if (!py_outputs) return nullptr;
    for(const auto i : c10::irange(num_inputs)) {
      THPUtils_assert(allow_unreachable || outputs[i].defined(), "One of the "
                      "differentiated Tensors appears to not have been used "
                      "in the graph. Set allow_unused=True if this is the "
                      "desired behavior.");
      PyTuple_SET_ITEM(py_outputs.get(), i, THPVariable_Wrap(outputs[i]));
    }
    return py_outputs.release();
  } else {
    Py_RETURN_NONE;
  }
  END_HANDLE_TH_ERRORS
}
```
