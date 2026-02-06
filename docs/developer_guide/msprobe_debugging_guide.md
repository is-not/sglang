# MSProbe Debugging Guide
During inference or training runs we often encounter accuracy anomalies such as outputs drifting away from the expectation, unstable numerical behavior (NaN/Inf), or predictions that no longer match the labels. To pinpoint the root cause we have to monitor and capture intermediate data produced while the model executes—feature maps, weights, activations, and layer outputs. By capturing key tensors at specific stages, logging I/O pairs for the core layers, and retaining contextual metadata (prompts, tensor dtypes, hardware configuration, etc.), we can systematically trace where the accuracy degradation or numerical error started. This guide describes the end-to-end workflow for diagnosing accuracy issues for AI models (with a focus on SGLang services): preparation, data capture, and analysis & verification.

## Background Concepts
`msProbe` supports three accuracy levels:

- L0: Dump tensors or tensor statistics at the module level and generates construct.json so that visualization tools can rebuild the network structure. A model or submodule handle must be passed in.

- L1: Dump tensors or tensor statistics at the torch API level.

- mix: L0 + L1, which is useful when you need both graph reconstruction and numerical comparisons.

## Prerequisites
### Install msProbe
Install msProbe with pip:
```shell
pip install mindstudio-probe --pre
```
## Collecting Data with msProbe
When msProbe is enabled, cuda graph and server warmup are disabled.

### Configuration Introduction
When using msProbe dump or other operations, you can provide a JSON file to customize parameters for personalized data dump; When no JSON file is specified, the default configuration will be used.

| Field | Description                                                                                                                                                                                                              | Required |
|:---:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---:|
| `task` | Type of dump task. Common PyTorch values include `"statistics"` and `"tensor"`. A statistics task collects tensor statistics (mean, variance, max, min, etc.) while a tensor task captures arbitrary tensors.            | Yes |
| `dump_path` | Directory where dump results are stored. When omitted, `msProbe` uses its default path.                                                                                                                                  | No |
| `rank` | Ranks to sample. An empty list collects every rank. For single-card tasks you must set this field to `[]`.                                                                                                               | No |
| `step` | Token iteration(s) to sample. An empty list means every iteration.                                                                                                                                                       | No |
| `level` | Dump level string (`"L0"`, `"L1"`, or `"mix"`). `L0` targets `nn.Module`, `L1` targets `torch.api`, and `mix` collects both.                                                                                             | Yes |
| `async_dump` | Whether to enable asynchronous dump (supported for PyTorch `statistics`/`tensor` tasks). Defaults to `false`.                                                                                                            | No |
| `scope` | Customize the scope of dump. An empty list dumps every module or torch API.                                                                                                                                              | No |
| `list` | Customize dump list, only dumps elements from the list. An empty list dumps every module or torch API.                                                                                                                   | No |

- `scope` (list[str]): In PyTorch pynative scenarios this field restricts the dump range. Provide two module or API names that follow the tool's naming convention to lock a range; only data between the two names will be dumped. Examples:

  ```text
  "scope": ["Module.conv1.Conv2d.forward.0", "Module.fc2.Linear.forward.0"]
  "scope": ["Tensor.add.0.forward", "Functional.square.2.forward"]
  ```

  The `level` setting determines what can be provided—modules when `level=L0`, APIs when `level=L1`, and either modules or APIs when `level=mix`.

- `list` (list[str]): Custom operator list. Options include:
    - Supply the full names of specific APIs in PyTorch pynative scenarios to only dump those APIs. Example: `"list": ["Tensor.permute.1.forward", "Tensor.transpose.2.forward", "Torch.relu.3.backward"]`.
    - When `level=mix`, you can provide module names so that the dump expands to everything produced while the module is running. Example: `"list": ["Module.module.language_model.encoder.layers.0.mlp.ParallelMlp.forward.0"]`.
    - Provide a substring such as `"list": ["relu"]` to dump every API whose name contains the substring. When `level=mix`, modules whose names contain the substring are also expanded.

Example configuration:

```bash
cat <<'JSON' > /data/msprobe_config.json
{
  "task": "statistics",
  "dump_path": "/home/data_dump",
  "rank": [],
  "step": [],
  "level": "L1",
  "async_dump": false,

  "statistics": {
    "scope": [],
    "list": [],
    "tensor_list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
JSON
```

### Enable `msProbe` in SGLang

1. Use parameter `--enable-msprobe` to enable msProbe in SGLang.

    ```bash
    python3 -m sglang.launch_server \
     --model-path Qwen/Qwen2.5-0.5B-Instruct \
     --host 127.0.0.1 \
     --port 1027 \
     --enable-msprobe
    ```
    The default dump parameter is: `"task": "statistics"`, `"dump_path": "./dump_path"`, `"rank": []`, `"step": []`, `"level": "L1"`, `"async_dump": false`.

    This means that msProbe will dump all torch APIs statistics data of all ranks and steps during the model running process, and save it in the "dump_path" folder of the current path.

2. Optionally, you can use parameter `--msprobe-config-path` to pass in a JSON file for custom data dump.

    For example, if you want to dump **tensor** data for **all torch APIs and modules** during the model running process, only collecting **rank0 and rank1**, only collecting **step0, step2, and step10**, and saving the data to the path **"home/my_dump_data"**, you need to configure the following settings:

   - Prepare a JSON file, for example named "msprobe-config.json", and store it in the "/home" path:
        ```json
        {
            "task": "tensor",
            "dump_path": "home/my_dump_data",
            "rank": [0, 1],
            "step": [0, 1, 10],
            "level": "mix",
            "async_dump": false,

            "tensor": {
                "scope": [],
                "list":[],
                "data_mode": ["all"]
            }
        }
        ```
   - Use parameter `--msprobe-config-path`:
       ```bash
       python3 -m sglang.launch_server \
        --model-path Qwen/Qwen2.5-0.5B-Instruct \
        --host 127.0.0.1 \
        --port 1027 \
        --enable-msprobe
        --msprobe-config-path /home/msprobe-config.json
       ```

### Send requests and collect dumps

1. Send inference requests as usual, for example:

   ```bash
   curl -H "Content-type: application/json" \
    -X POST \
    -d '{
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": "Hello, my name is"
            }
        ],
        "max_tokens": 10
    }' \
    http://127.0.0.1:1027/v1/chat/completions
   ```

2. Each request drives the sequence `msprobe: start -> forward -> stop -> step`. The runner invokes `step()` on every code path, so you always get a complete dataset even if inference returns early.

3. Dump files are written into `dump_path`. They usually contain:
   - `dump.json`, which records metadata such as dtype, shape, min, max, mean, L2 norm, and `requires_grad`.
   - `construct.json`, hierarchical structure description, when `level` is `L0` or `mix` (required for visualization), its content is not empty.
   - `stack.json`, Record the call stack information of API/Module.
   - `dump_tensor_data`, generated when `task` is `tensor` and save the collected tensor data.

   See [dump directory description](#dump-directory-description) for details.


## Analyze the results

### Prerequisites

You typically need two dump datasets: one from the "problem side" (the run that exposes the accuracy or numerical error) and another from the "benchmark side" (a good baseline). These datasets do not have to be identical—they can come from different branches, framework versions, or even alternative implementations (operator substitutions, different graph-optimization switches, etc.). As long as they use the same or similar inputs, hardware topology, and sampling points (step/token), `msProbe` can compare them and locate the divergent nodes. If you cannot find a perfectly clean benchmark, start by capturing the problem-side data, craft the smallest reproducible case by hand, and perform a self-comparison. Below we assume the problem dump is `problem_dump` and the benchmark dump is `bench_dump`.

### Visualization

Use `msprobe graph_visualize` to generate results that can be opened inside `tensorboard`.

1. Ensure that `constructor.json` in dump is not empty (i.e., `level = L0` or `level = mix`).
2. Generate the necessary files for visualization.

    Taking the directory structure in [dump directory description](#dump-directory-description) as an example:

   - Example 1: Single rank graph comparison
     ```shell
     msprobe graph_visualize -tp ./problem_dump/srep0/rank0 -gp ./bench_dump/srep0/rank0 -o ./graph_output
     ```
   - Example 2: Multi rank batch graph comparison
     ```shell
     msprobe graph_visualize -tp ./problem_dump/srep0 -gp ./bench_dump/srep0 -o ./graph_output
     ```
   - Example 3: Multi step batch graph comparison
     ```shell
     msprobe graph_visualize -tp ./problem_dump -gp ./bench_dump -o ./graph_output
     ```
   - Example 4: Overflow check
     ```shell
     msprobe graph_visualize -tp ./problem_dump/srep0/rank0 -gp ./bench_dump/srep0/rank0 -o ./graph_output -oc
     ```
   - Example 5: In the above example, the `-gp` parameter can be omitted for graph build task
     ```shell
     msprobe graph_visualize -tp ./problem_dump/srep0/rank0 -o ./graph_output
     ```

   After the comparison or build task finishes, a `*.vis.db` file is created under `graph_output`.

   - Graph build: `build_{timestamp}.vis.db`
   - Graph comparison: `compare_{timestamp}.vis.db`

3. Launch `tensorboard` and load the output directory to inspect structural differences, numerical comparisons, overflow detection results, cross-device communication nodes, and filters/search. Pass the directory containing the `.vis.db` files to `--logdir`:

   ```bash
   tensorboard --logdir out_path --bind_all --port [optional_port]
   ```

4. Inspect the visualization. The UI usually displays the overall model structure with operators, parameters, and tensor I/O. Click any node to expand its children.
   - **Difference visualization**: Comparison results highlight divergent nodes with different colors (the larger the difference, the redder the node). Click a node to view its detailed information including tensor inputs/outputs, parameters, and operator type. Analyze the data difference and the surrounding connections to pinpoint the exact divergence.
   - **Helper features**:
     - Switch rank/step: Quickly check difference nodes on different ranks and steps.
     - Search/filter: Use the search box to filter nodes by operator name, etc.
     - Manual mapping: Automatic mapping cannot cover every case, so the tool lets you manually map nodes between the problem and benchmark graphs before generating comparison results.

## 5. Troubleshooting

- No dump files: Confirm that the JSON path is correct and every node has write permission.
- Dumps are too large: Start with a `statistics` task to locate abnormal tensors, then narrow the scope with `scope`/`list`/`tensor_list`, `filters`, `token_range`, etc.

---

## Appendix

### Dump directory description

```text
├── problem_dump or bench_dump
│   ├── step0
│   │   ├── rank0
│   │   │   ├── dump_tensor_data
│   │   │   │    ├── Tensor.permute.1.forward.pt
│   │   │   │    ├── Functional.linear.5.backward.output.pt    # Format: {api_type}.{api_name}.{call_count}.{forward/backward}.{input/output}.{arg_index}.
│   │   │   │    │                                              # arg_index is the nth input or output of the API. If an input is a list, keep numbering with decimals (e.g., 1.1 is the first element of the first argument).
│   │   │   │    ├── Module.conv1.Conv2d.forward.0.input.0.pt          # Format: {Module}.{module_name}.{class_name}.{forward/backward}.{call_count}.{input/output}.{arg_index}.
│   │   │   │    ├── Module.conv1.Conv2d.forward.0.parameters.bias.pt  # Module parameter data: {Module}.{module_name}.{class_name}.forward.{call_count}.parameters.{parameter_name}.
│   │   │   │    └── Module.conv1.Conv2d.parameters_grad.weight.pt     # Module parameter gradients: {Module}.{module_name}.{class_name}.parameters_grad.{parameter_name}. Gradients do not include call_count because the same gradient updates all invocations.
│   │   │   │                                                          # When the `model` argument passed to dump is a List[torch.nn.Module] or Tuple[torch.nn.Module], module-level data names also include the index inside the list ({Module}.{index}.*), e.g., Module.0.conv1.Conv2d.forward.0.input.0.pt.
│   │   │   ├── dump.json
│   │   │   ├── stack.json
│   │   │   ├── dump_error_info.log
│   │   │   └── construct.json
│   │   ├── rank1
│   │   │   ├── dump_tensor_data
│   │   │   │   └── ...
│   │   │   ├── dump.json
│   │   │   ├── stack.json
│   │   │   ├── dump_error_info.log
│   │   │   └── construct.json
│   │   ├── ...
│   │   │
│   │   └── rank7
│   ├── step1
│   │   ├── ...
│   ├── step2
```

- `rank`: Device ID. Each card writes its data to the corresponding `rank{ID}` directory. In non-distributed scenarios the directory is simply named `rank`.
- `dump_tensor_data`: Save the collected tensor data.
- `dump.json`: Statistics for the forward data of each API or module, including names, dtype, shape, max, min, mean, L2 norm (square root of the L2 variance), and CRC-32 when `summary_mode="md5"`. See [dump.json file description](#dumpjson-file-description) for details.
- `dump_error_info.log`: Present only when the dump tool encountered an error and records the failure log.
- `stack.json`: Call stacks for APIs/modules.
- `construct.json`: Hierarchical structure description. Empty when `level=L1`.

### dump.json file description

#### L0 level

An L0 `dump.json` contains forward/backward I/O for modules together with parameters and parameter gradients. Using PyTorch's `Conv2d` as an example, the network code looks like:

`output = self.conv2(input)  # self.conv2 = torch.nn.Conv2d(64, 128, 5, padding=2, bias=True)`

`dump.json` contains the following entries:

- `Module.conv2.Conv2d.forward.0`: Forward data of the module. `input_args` represents positional inputs, `input_kwargs` represents keyword inputs, `output` stores forward outputs, and `parameters` stores weights/biases.
- `Module.conv2.Conv2d.parameters_grad`: Parameter gradients (weight and bias).
- `Module.conv2.Conv2d.backward.0`: Backward data of the module. `input` represents gradients that flow into the module (gradients of the forward outputs) and `output` represents gradients that flow out (gradients of the module inputs).

**Note**: When the `model` parameter passed to the dump API is `List[torch.nn.Module]` or `Tuple[torch.nn.Module]`, module-level names include the index inside the list (`{Module}.{index}.*`). Example: `Module.0.conv1.Conv2d.forward.0`.

```json
{
 "task": "tensor",
 "level": "L0",
 "framework": "pytorch",
 "dump_data_dir": "/dump/path",
 "data": {
  "Module.conv2.Conv2d.forward.0": {
   "input_args": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      8,
      16,
      14,
      14
     ],
     "Max": 1.638758659362793,
     "Min": 0.0,
     "Mean": 0.2544615864753723,
     "Norm": 70.50277709960938,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.input.0.pt"
    }
   ],
   "input_kwargs": {},
   "output": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      8,
      32,
      10,
      10
     ],
     "Max": 1.6815717220306396,
     "Min": -1.5120246410369873,
     "Mean": -0.025344856083393097,
     "Norm": 149.65576171875,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.output.0.pt"
    }
   ],
   "parameters": {
    "weight": {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32,
      16,
      5,
      5
     ],
     "Max": 0.05992485210299492,
     "Min": -0.05999220535159111,
     "Mean": -0.0006165213999338448,
     "Norm": 3.421217441558838,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.parameters.weight.pt"
    },
    "bias": {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32
     ],
     "Max": 0.05744686722755432,
     "Min": -0.04894155263900757,
     "Mean": 0.006410328671336174,
     "Norm": 0.17263513803482056,
     "requires_grad": true,
     "data_name": "Module.conv2.Conv2d.forward.0.parameters.bias.pt"
    }
   }
  },
  "Module.conv2.Conv2d.parameters_grad": {
   "weight": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32,
      16,
      5,
      5
     ],
     "Max": 0.018550323322415352,
     "Min": -0.008627401664853096,
     "Mean": 0.0006675920449197292,
     "Norm": 0.26084786653518677,
     "requires_grad": false,
     "data_name": "Module.conv2.Conv2d.parameters_grad.weight.pt"
    }
   ],
   "bias": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32
     ],
     "Max": 0.014914230443537235,
     "Min": -0.006656786892563105,
     "Mean": 0.002657240955159068,
     "Norm": 0.029451673850417137,
     "requires_grad": false,
     "data_name": "Module.conv2.Conv2d.parameters_grad.bias.pt"
    }
   ]
  },
  "Module.conv2.Conv2d.backward.0": {
   "input": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      8,
      32,
      10,
      10
     ],
     "Max": 0.0015069986693561077,
     "Min": -0.001139344065450132,
     "Mean": 3.3215508210560074e-06,
     "Norm": 0.020567523315548897,
     "requires_grad": false,
     "data_name": "Module.conv2.Conv2d.backward.0.input.0.pt"
    }
   ],
   "output": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      8,
      16,
      14,
      14
     ],
     "Max": 0.0007466732058674097,
     "Min": -0.00044813455315306783,
     "Mean": 6.814070275140693e-06,
     "Norm": 0.01474067009985447,
     "requires_grad": false,
     "data_name": "Module.conv2.Conv2d.backward.0.output.0.pt"
    }
   ]
  }
 }
}
```

#### L1 level

An L1 `dump.json` records forward/backward I/O for APIs. Using PyTorch's `relu` function as an example (`output = torch.nn.functional.relu(input)`), the file contains:

- `Functional.relu.0.forward`: Forward data of the API. `input_args` are positional inputs, `input_kwargs` are keyword inputs, and `output` stores the forward outputs.
- `Functional.relu.0.backward`: Backward data of the API. `input` represents the gradients of the forward outputs, and `output` represents the gradients that flow back to the forward inputs.

```json
{
 "task": "tensor",
 "level": "L1",
 "framework": "pytorch",
 "dump_data_dir":"/dump/path",
 "data": {
  "Functional.relu.0.forward": {
   "input_args": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32,
      16,
      28,
      28
     ],
     "Max": 1.3864083290100098,
     "Min": -1.3364859819412231,
     "Mean": 0.03711778670549393,
     "Norm": 236.20692443847656,
     "requires_grad": true,
     "data_name": "Functional.relu.0.forward.input.0.pt"
    }
   ],
   "input_kwargs": {},
   "output": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32,
      16,
      28,
      28
     ],
     "Max": 1.3864083290100098,
     "Min": 0.0,
     "Mean": 0.16849493980407715,
     "Norm": 175.23345947265625,
     "requires_grad": true,
     "data_name": "Functional.relu.0.forward.output.0.pt"
    }
   ]
  },
  "Functional.relu.0.backward": {
   "input": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32,
      16,
      28,
      28
     ],
     "Max": 0.0001815402356442064,
     "Min": -0.00013352684618439525,
     "Mean": 0.00011915402356442064,
     "Norm": 0.007598237134516239,
     "requires_grad": false,
     "data_name": "Functional.relu.0.backward.input.0.pt"
    }
   ],
   "output": [
    {
     "type": "torch.Tensor",
     "dtype": "torch.float32",
     "shape": [
      32,
      16,
      28,
      28
     ],
     "Max": 0.0001815402356442064,
     "Min": -0.00012117840378778055,
     "Mean": 2.0098118724831693e-08,
     "Norm": 0.006532244384288788,
     "requires_grad": false,
     "data_name": "Functional.relu.0.backward.output.0.pt"
    }
   ]
  }
 }
}
```

#### mix level

A `mix` dump.json contains both L0 and L1 level data; the file format is the same as the examples above.
