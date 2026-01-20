# unicv

[![PyPI Latest Release](https://img.shields.io/pypi/v/unicv.svg?logo=python&logoColor=white&color=blue)](https://pypi.org/project/unicv/)
<!-- [![GitHub Release Date](https://img.shields.io/github/release-date/aether-raid/unicv?logo=github&label=latest%20release&color=blue)](https://github.com/aether-raid/unicv/releases/latest)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/aether-raid/unicv/python-publish.yml?label=PyPI%20Publish&color=blue) -->

The **OmniCV** library provides a unified and extensible framework for computer vision models that operate across heterogeneous input and output representations in a standard, modular manner.

The architecture and design philosophy of OmniCV is inspired by modular deep learning ecosystems such as [`pytorch`](https://github.com/pytorch/pytorch) and HuggingFace’s [`transformers`](https://github.com/huggingface/transformers), as well as recent efforts toward foundation models and generalist perception systems in computer vision. Rather than prescribing fixed pipelines (e.g. RGB → Depth or RGB → Mesh), OmniCV abstracts vision algorithms as composable transformations between representation spaces.

At the core of OmniCV lies an abstract function, denoted as **`f`**, which defines a standardized interface for mapping *any combination of visual input modalities* to *any combination of output modalities*. These modalities include, but are not limited to:

- RGB images
- Depth maps
- Point clouds
- Meshes
- Gaussian splats and other implicit or semi-implicit scene representations

Concrete vision algorithms—such as monocular depth estimation, RGB-to-point-cloud reconstruction, or RGB-D refinement—are implemented as subclasses of this abstract interface. Existing models available online (e.g. DepthPro, MiDaS, CDM, or point-cloud reconstruction networks) can be redefined within this framework without altering their internal logic, allowing them to be seamlessly integrated into a shared system.

This abstraction enables OmniCV to decouple **input modality**, **latent processing**, and **output representation**, encouraging reuse, composition, and extension of vision algorithms. Models may share encoders, latent spaces, or decoders, and can be combined or chained to support progressive or multi-stage reconstruction pipelines.

The conceptual motivation for OmniCV is closely aligned with the emergence of **foundation models for perception**, where a single system is expected to reason across tasks, representations, and data sources. By enforcing a common interface at the representation level, OmniCV facilitates cross-representation supervision, multi-task learning, and interoperability between otherwise incompatible vision methods.

With this design, OmniCV aims to support vision systems that are:

1. **Modality-agnostic**, capable of ingesting arbitrary combinations of visual inputs (e.g. RGB, depth, point clouds) without architectural redesign.
2. **Representation-agnostic**, able to emit multiple scene representations from a shared latent abstraction.
3. **Algorithm-agnostic**, allowing existing and future computer vision models to be wrapped, extended, or replaced under a common interface.
4. **Composable**, enabling complex pipelines to be constructed by chaining or jointly training multiple `f`-based modules.
5. **Extensible**, supporting both classical CV algorithms and modern deep learning approaches, including implicit scene representations and neural rendering techniques.
6. **Foundation-ready**, serving as an architectural substrate for training large, generalist vision models capable of cross-task and cross-representation transfer.

In addition to standard convolutional and Transformer-based architectures, OmniCV is designed to accommodate emerging paradigms such as implicit neural representations, Gaussian splatting, and hybrid geometric–neural pipelines, enabling a unified experimental platform for next-generation 3D perception systems.

## 🚧 Roadmap

We are actively building out the core modules of **unicv**.  
Here’s the current progress:

- [ ] Set Up the whole library


## 🛠️ Installation and Set-Up

### Installing from PyPI

Yes, we have published our framework on PyPI! To install the unicv library and all its dependencies, the easiest method would be to use `pip` to query PyPI. This should, by default, be present in your Python installation. To, install run the following command in a terminal or Command Prompt / Powershell:

```bash
$ pip install unicv
```

Depending on the OS, you might need to use `pip3` instead. If the command is not found, you can choose to use the following command too:

```bash
$ python -m pip install unicv
```

Here too, `python` or `pip` might be replaced with `py` or `python3` and `pip3` depending on the OS and installation configuration. If you have any issues with this, it is always helpful to consult 
[Stack Overflow](https://stackoverflow.com/).

### Installing from Source

Git is needed to install this repository from source. This is not completely necessary as you can also install the zip file for this repository and store it on a local drive manually. To install Git, follow [this guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

After you have successfully installed Git, you can run the following command in a terminal / Command Prompt:

```bash
$ git clone https://github.com/aether-raid/unicv.git
```

This stores a copy in the folder `unicv`. You can then navigate into it using `cd unicv`. Then, you can run the following:

```bash
$ pip install .
```

This should install `unicv` to your local Python instance.

<!-- ## 💻 Getting Started -->

