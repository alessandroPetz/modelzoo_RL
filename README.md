# Model zoo (April 2026)

The model zoo is a GitLab repository (https://gitlab.seanergys.fz-juelich.de/software/modelzoo) hosting code that implements models and Day-0 datasets, and has a CI pipeline for automatic testing. It was structured with several **requirements** in mind:
* It should be public 
* Enable reproducibility and versioning of models 
* Enable reuse of datasets
* Facilitate use of the same dataset across different models
* Facilitate use of more datasets by one model
* Data is not stored within the model zoo
* Pretrained models are stored within the model zoo
* Easy integration of new models and datasets - integrated CI pipeline that discovers new models and datasets and tests them automatically
* Facilitate import of models and train/test configuration by Seanergys AI production tool


## Quick start

**Requirements:** Linux or macOS with `curl`, `git` and `make`. Everything else — `uv`,
a supported Python (3.12–3.14), `poetry`, the virtual environment and all dependencies —
is installed for you.

```bash
git clone https://gitlab.seanergys.fz-juelich.de/software/modelzoo.git
cd modelzoo
make test
```

That is the whole setup. Every `make` target that needs a working environment depends on
`make env`, which:

1. runs `scripts/ensure_tools.sh` — checks `curl`/`git`/`make`, then installs **uv** and
   **poetry** into `~/.local/bin` if they are missing, and fetches a supported Python with
   `uv python install` when the system one is out of range (no `sudo`, nothing global);
2. verifies `poetry.lock` is in sync with `pyproject.toml`;
3. creates `.venv/` if absent and runs `poetry install --with dev,ci`, re-running it only
   when `pyproject.toml` or `poetry.lock` actually changed.

So `make test`, `make lint`, `make docs-ci` and friends all work on a fresh clone with no
manual setup step.

| Command | What it does |
|---|---|
| `make bootstrap` | Do the whole setup explicitly, without running tests |
| `make doctor` | Report toolchain / `.venv` / fixture state, change nothing |
| `make tools` | Ensure `uv`, Python 3.12–3.14 and `poetry` exist (installs the missing ones) |
| `make tools-check` | Same checks, **never** installs — non-zero exit if anything is missing |
| `make env` | Ensure `.venv/` exists and matches `poetry.lock` (dev + ci groups) |
| `make help` | Full list of targets |

**Opting out of automatic installation.** On a locked-down machine, or in a container image
that must not fetch tools, set `MODELZOO_NO_AUTO_INSTALL=1` — the toolchain step then only
reports what is missing and fails instead of installing:

```bash
MODELZOO_NO_AUTO_INSTALL=1 make test
```

If the new tools are not on your `PATH` in a fresh shell, add this to `~/.bashrc`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Running the tests

```bash
make test               # unit + smoke — fast, offline, run before every push
make test-all           # unit + smoke + integration
make pre-push           # format-check + secrets + lint + fast tests
make ci                 # full local mirror of the GitLab pipeline
```

The integration fixtures are committed, so the offline tests need no download — and if one
is ever missing, the test session fetches it itself before running (any discovered dataset,
no configuration). `make sample-data` refreshes them explicitly; `MODELZOO_FIXTURES=never`
(implied by `CI_SKIP_ZENODO=1`) forbids any network access.

The full local-development reference lives in [`docs/ci-cd.md`](docs/ci-cd.md).

## Repository structure

The repository has 4 main folders: two contain classes for the dataloader and datasets, one the logger, and one contains all models.

![](https://gitlab.jsc.fz-juelich.de/hedgedoc/uploads/e698553a-2e64-46a6-bf1d-c892d98bab4a.png)

Both dataset and model folders contain a common folder with base classes.
Models are organised by task. Under each task, we can have multiple models. Each model has its own folder, containing the model class and a subfolder with pretrained models. 

![](https://)![](https://gitlab.jsc.fz-juelich.de/hedgedoc/uploads/65c3bf13-de41-4d43-b2c8-2339b524c32a.png)

## Data and models

The main classes of the Model Zoo are related to datasets and models, and their interaction. They have been designed so that:
* datasets are standalone and not aware of the models that use them
* models are aware of the datasets they can work with - they contain a list of compatible datasets
* the model-dataset relation is N-N
* datasets and models are discoverable, use uniform APIs and can be tested automatically in the CI pipeline - train and test configurations are available in models for each dataset 
* classes are easily validated - we employ Pydantic for validation. This means that object initialisation is not done in the __init__ method, but in validate_model inherited from pydantic.BaseModel


![](https://gitlab.jsc.fz-juelich.de/hedgedoc/uploads/91120393-aa67-4787-8410-b1e0776af410.png)


### Data classes
We use the PyTorch interface to work with data, through Dataset and Dataloader objects. We provide base classes and implement several datasets.

Base classes:
* SeanergysDataloader:
    * Main dataloader class, extends torch.utils.data.DataLoader
    * Should not be extended but works out-of-the-box with any Seanergys dataset, using the PyTorch-defined implementation
* SeanergysDataset:
    * Base abstract data class, extends torch.utils.data.Dataset and pydantic.BaseModel
    * Should be extended by all Seanergys datasets, by implementing the \_\_len\_\_ , \_\_get_item\_\_ and validate_model  methods.
        * \_\_get_item\_\_, inherited from torch.utils.data.Dataset, returns one record in the data, identified by an index. It is used by the Dataloader to provide access to data.
        * the method validate_model, inherited from pydantic.BaseModel, can contain initialisation logic for the Dataset object.
        * Lazy loading is possible by keeping a mapping between indices and data batches to be loaded in memory - it is reccomended not to use shuffle=True in the Dataloader when working with lazy loading datasets, but organise shuffling externally to the loader.

* SeanergysParquetDataset:
    * Utility class for datasets using parquet files
    * Implements the SenergysDataset class
    * Can be extended or used as is

Dataset classes:
* PM100Dataset 
    * accesses Marconi 100 job data from the Zenodo repository 
    * extends SeanergysParquetDataset
* FDataDataset
    * accesses Fugaku Data data from Zenodo 
    * extends SeanergysParquetDataset
* SCRIPTAIDataset
    * accesses job script data from Zenodo
    * Extends SeanergysDataset
* DurationPredictionBull
    * HPC job duration prediction model based on an ensemble of neural networks.
    * Predicts job duration from job metadata and learned embeddings.
    * Supports the FDataDataset.
    
Further classes are being implemented: M100 and IT4I.

### Model classes

The model classes follow standard ML pipelines and provides functionality for reproducibility and transparency, model saving and loading, training, testing and evaluating, retrieving default training and testing configuration, hosting pretrained models. 

Base model classes:

* SeanergysModel
    * Abstract class extending pydantic.BaseModel
    * All models must extend this class 
    * Selected attributes:
        * metadata - object of type SeanergysModelMetadata - function as model card including all information about the model
        * task_type - of enum type SeanergysModelTask, can be regression or classification
        * supported_datasets - list of dataset names that allow ModelZoo users to automatically detect compatible datasets (used for instance in the CI pipeline for integration tests)
        * logger - SeanergysLogger object 
    * Selected methods:
        * validate_model - pydantic model validation. This method can also contain initialisation logic that would be included in the class constructor in an implementation not using Pydantic. 
        * load - load model from file
        * save - save model to file
        * train - takes as input a SeanergysDataLoader for training, optionally also one for validation, and callables for training loss and validation metrics. It trains the model on the training data.
        * predict - takes as input a SeanergysDataLoader and applies the model to all data included in it. 
        * evaluate - like predict, but return values for eavluation metrics rather than the predictions themselves.
        * get_train_config 
            * static method
            * returns a SeanergysModel instance with specific hyperparameters and a SenargysDataLoader instance with training data and preprocessing specifications. 
            * it requires in input the name of the dataset to be used, and whether we need a real training specification or small dummy data is enough. 
            * this method is used by the CI pipeline to constuct automatic tests for all compatible model-dataset combinations.
        * get_test_config
            * static method
            * takes as input a model instance (to be able to access preprocessing methods), a dataset name and a flag for dummy data
            * returns a Dataloader with the correct preprocessing settings
* SeanergysSklearnModel
    * Helper class for models usingsklearn models in the background
    * Implements SeanergysModel
    * Can be used as is or extended by models

Model classes: 

* MACK
    * model that classifies performance of HPC jobs, beyond the binary memory/compute-bound classification.
    * To be presented in CCGrid 2026. https://ccgrid2026.cdms.westernsydney.edu.au/
* MCBound 
    * model that classifies HPC jobs as memory/compute bound.
    * presented in Antici, Francesco, et al. "Mcbound: An online framework to characterize and classify memory/compute-bound hpc jobs." SC24: International Conference for High Performance Computing, Networking, Storage and Analysis. IEEE, 2024. https://ieeexplore.ieee.org/abstract/document/10793161 
* JPCP 
    * job power consumption prediction model
    * presented in Antici, Francesco, et al. "An online algorithm for power consumption prediction of HPC workload." Future Generation Computer Systems (2025): 108064. https://www.sciencedirect.com/science/article/pii/S0167739X25003590

## Logging

Logging is performed by a SeanergysLogger object:
* extends logging.Logger
* can log to stdout or file
* is passed to model and dataset classes for logging. If no logger is provided in constructor, a default stdout logger is created. 

## CI pipeline

The CI pipeline automatically tests all code committed to the Model Zoo repository. It is triggered by every commit and merge request, and runs through several stages in sequence.

First, basic sanity checks ensure the repository is well-formed and free of obvious errors. Then, code quality checks verify that the code follows consistent style conventions. Finally, three levels of tests are executed:

- Unit tests verify individual components and base class methods in isolation.
- Smoke tests check that all models and datasets can be discovered and correctly instantiated. This is where Pydantic validation runs, and where automatic discovery of new classes happens — any newly added model or dataset is picked up without changes to the pipeline.
- Integration tests verify that full training and testing pipelines work end-to-end using dummy data, for every compatible model–dataset combination.

Tests are parametrised to cover all registered models and all *compatible model–dataset pairs* automatically. Specifically, models are discovered automatically by parsing the model directories. Subsequently, for each model, the list of compatible datasets and train/test configurations is retrieved and their integration tested using dummy versions of the datasets. 

After the tests, coverage from every layer is combined into a single report, the build stage produces the installable wheel, and a final stage verifies the distributed wheel imports cleanly.

This means that contributing a new model or dataset is enough to have it tested — the pipeline discovers and tests it without any manual configuration.

A complete, always-current reference of **every pipeline job, its trigger rule, its pass/fail behaviour, and the full model×dataset capability matrix** is maintained in [`docs/ci-cd.md`](docs/ci-cd.md). That document is auto-generated from the pipeline definition and the code, so it never drifts; to propose a change to any CI rule or gate, open an issue using the *CI Change Request* template.

New to the pipeline? Start with the basic-language [`CI/CD tutorials`](docs/tutorials/README.md), which explain the stages, tests, important files, automatic discovery, and documentation generator step by step.

The CI pipeline is still under development, and additional test coverage is planned.
