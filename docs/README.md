# Documentation

This folder contains the scripts necessary to build the documentation for Merlin-KV.
You can view the generated [Merlin-KV documentation](https://nvidia-merlin.github.io/merlin-kv/master/README.html).

## Contributing to Docs

Follow the instructions below to be able to build the docs.

1. Install required documentation tools and extensions:

```shell
pip install -r docs/requirements-doc.txt
```

1. Build the documentation:

`make -C docs clean html`

The preceding command runs Sphinx in your shell and outputs to build/html/index.html.

The build process for Merlin-KV is unique among the Merlin projects because it
uses Doxygen, Breathe, and Exhale to create API documentation from the C++ source.

## Preview the changes

View docs web page by opening the HTML in your browser.
Run the following command from the root of the repository:

```bash
python -m http.server 8000 --directory docs/build/html
```

Afterward, open a web browser and access `https://localhost:8000`.

Check that your edits formatted correctly and read well.

## Decisions

### Source management: README and index files

* To preserve Sphinx's expectation that all source files are child files and directories
  of the `docs/source` directory, other content, such as the `README.md` file is
  copied to the source directory. You can determine which directories and files are copied by
  viewing `docs/source/conf.py` and looking for the `copydirs_additional_dirs` list.
  Directories are specified relative to the Sphinx source directory, `docs/source`.

* One consequence of the preceding bullet is that any change to the original files,
  such as adding or removing a topic, requires a similar change to the `docs/source/toc.yaml`
  file.  Updating the `docs/source/toc.yaml` file is not automatic.

* Because the GitHub browsing expectation is that a `README.md` file is rendered when you
  browse a directory, when a directory is copied, the `README.md` file is renamed to
  `index.md` to meet the HTML web server expectation of locating an `index.html` file
  in a directory.

### Adding links

TIP: When adding a link to a method or any heading that has underscores in it, repeat
the underscores in the link even though they are converted to hyphens in the HTML.

Refer to the following examples:

* `../QAList.md#24-how-to-set-workspace_size_per_gpu_in_mb-and-slot_size_array`
* `./api/python_interface.md#save_params_to_files-method`

#### Docs-to-docs links

There is no concern for the GitHub browsing experience for files in the `docs/source/` directory.
You can use a relative path for the link.  For example--and using the HugeCTR repository
for an example--the following link is in the `docs/source/hugectr_user_guide.md` file
and links to the "Build HugeCTR from Source" heading in the
`docs/source/hugectr_contributor_guide.md` file:

```markdown
To build HugeCTR from scratch, refer to
[Build HugeCTR from source code](./hugectr_contributor_guide.md#build-hugectr-from-source).
```

#### Docs-to-repository links

Some files that we publish as docs, such as HugeCTR's `release_notes.md` file, refer readers to files
that are not published as docs. For example, we currently do not publish information from the following
directories:

* `gpu_cache`
* `onnx_converter`
* `samples`
* `tools`
* `tutorial`

To refer a reader to a README or program in one of the preceding directories, state that
the link is to the repository:

```markdown
+ **Python Script and documentation demonstrating how to analyze model files**: In this release,
we provide a script to retrieve vocabulary information from model file. Please find more details
in the README in the
[tools/model_analyzer](https://github.com/NVIDIA-Merlin/HugeCTR/tree/v3.5/tools/model_analyzer)
directory of the repository.
```

The idea is to let a reader know that following the link&mdash;whether from an HTML docs page or
from browsing GitHub&mdash;results in viewing our repository on GitHub.

> TIP: In the `release_notes.md` file, use the tag such as `v3.5` instead of `master` so that
> the link is durable.

#### Links to notebooks

The notebooks are published as documentation. The few exceptions are identified in the
`docs/source/conf.py` file in the `exclude_patterns` list:

```python
exclude_patterns = [
    "notebooks/prototype_indices.ipynb",
]
```

If the document that you link from is also published as docs, such as `release_notes.md`, then
a relative path works both in the HTML docs page and in the repository browsing experience:

```markdown
+ **Support HDFS Parameter Server in Training**:
    + ...snip...
    + ...snip...
    + Added a [notebook](notebooks/training_with_hdfs.ipynb) to show how to use HugeCTR with HDFS.
```

If the document that you link from is not published as docs, such as a file in the `tools`
or `samples` directory, then either a relative path or a link to the HTML notebook is OK.

A link to the HTML notebook is like the following:

```markdown
<https://nvidia-merlin.github.io/HugeCTR/master/notebooks/continuous_training.html>
```

#### Links from notebooks to docs

Use a link to the HTML page like the following:

```markdown
<https://nvidia-merlin.github.io/HugeCTR/master/hugectr_user_guide.html>
```

> I'd like to change this in the future. My preference would be to use a relative
> path, but I need to research and change how Sphinx handles relative links.
