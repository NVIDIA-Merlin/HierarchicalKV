# Documentation

This folder contains the scripts necessary to build the documentation for HierarchicalKV.
You can view the generated [HierarchicalKV documentation](https://nvidia-merlin.github.io/HierarchicalKV/master/README.html).

## Contributing to Docs

Follow the instructions below to be able to build the docs.

1. Install required documentation tools and extensions:

```shell
sudo apt-get install doxygen
pip install -r docs/requirements-doc.txt
```

2. Build the documentation:

`make -C docs clean html`

The preceding command runs Sphinx in your shell and outputs to build/html/index.html.

The build process for HierarchicalKV is unique among the Merlin projects because it
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

### Rebuild the documentation on GitHub Pages

The `.github/workflows/docs-sched-rebuild.yaml` file rebuilds the documentation
for the `master` branch and the six most recent tags.  The job runs daily,
but you can trigger it manually by going to the following URL and clicking
the *Run workflow* button.

<https://github.com/NVIDIA-Merlin/HierarchicalKV/actions/workflows/docs-sched-rebuild.yaml>

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

* `../somefile.md#2heading-with-spaces-and_underscore_separated_words-too`
* `./otherfile.md#save_params_to_files-method`

#### Docs-to-docs links

There is no concern for the GitHub browsing experience for files in the `docs/source/` directory.
You can use a relative path for the link.  For example--both the `README.md` file and the
`CONTRIBUTING.md` file are copied to `docs/source`. Because they are are both in the same
directory, you could add a link to a heading in the `README.md` file like this:

```markdown
To build HierarchicalKV from scratch, refer to
[How to Build](./README.md#how-to-build) in the `README` file.
```

When Sphinx renders the link, the `.md` file suffix is replaced with `.html`.

#### Docs-to-repository links

Some files that we publish as docs, such as the `CONTRIBUTING.md` file, refer readers to files
that are not published as docs. For example, we currently do not publish the `STYLE_GUIDE.md`
file.

To refer a reader to the `STYLE_GUIDE.md`, a README, or program, state that the link is to
the repository:

```markdown
## Coding Style
Refer to the [Style Guide](http://github.com/NVIDIA-Merlin/HierarchicalKV/STYLE_GUIDE.md)
in the GitHub repository for more details.
```

The idea is to let a reader know that following the link&mdash;whether from an HTML docs page or
from browsing GitHub&mdash;results in viewing our repository on GitHub.

