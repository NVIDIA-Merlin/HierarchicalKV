# Contributing

What's the Merlin-HKVS?

- A hierarchical key-value storage library designed for the large models in recommenders systems
- Storing the key-value (embedding) on the HBM and Host memory (support SSD/NVMe in the future)
- The performance is close to those implementations running on pure HBM thru innovative design
- Can also be used as a generic key-value storage

## Maintainership

We adopt proxy maintainership as in [Merlin KV](https://github.com/NVIDIA-Merlin/merlin-kv):

*Projects and subpackages are compartmentalized and each is maintained by those
with expertise and vested interest in that component.*

*Subpackage maintainership will only be granted after substantial contribution
has been made in order to limit the number of users with write permission.
Contributions can come in the form of issue closings, bug fixes, documentation,
new code, or optimizing existing code. Submodule maintainership can be granted
with a lower barrier for entry as this will not include write permissions to
the repo.*

## Contributing

Merlin-KV is a community-led open source project. As such,
the project depends on public contributions, bug fixes, and documentation. This
project adheres to NVIDIA's Code of Conduct.

### Pull Requests
We welcome contributions via pull requests.
Before sending out a pull request, we recommend that you open an issue and
discuss your proposed change. Some changes may require a design review.
All submissions require review by project owners.

**NOTE**:
If your PR cannot be mereged, and system indicate you like "Merging is blocked,
The base branch requres all commits to be signed'
You have to configure your git and GPG key to sign your commit. [Sign your commit with GPG key](https://docs.github.com/en/github/authenticating-to-github/managing-commit-signature-verification/about-commit-signature-verification#gpg-commit-signature-verification) .

### Design Review
A new project in this repository or a significant change to an existing project
requires a design review. We recommend that you discuss your idea in the mailing
list (merlinteam@nvidia.com) before moving forward.

The centerpiece of a design review is a design doc which needs to include the following:
* Motivation of the change
* High-level design
* Detailed changes and related changes in TensorFlow
* [Optional] other alternatives that have been considered
* [Optional] testing plan
* [Optional] maintenance plan

The author needs to send out the design doc via a pull request. Project owners or
Merlin Team members will discuss proposals in a monthly meeting
or an ad-hoc design review meeting. After a proposal is approved, the author
could then start contributing the implementation.

### Coding Style
Refer to the [Style Guide](http://github.com/NVIDIA-Merlin/merlin-kv/STYLE_GUIDE.md)
in the GitHub repository for more details.

### Additional Requirements
In addition to the above requirements, contribution also needs to meet the following criteria:
* The change needs to include unit tests and integration tests if any.
* Each project needs to provide documentation for when and how to use it.

## Community

* Merlin-KV code (https://github.com/NVIDIA-Merlin/merlin-kv)

## Licence
Apache License 2.0

