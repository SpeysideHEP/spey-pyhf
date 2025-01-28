# Contributing to Spey

We are happy to accept contributions to `spey-pyhf` plug-in via
  [Pull Requests to our GitHub repository](https://github.com/SpeysideHEP/spey-pyhf/pulls).
You can begin this with forking the `main` repository.

Unless there is a very small fix that does not require any discussion, please
always first [open an issue](https://github.com/SpeysideHEP/spey-pyhf/issues/new/choose)
to discuss your request with the development team.

If the desired change is not limited to a couple of lines of code, please create
a draft pull request. This draft should detail the context of the change, its
description and the benefits of the implementation.

- If there is a change within the Python interface of the program, please
   proceed with standard tests and write extra tests if necessary.
- Please additionally make sure to add examples on how to use the new
   implementation.
- If there are any drawbacks of your implementation, these should be specified.
  Possible solutions should be offered, if any.

### Pull request procedure

Here are the steps to follow to make a pull request:

1. Fork the `spey-pyhf` repository.
2. Install pre-commit using `pip install pre-commit`
3. Go to `spey-pyhf` main folder and type `pre-commit install`.
4. Open an issue and discuss the implementation with the developers.
5. Commit your changes to a feature branch on your fork and push all your changes.
6. Start a draft pull request and inform the developers about your progress.
7. Pull the ``main`` branch to ensure there are no conflicts with the current code developments.
8. Modify the appropriate section of
   `docs/releases/changelog-dev.md`.
9. Once complete, request a review from one of the maintainers.
