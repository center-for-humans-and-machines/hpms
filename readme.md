# Hybrid Psychiatrist-in-the-Loop Monitoring System for LLM Conversational Agents

[Dataset card](./dataset-card.md)

## Requirements

Assuming macOS with [Homebrew](https://brew.sh/) installed.

- Python 3.13 or later
- [`pre-commit`](https://pre-commit.com/) - linter orchestration

  ```sh
  brew install pre-commit
  ```

- [`pylint`](https://pylint.readthedocs.io/en/stable/) - local linter

  ```sh
  brew install pylint
  ```

- [`poetry`](https://python-poetry.org/) - package manager

  ```sh
  pipx install poetry==2.1.2
  ```

- [`font-linux-libertine`](https://formulae.brew.sh/cask/font-linux-libertine) - font for plots

  ```sh
  brew install --cask font-linux-libertine
  ```

- Follow [data.qmd](./docs/data.qmd) to download the datasets

## Installation

- Install Python dependencies

  ```sh
  ./script/bootstrap
  ```

- Install Python package

  ```sh
  ./script/install
  ```

## Usage

The data from the paper is available in the [`data/paper`](./data/paper/) folder and are needed to run some of the notebooks.
There are two main scripts to run the experiments:

1. `generate_dataset.py`: generates the synthetic conversational datasets
1. `evaluate_dataset.py`: evaluates the datasets using automated metrics

Both scripts are executed in the [regression test pipeline](./.github/workflows/regression-test.yml) using GitHub Actions.

### Scripts

#### Linting

- Lint the code locally

  ```sh
  ./script/lint
  ```

#### Testing

- Show help message for the test script

  ```sh
  ./script/test -h
  ```

- Run tests using `pytest` (except for regression tests)

  ```sh
  ./script/test
  ```

- Run regression tests using `pytest`

  ```sh
  ./script/test -r
  ```

## Data Generation

1. Generate dataset:

   **Round 2**

   1. Review `.env` file to set provider and model
   1. Repeat the following process for each model

      ```bash
      poetry run python generate_dataset.py --round-number=2 --tag=acm-tist
      ```

   **Round 3**

   1. Review `.env` file to set provider and model
   1. Repeat the process for each model, but with `--round-number=3`

      ```bash
      poetry run python generate_dataset.py --round-number=3 --tag=acm-tist
      ```

## Data Evaluation

1. Evaluate dataset:

   **Round 2**

   1. Repeat the following process for each model

      ```bash
      poetry run python evaluate_dataset.py --round-number=2
      ```

   **Round 3**

   1. Repeat the process for each model, but with `--round-number=3`

      ```bash
      poetry run python evaluate_dataset.py --round-number=3
      ```

## Contributing

Please read [contributing.md](contributing.md) for details on the guidelines for this project.

## Credits

- Scripts follow [rodrigobdz/styleguide-sh](https://github.com/rodrigobdz/styleguide-sh)
- Linter configuration files imported from [rodrigobdz/linters](https://github.com/rodrigobdz/linters)
- Readme is based on [rodrigobdz/minimal-readme](https://github.com/rodrigobdz/minimal-readme)

## License

[CC-BY-4.0](license)
