# KGE Toolbox

## Environment

create a conda environment with `pytorch` `cython` and `scikit-learn` :
```shell
conda create --name toolbox_env python=3.7
source activate toolbox_env
conda install --file requirements.txt -c pytorch
```
## How to run

```shell
python train.py --batch_size=512 --name=TryMyModel
```

## Contributing to Toolbox
<!--- If your README is long or you have some specific process or steps you want contributors to follow, consider creating a separate CONTRIBUTING.md file--->
To contribute to toolbox, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin toolbox/<location>`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## Contributors

Thanks to the following people who have contributed to this project:

You might want to consider using something like the [All Contributors](https://github.com/all-contributors/all-contributors) specification and its [emoji key](https://allcontributors.org/docs/en/emoji-key).
